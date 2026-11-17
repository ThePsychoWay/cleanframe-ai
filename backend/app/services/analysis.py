"""Analyses an uploaded video: downloads it, probes its real properties and records the result."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db import get_sessionmaker
from app.errors import RESOURCES_EXHAUSTED, STORAGE_UNAVAILABLE, ran_out_of_resources
from app.logging_config import bind_context
from app.models import Project, ProjectStatus
from app.schemas.mask import MaskDocument
from app.security.malware import MalwareFound, ScannerUnavailable, get_scanner
from app.disk import InsufficientStorage, ensure_work_space
from app.services.automation import STATE_FAILED, apply_detection, detect_project, start_render
from app.storage.keys import proxy_key, thumbnail_key
from app.storage.s3 import ObjectStorage, StorageError, get_storage
from app.video.probe import VideoRejected, parse_video_info, run_ffprobe
from app.video.proxy import MediaPreparationError, build_proxy, build_thumbnail

logger = logging.getLogger(__name__)

MALWARE_MESSAGE = "This file was blocked because the security scan flagged it. It was not processed."
SCAN_UNAVAILABLE_MESSAGE = "We couldn't run the security scan on this file just now, so it was not processed. Please try again in a few minutes."
GENERIC_FAILURE = "We couldn't analyze this video. Please try again, or upload another copy."


def _detect(local: Path, frame_count: int) -> tuple[MaskDocument | None, dict[str, Any]]:
    """Detection never fails an analysis: if it can't run, the project waits for a person to mark, and says why."""
    try:
        return detect_project(local, frame_count)
    except Exception:
        logger.exception("automatic detection failed")
        return None, {"state": STATE_FAILED, "found": 0, "removing": 0, "overlays": [], "notes": ["Automatic detection couldn't run on this video."]}


def analyze_project(
    project_id: uuid.UUID,
    *,
    storage: ObjectStorage | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> None:
    """UPLOADED -> ANALYZING -> MASKING (ready), or FAILED with a message the person can act on."""
    storage = storage or get_storage()
    session_factory = session_factory or get_sessionmaker()
    settings = get_settings()
    started = time.monotonic()

    with session_factory() as db:
        # Claim the project atomically so a duplicate or retried task can't analyse it twice.
        claimed = db.execute(
            update(Project)
            .where(Project.id == project_id, Project.status == ProjectStatus.UPLOADED.value)
            .values(status=ProjectStatus.ANALYZING.value, error=None)
        ).rowcount
        db.commit()
        if not claimed:
            logger.info("analysis skipped: project not waiting for analysis", extra={"project_id": str(project_id)})
            return

        project = db.get(Project, project_id)
        if project is None:  # deleted between claim and load
            return
        bind_context(project_id=project_id, user_id=project.user_id)
        logger.info("analysis started")

        detection: tuple[MaskDocument | None, dict[str, Any]] | None = None
        start_automatically = False
        try:
            ensure_work_space(project.size_bytes, factor=1.5, what="this video")  # the copy being analysed and its preview
            with tempfile.TemporaryDirectory(prefix="cf-analysis-", dir=settings.worker_tmp_dir or None) as tmp:
                # The local name is fixed; nothing from the person's file name reaches the filesystem.
                local = Path(tmp) / f"input{Path(project.original_file_key).suffix}"
                storage.download_to(project.original_file_key, local)
                scanner = get_scanner(settings)
                if scanner is not None:
                    scanner.scan_file(local)  # before ffprobe or FFmpeg ever open the file
                info = parse_video_info(
                    run_ffprobe(local),
                    local.suffix,
                    max_duration=settings.max_video_duration_seconds,
                    max_pixels=settings.max_video_pixels,
                )

                # The original is already here, so make the editor's preview in the same pass.
                proxy_path = Path(tmp) / "proxy.mp4"
                thumbnail_path = Path(tmp) / "thumbnail.jpg"
                proxy = build_proxy(local, proxy_path)
                build_thumbnail(proxy_path, thumbnail_path, duration=info.duration)
                proxy_object = proxy_key(project.user_id, project.id)
                thumbnail_object = thumbnail_key(project.user_id, project.id)
                storage.upload_file(proxy_path, proxy_object, "video/mp4")
                storage.upload_file(thumbnail_path, thumbnail_object, "image/jpeg")

                # Fully automatic mode: look for the overlays now, while the original is on local disk.
                if settings.auto_process:
                    detection = _detect(local, proxy.frame_count)

            project.proxy_file_key = proxy_object
            project.thumbnail_file_key = thumbnail_object
            project.proxy_width = proxy.width
            project.proxy_height = proxy.height
            project.container = info.container
            project.codec = info.codec
            project.audio_codec = info.audio_codec
            project.width = info.width
            project.height = info.height
            # The preview's constant-rate timeline is what masks and tracking refer to, so it decides the frame
            # count and rate. They match the source for constant-rate video.
            project.fps = proxy.fps
            project.duration = info.duration
            project.bitrate = info.bitrate
            project.frame_count = proxy.frame_count
            project.status = ProjectStatus.MASKING.value
            project.error = None
            if detection is not None:
                start_automatically = apply_detection(db, project, *detection)
            logger.info(
                "analysis completed",
                extra={"project_id": str(project_id), "duration_ms": int((time.monotonic() - started) * 1000)},
            )
        except MalwareFound as found:
            project.status = ProjectStatus.FAILED.value
            project.error = MALWARE_MESSAGE
            logger.warning("upload blocked by the malware scan", extra={"project_id": str(project_id), "signature": found.signature})
        except ScannerUnavailable as unavailable:
            project.status = ProjectStatus.FAILED.value
            project.error = SCAN_UNAVAILABLE_MESSAGE
            logger.error("malware scanner unavailable; upload not processed", extra={"project_id": str(project_id), "reason": str(unavailable)})
        except VideoRejected as rejected:
            project.status = ProjectStatus.FAILED.value
            project.error = rejected.message[:500]
            logger.info("analysis rejected video", extra={"project_id": str(project_id)})
        except InsufficientStorage as full:
            project.status = ProjectStatus.FAILED.value
            project.error = full.message[:500]
            logger.error("analysis stopped: not enough disk space")
        except MediaPreparationError as unprepared:
            project.status = ProjectStatus.FAILED.value
            project.error = unprepared.message[:500]
            logger.warning("preview generation failed", extra={"project_id": str(project_id)})
        except Exception as error:
            project.status = ProjectStatus.FAILED.value
            project.error = (
                RESOURCES_EXHAUSTED if ran_out_of_resources(error) else STORAGE_UNAVAILABLE if isinstance(error, StorageError) else GENERIC_FAILURE
            )
            logger.exception("analysis failed", extra={"duration_ms": int((time.monotonic() - started) * 1000)})
        db.commit()
        if start_automatically and project.status == ProjectStatus.MASKING.value:
            job = start_render(db, project)
            if job is None:
                db.commit()  # the reason was written to project.automation
            else:
                logger.info("automatic render queued", extra={"job_id": str(job.id), "project_id": str(project_id)})
