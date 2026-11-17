"""Additional exports of a finished video: the same restored picture at another size or in another codec.

The AI work is not repeated. The finished file is decoded once and encoded once more at the requested size, so an
export costs a transcode (seconds to a few minutes) instead of a render (minutes to hours). It can only make the
picture smaller or change its format: a larger size would only stretch the pixels that are there, so it is never
offered. The audio is copied untouched whenever the new container can hold it.

    API creates the job (queued)       a Job of type "export" next to the render jobs; the project stays COMPLETED
    worker claims it                   queued -> running
    worker finishes                    running -> completed, the file is stored under outputs/ like a render's
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db import get_sessionmaker
from app.disk import InsufficientStorage, ensure_work_space
from app.errors import RESOURCES_EXHAUSTED, ran_out_of_resources
from app.logging_config import bind_context
from app.models import Job, Project
from app.models.job import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_EXPORT,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    STAGE_PREPARING,
    STAGE_RECONSTRUCTING,
    STAGE_SAVING,
    STAGE_VERIFYING,
)
from app.services.rendering import CONTENT_TYPES, Heartbeat, _now
from app.storage.keys import output_key
from app.storage.s3 import ObjectStorage, StorageError, get_storage
from app.video.audio import AudioAction, plan_audio
from app.video.encoders import choose_encoder
from app.video.ffmpeg import FFmpegProcess, ProcessingCancelled, ProcessingError, safe_unlink
from app.video.frames import output_size
from app.video.source import inspect_source
from app.video.verify import Expectation, verify_output

logger = logging.getLogger(__name__)

EXPORT_TIME_LIMIT_SECONDS = 3600
GENERIC_FAILURE = "The export couldn't be made. Please try again."


def export_filename_suffix(settings: dict[str, Any], result_height: int) -> str:
    height = settings.get("height") or result_height
    codec = settings.get("codec") or ""
    return f"{height}p" + (f"-{codec}" if codec and codec not in ("h264", "vp9") else "")


def _transcode(source: Path, destination: Path, *, settings: dict[str, Any], on_progress: Any) -> dict[str, Any]:
    """Re-encode `source` once. Returns what was made. `on_progress(frame)` returns False to stop."""
    media = inspect_source(source)
    container = str(settings.get("container", "mp4"))
    quality = str(settings.get("quality", "balanced"))
    encoder = choose_encoder(container, settings.get("codec"), quality, get_settings().video_encoder)
    out_width, out_height = output_size(media.even_size[0], media.even_size[1], settings.get("height"))
    color = media.color
    action, audio_codec = plan_audio(media, container, True)
    audio_args: list[str]
    if action is AudioAction.NONE:
        audio_args = ["-an"]
    elif action is AudioAction.COPIED:
        audio_args = ["-map", "0:a", "-c:a", "copy"]
    else:
        audio_args = ["-map", "0:a", "-c:a", "aac" if container == "mp4" else "libopus", "-b:a", "192k" if container == "mp4" else "160k"]
    scale = (
        f"scale={out_width}:{out_height}:flags=lanczos+accurate_rnd+full_chroma_int:"
        f"in_range={color.range}:out_color_matrix={color.scale_matrix}:out_range={color.range},format=yuv420p"
    )
    tags = ["-colorspace", color.matrix, "-color_primaries", color.primaries, "-color_trc", color.transfer, "-color_range", color.range]
    container_args = ["-movflags", "+faststart", "-f", "mp4"] if container == "mp4" else ["-f", "webm"]
    process = FFmpegProcess(
        [
            "-i", f"file:{source}", "-map", "0:v:0", "-vf", scale, "-fps_mode", "passthrough",
            *encoder.args, *tags, *audio_args, "-map_metadata", "-1", *container_args,
            "-progress", "pipe:1", "-nostats", f"file:{destination}",
        ],
        stdout=True,
    )  # fmt: skip
    try:
        started = time.monotonic()
        for raw in iter(process.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("frame="):
                try:
                    keep_going = on_progress(int(line.partition("=")[2]))
                except ValueError:
                    continue
                if keep_going is False:
                    raise ProcessingCancelled
            if time.monotonic() - started > EXPORT_TIME_LIMIT_SECONDS:
                raise ProcessingError("The export took too long and was stopped.")
        code = process.process.wait()
        if code != 0:
            logger.warning("export encode failed (%s): %s", code, process.error_tail())
            raise ProcessingError("The video couldn't be encoded at that size. Please try another option.")
    except BaseException:
        process.kill()
        safe_unlink(destination)
        raise
    finally:
        process.close()
    return {
        "width": out_width, "height": out_height, "fps": media.fps, "container": container, "codec": encoder.codec,
        "encoder": encoder.name, "hardware_encoder": encoder.hardware, "audio": action.value, "audio_codec": audio_codec,
        "source_width": media.even_size[0], "source_height": media.even_size[1],
    }  # fmt: skip


def run_export(
    job_id: uuid.UUID,
    *,
    storage: ObjectStorage | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> None:
    settings = get_settings()
    storage = storage or get_storage()
    session_factory = session_factory or get_sessionmaker()

    with session_factory() as db:
        claimed = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.type == JOB_EXPORT, Job.status == JOB_QUEUED, Job.cancel_requested.is_(False))
            .values(status=JOB_RUNNING, stage=STAGE_PREPARING, started_at=_now(), heartbeat_at=_now())
        ).rowcount
        db.commit()
        if not claimed:
            logger.info("export skipped: job is not waiting", extra={"job_id": str(job_id)})
            return
        job = db.get(Job, job_id)
        assert job is not None
        project = db.get(Project, job.project_id)
        project_id, user_id, request = job.project_id, job.user_id, dict(job.settings)
        source_key = project.output_file_key if project is not None else None
        source_job = request.get("source_job_id")
        if project is None or not source_key or source_key != request.get("source_key"):
            _finish(session_factory, job_id, JOB_FAILED, "The finished video changed or is gone, so this export was dropped. Make it again.")
            return

    bind_context(job_id=job_id, project_id=project_id, user_id=user_id)
    logger.info("export started", extra={"request": request, "source_job": source_job})
    started = time.monotonic()
    new_key: str | None = None
    state = {"last": 0.0}

    def progress(frame: int) -> bool:
        now = time.monotonic()
        if now - state["last"] < 1.0:
            return True
        state["last"] = now
        with session_factory() as db:
            current = db.get(Job, job_id)
            if current is None or current.status != JOB_RUNNING or current.cancel_requested:
                return False
            db.execute(update(Job).where(Job.id == job_id).values(stage=STAGE_RECONSTRUCTING, frames_done=min(frame, current.frames_total), heartbeat_at=_now()))
            db.commit()
        return True

    try:
        with Heartbeat(session_factory, job_id, settings.job_heartbeat_seconds):
            size_hint = storage.object_size(source_key) or 0
            ensure_work_space(size_hint, factor=2.5, what="this export")
            container = str(request.get("container", "mp4"))
            with tempfile.TemporaryDirectory(prefix="cf-export-", dir=settings.worker_tmp_dir or None) as tmp:
                local = Path(tmp) / f"result.{request.get('source_container', 'mp4')}"
                storage.download_to(source_key, local)
                made = _transcode(local, Path(tmp) / f"export.{container}", settings=request, on_progress=progress)
                done = Path(tmp) / f"export.{container}"
                with session_factory() as db:
                    db.execute(update(Job).where(Job.id == job_id).values(stage=STAGE_VERIFYING, heartbeat_at=_now()))
                    db.commit()
                media_frames = int(request["frames"])
                report = verify_output(
                    done,
                    Expectation(
                        width=made["width"], height=made["height"], frames=media_frames, fps=float(made["fps"]),
                        expect_audio=made["audio"] in ("copied", "converted"), source_audio_gap=None,
                    ),
                )  # fmt: skip
                new_key = output_key(user_id, project_id, job_id, container)
                with session_factory() as db:
                    db.execute(update(Job).where(Job.id == job_id).values(stage=STAGE_SAVING, heartbeat_at=_now()))
                    db.commit()
                size = done.stat().st_size
                storage.upload_file(done, new_key, CONTENT_TYPES[container])
    except ProcessingCancelled:
        _discard(storage, new_key)
        _finish(session_factory, job_id, JOB_CANCELLED, None)
        return
    except Exception as error:
        _discard(storage, new_key)
        logger.exception("export failed")
        _finish(session_factory, job_id, JOB_FAILED, _message_for(error))
        return

    result = {**made, "frames": report.frames, "size_bytes": size, "export_seconds": round(time.monotonic() - started, 1)}
    with session_factory() as db:
        saved = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == JOB_RUNNING, Job.cancel_requested.is_(False))
            .values(
                status=JOB_COMPLETED, stage=STAGE_SAVING, completed_at=_now(), error=None, result=result,
                output_file_key=new_key, frames_done=report.frames, frames_total=report.frames, warnings=report.warnings,
            )
        ).rowcount  # fmt: skip
        db.commit()
    if not saved:
        _discard(storage, new_key)
        _finish(session_factory, job_id, JOB_CANCELLED, None)
        return
    logger.info("export finished", extra={"duration_ms": int((time.monotonic() - started) * 1000), "bytes": size})


def _finish(session_factory: sessionmaker[Session], job_id: uuid.UUID, status: str, error: str | None) -> None:
    with session_factory() as db:
        db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status.in_((JOB_QUEUED, JOB_RUNNING)))
            .values(status=status, error=error[:500] if error else None, completed_at=_now())
        )
        db.commit()


def _discard(storage: ObjectStorage, key: str | None) -> None:
    if key:
        try:
            storage.delete(key)
        except Exception:
            logger.warning("couldn't remove an unneeded file", extra={"key": key}, exc_info=True)


def _message_for(error: BaseException) -> str:
    if isinstance(error, InsufficientStorage):
        return error.message
    if isinstance(error, ProcessingError) and not ran_out_of_resources(error):
        return str(error)
    if ran_out_of_resources(error):
        return RESOURCES_EXHAUSTED
    if isinstance(error, StorageError):
        return "The video file couldn't be read or saved. Please try again."
    return GENERIC_FAILURE


def drop_exports_of(db: Session, storage: ObjectStorage, project_id: uuid.UUID) -> None:
    """A new render replaced the finished video, so exports made from the old one no longer match it: remove them."""
    for job in list(db.scalars(select(Job).where(Job.project_id == project_id, Job.type == JOB_EXPORT))):
        if job.status in (JOB_QUEUED, JOB_RUNNING):
            job.cancel_requested = True
            continue
        if job.output_file_key:
            _discard(storage, job.output_file_key)
        db.delete(job)
