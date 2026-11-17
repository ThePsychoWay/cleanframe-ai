"""Runs a render job on a worker and keeps its record true.

    API creates the job (queued) and the project goes MASKING|COMPLETED -> PROCESSING
    worker claims it        queued  -> running          (only one worker can: the claim is a conditional UPDATE)
    worker finishes         running -> completed        project -> COMPLETED, output stored
    worker fails            running -> failed           project -> back to what it was (MASKING, or COMPLETED if it has an earlier result)
    person cancels          queued|running -> cancelled project -> same as failed

Everything written to the job is something that happened: `stage` is set when the worker reaches it, `frames_done`
when a frame came out of the engine. A heartbeat (a thread, so long steps with no frames still beat) lets the API tell a
slow job from a dead worker. Messages in `error` are written for the person; details go to the log.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker, undefer

from app.ai.base import InpaintingEngine, InpaintingError, ModelMemoryExhausted
from app.ai.registry import get_engine
from app.config import get_settings
from app.db import get_sessionmaker
from app.disk import InsufficientStorage, ensure_work_space
from app.errors import RESOURCES_EXHAUSTED, ran_out_of_resources
from app.logging_config import bind_context
from app.models import Job, Mask, Project, ProjectStatus
from app.models.job import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    STAGE_PREPARING,
    STAGE_RECONSTRUCTING,
    STAGE_SAVING,
)
from app.models.mask import TRACKING_DONE
from app.schemas.mask import MaskDocument
from app.services.cleanup import extend_retention
from app.services.quality import check_result, warnings_from
from app.storage.keys import output_key
from app.storage.s3 import ObjectStorage, StorageError, get_storage
from app.video.ffmpeg import ProcessingCancelled, ProcessingError
from app.video.pipeline import RenderSettings, render_restored_video

logger = logging.getLogger(__name__)

GENERIC_FAILURE = "Rendering couldn't finish. Please try again."
TIMED_OUT = "Rendering took too long and was stopped. Try a shorter clip, a lower quality, or fewer marked areas."
STORAGE_FAILURE = "The video file couldn't be read or saved. Please try again."
WORKER_LOST = "Rendering didn't finish (the worker stopped). Please try again."
NOT_PICKED_UP = "No worker picked this job up. Please try again, or contact support if it keeps happening."
PROGRESS_INTERVAL_SECONDS = 1.0
CONTENT_TYPES = {"mp4": "video/mp4", "webm": "video/webm"}


def _now() -> datetime:
    return datetime.now(UTC)


def aware(value: datetime | None) -> datetime | None:
    """SQLite hands datetimes back naive; they were stored as UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def restore_project_status(db: Session, project_id: uuid.UUID) -> None:
    """After a render that didn't produce a result, the project goes back to being editable: COMPLETED if an
    earlier result still exists, otherwise MASKING."""
    project = db.get(Project, project_id)
    if project is not None and project.status == ProjectStatus.PROCESSING.value:
        project.status = (ProjectStatus.COMPLETED if project.output_file_key else ProjectStatus.MASKING).value


def _finish_without_result(
    session_factory: sessionmaker[Session], job_id: uuid.UUID, project_id: uuid.UUID, status: str, error: str | None
) -> None:
    with session_factory() as db:
        db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status.in_((JOB_QUEUED, JOB_RUNNING)))
            .values(status=status, error=error[:500] if error else None, completed_at=_now())
        )
        restore_project_status(db, project_id)
        db.commit()


class Heartbeat:
    """Writes `heartbeat_at` on a timer while a job runs, so a long step with no progress still shows the worker is alive."""

    def __init__(self, session_factory: sessionmaker[Session], job_id: uuid.UUID, interval: float) -> None:
        self._factory, self._job_id, self._interval = session_factory, job_id, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{job_id}", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                with self._factory() as db:
                    db.execute(update(Job).where(Job.id == self._job_id, Job.status == JOB_RUNNING).values(heartbeat_at=_now()))
                    db.commit()
            except Exception:  # a missed beat is not worth killing the render over
                logger.warning("heartbeat failed", exc_info=True)

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


class _Progress:
    """Turns the pipeline's progress calls into job updates, at most once a second, and carries the stop signal back."""

    def __init__(self, session_factory: sessionmaker[Session], job_id: uuid.UUID) -> None:
        self._factory, self._job_id = session_factory, job_id
        self._last = 0.0
        self._stage: str | None = None

    def __call__(self, stage: str, done: int, total: int) -> bool:
        now = time.monotonic()
        changed = stage != self._stage
        finished = total > 0 and done >= total
        if not (changed or finished or now - self._last >= PROGRESS_INTERVAL_SECONDS):
            return True
        self._last, self._stage = now, stage
        values: dict[str, Any] = {"stage": stage, "heartbeat_at": _now()}
        if stage == STAGE_RECONSTRUCTING:
            values.update(frames_done=done, frames_total=total)
            if changed:
                values["reconstruct_started_at"] = _now()
        with self._factory() as db:
            # Only a running job that nobody asked to stop takes the update; no row updated means stop.
            updated = db.execute(
                update(Job).where(Job.id == self._job_id, Job.status == JOB_RUNNING, Job.cancel_requested.is_(False)).values(**values)
            ).rowcount
            db.commit()
        return bool(updated)

    def still_wanted(self) -> bool:
        with self._factory() as db:
            job = db.get(Job, self._job_id)
            return bool(job and job.status == JOB_RUNNING and not job.cancel_requested)


def run_render(
    job_id: uuid.UUID,
    *,
    storage: ObjectStorage | None = None,
    session_factory: sessionmaker[Session] | None = None,
    engine: InpaintingEngine | None = None,
) -> None:
    settings = get_settings()
    storage = storage or get_storage()
    session_factory = session_factory or get_sessionmaker()

    # Claim. Only one worker can move a job from queued to running, so a message delivered twice does nothing.
    with session_factory() as db:
        claimed = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == JOB_QUEUED, Job.cancel_requested.is_(False))
            .values(status=JOB_RUNNING, stage=STAGE_PREPARING, started_at=_now(), heartbeat_at=_now())
        ).rowcount
        db.commit()
        if not claimed:
            logger.info("render skipped: job is not waiting", extra={"job_id": str(job_id)})
            return
        job = db.get(Job, job_id)
        assert job is not None
        project = db.get(Project, job.project_id)
        mask = db.scalar(select(Mask).options(undefer(Mask.tracking_data)).where(Mask.project_id == job.project_id))
        project_id, user_id = job.project_id, job.user_id
        if project is None or mask is None or project.status != ProjectStatus.PROCESSING.value:
            _finish_without_result(session_factory, job_id, project_id, JOB_CANCELLED, None)
            logger.info("render stopped: project is gone or changed", extra={"job_id": str(job_id)})
            return
        render_settings = job.settings
        mask_source = mask.source
        original_key, expected_frames, project_size = project.original_file_key, project.frame_count, project.size_bytes
        document = MaskDocument.model_validate(mask.mask_data)
        tracking = mask.tracking_data if job.tracking_used and mask.tracking_status == TRACKING_DONE else None

    bind_context(job_id=job_id, project_id=project_id, user_id=user_id)
    logger.info("render started")
    started = time.monotonic()
    progress = _Progress(session_factory, job_id)
    new_key: str | None = None
    try:
        with Heartbeat(session_factory, job_id, settings.job_heartbeat_seconds):
            if not expected_frames:
                raise ProcessingError("This project hasn't been analysed yet.")
            engine = engine or get_engine()
            engine.require_available()
            container = str(render_settings.get("container", "mp4"))
            # The original, the picture being encoded, the picture with its audio and the stored result can exist together.
            ensure_work_space(project_size, factor=3.5, what="this video")
            with tempfile.TemporaryDirectory(prefix="cf-render-", dir=settings.worker_tmp_dir or None) as tmp:
                local = Path(tmp) / "original"
                storage.download_to(original_key, local)
                if not progress.still_wanted():
                    raise ProcessingCancelled
                result = render_restored_video(
                    local,
                    Path(tmp) / f"restored.{container}",
                    document=document,
                    tracking=tracking,
                    engine=engine,
                    work_dir=Path(tmp),
                    settings=RenderSettings(
                        quality=render_settings.get("quality", "balanced"),
                        container=container,
                        output_height=render_settings.get("output_height"),
                        codec=render_settings.get("codec"),
                        keep_audio=bool(render_settings.get("keep_audio", True)),
                    ),
                    expected_frames=expected_frames,
                    progress=progress,
                )
                quality = _quality(local, result.path, document, tracking, expected_frames)
                if progress(STAGE_SAVING, 0, 1) is False:
                    raise ProcessingCancelled
                new_key = output_key(user_id, project_id, job_id, container)
                size = result.path.stat().st_size
                storage.upload_file(result.path, new_key, CONTENT_TYPES[container])
    except ProcessingCancelled:
        _discard(storage, new_key)
        _finish_without_result(session_factory, job_id, project_id, JOB_CANCELLED, None)
        logger.info("render cancelled", extra={"duration_ms": int((time.monotonic() - started) * 1000)})
        return
    except Exception as error:
        _discard(storage, new_key)
        message = _message_for(error)
        logger.exception("render failed", extra={"duration_ms": int((time.monotonic() - started) * 1000)})
        _finish_without_result(session_factory, job_id, project_id, JOB_FAILED, message)
        return

    result_info = {
        "width": result.width, "height": result.height, "frames": result.frames, "fps": result.fps,
        "container": result.container, "codec": result.codec, "encoder": result.encoder, "hardware_encoder": result.hardware_encoder,
        "source_width": result.source_width, "source_height": result.source_height, "size_bytes": size, "audio": result.audio.value, "audio_codec": result.audio_codec,
        "tracking_used": result.tracking_used, "render_seconds": round(result.seconds, 1),
        "quality": quality, "automatic": mask_source == "auto",
    }  # fmt: skip
    old_key: str | None = None
    with session_factory() as db:
        saved = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == JOB_RUNNING, Job.cancel_requested.is_(False))
            .values(
                status=JOB_COMPLETED, stage=STAGE_SAVING, completed_at=_now(), error=None, result=result_info,
                warnings=[*result.warnings, *warnings_from(quality)], output_file_key=new_key, frames_done=result.frames, frames_total=result.frames,
            )
        ).rowcount  # fmt: skip
        project = db.get(Project, project_id)
        if saved and project is not None:
            old_key = project.output_file_key
            project.output_file_key = new_key
            project.status = ProjectStatus.COMPLETED.value
            extend_retention(project)  # the result stays downloadable for the full retention period
        db.commit()
    if not saved or project is None:
        # Cancelled or deleted while the file was being saved: nothing should point at it.
        _discard(storage, new_key)
        _finish_without_result(session_factory, job_id, project_id, JOB_CANCELLED, None)
        return
    if old_key and old_key != new_key:
        _discard(storage, old_key)
        from app.services.exporting import drop_exports_of

        with session_factory() as db:  # exports were made from the video that was just replaced
            drop_exports_of(db, storage, project_id)
            db.commit()
    logger.info(
        "render finished",
        extra={"duration_ms": int((time.monotonic() - started) * 1000), "frames": result.frames},
    )


def _quality(original: Path, restored: Path, document: MaskDocument, tracking: dict[str, Any] | None, frames: int) -> dict[str, Any]:
    """The result looked at after the fact. A check that fails to run is recorded as not run, never as passed."""
    try:
        return check_result(original, restored, document, tracking, frames)
    except Exception:
        logger.exception("quality check failed to run")
        return {"checked": False, "reason": "the check itself hit an error", "regions": []}


def _discard(storage: ObjectStorage, key: str | None) -> None:
    if key:
        try:
            storage.delete(key)
        except Exception:
            logger.warning("couldn't remove an unneeded file", extra={"key": key}, exc_info=True)


def _message_for(error: BaseException) -> str:
    if isinstance(error, InsufficientStorage):
        return error.message
    if isinstance(error, ModelMemoryExhausted):
        return str(error)  # names the video's size and what ran out; the resource wording below would hide that
    if isinstance(error, ProcessingError | InpaintingError) and not ran_out_of_resources(error):
        return str(error)  # written for people by the pipeline and the engines
    if ran_out_of_resources(error):
        return RESOURCES_EXHAUSTED
    if type(error).__name__ == "SoftTimeLimitExceeded":
        return TIMED_OUT
    if isinstance(error, StorageError):
        return STORAGE_FAILURE
    return GENERIC_FAILURE


def expire_if_lost(db: Session, job: Job) -> bool:
    """Report a job failed when its worker has plainly gone: a running job whose heartbeat stopped, or a queued job
    nobody picked up. Called when the job is read, so no scheduler is needed."""
    settings = get_settings()
    if job.status == JOB_RUNNING:
        last = aware(job.heartbeat_at) or aware(job.started_at)
        if last is None or (_now() - last).total_seconds() < settings.job_lost_after_seconds:
            return False
        message = WORKER_LOST
    elif job.status == JOB_QUEUED:
        created = aware(job.created_at)
        if created is None or (_now() - created).total_seconds() < settings.job_queue_timeout_seconds:
            return False
        message = NOT_PICKED_UP
    else:
        return False
    updated = db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status == job.status)
        .values(status=JOB_FAILED, error=message, completed_at=_now())
    ).rowcount
    if updated:
        restore_project_status(db, job.project_id)
    db.commit()
    db.refresh(job)
    if updated:
        logger.warning("job marked lost", extra={"job_id": str(job.id)})
    return bool(updated)
