"""Runs mask tracking for a project and records the outcome.

State machine (Project.status / Mask.tracking_status):

    MASKING / none|done|failed  --POST /track-->  TRACKING / running
    TRACKING / running  --finished-->  MASKING / done
    TRACKING / running  --error-->     MASKING / failed (+ a message the person can act on)

The project returns to MASKING either way, so the person can always keep editing. Saving the mask
clears results (see api/editor.py), because they describe marks that no longer exist.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db import get_sessionmaker
from app.errors import RESOURCES_EXHAUSTED, ran_out_of_resources
from app.logging_config import bind_context
from app.models import Mask, Project, ProjectStatus
from app.models.mask import TRACKING_DONE, TRACKING_FAILED, TRACKING_RUNNING
from app.schemas.mask import MaskDocument
from app.storage.s3 import ObjectStorage, get_storage
from app.tracking.tracker import TrackingCancelled, track_document

logger = logging.getLogger(__name__)

GENERIC_FAILURE = "Tracking couldn't finish. Please try again."
TIMED_OUT = "Tracking took too long and was stopped. Try a shorter clip or fewer marked areas."
WORKER_LOST = "Tracking didn't finish (the worker stopped). Please try again."
# The worker's hard limit is the soft limit + 120 s (see workers/tasks.py); this is comfortably past it,
# so a job that is still alive is never declared lost.
STALE_MARGIN_SECONDS = 180


def _fail(session_factory: sessionmaker[Session], project_id: uuid.UUID, mask_id: uuid.UUID, message: str) -> None:
    with session_factory() as db:
        db.execute(
            update(Mask)
            .where(Mask.id == mask_id, Mask.tracking_status == TRACKING_RUNNING)
            .values(tracking_status=TRACKING_FAILED, tracking_error=message[:500])
        )
        db.execute(
            update(Project)
            .where(Project.id == project_id, Project.status == ProjectStatus.TRACKING.value)
            .values(status=ProjectStatus.MASKING.value)
        )
        db.commit()


def run_tracking(
    project_id: uuid.UUID,
    *,
    storage: ObjectStorage | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> None:
    storage = storage or get_storage()
    session_factory = session_factory or get_sessionmaker()
    started = time.monotonic()

    with session_factory() as db:
        project = db.get(Project, project_id)
        mask = db.scalar(select(Mask).where(Mask.project_id == project_id))
        if (
            project is None
            or mask is None
            or project.status != ProjectStatus.TRACKING.value
            or mask.tracking_status != TRACKING_RUNNING
        ):
            logger.info("tracking skipped: nothing waiting", extra={"project_id": str(project_id)})
            return
        if not (project.proxy_file_key and project.proxy_width and project.proxy_height and project.frame_count):
            _fail(session_factory, project_id, mask.id, "The preview for this project isn't ready.")
            return
        proxy_object = project.proxy_file_key
        width, height, frame_count = project.proxy_width, project.proxy_height, project.frame_count
        document = MaskDocument.model_validate(mask.mask_data)
        mask_id = mask.id
        user_id = str(project.user_id)

    bind_context(project_id=project_id, user_id=user_id)
    logger.info("tracking started", extra={"regions": len(document.regions)})

    def progress(done: int, total: int) -> bool:
        """Record progress. False means the mask is gone or no longer running: stop wasting the worker."""
        with session_factory() as db:
            updated = db.execute(
                update(Mask)
                .where(Mask.id == mask_id, Mask.tracking_status == TRACKING_RUNNING)
                .values(tracking_progress=done, tracking_total=total)
            ).rowcount
            db.commit()
        return bool(updated)

    try:
        with tempfile.TemporaryDirectory(prefix="cf-tracking-") as tmp:
            local = Path(tmp) / "proxy.mp4"
            storage.download_to(proxy_object, local)
            data = track_document(
                document, local, width=width, height=height, frame_count=frame_count, progress=progress
            )
    except TrackingCancelled:
        logger.info("tracking stopped: project or marks changed", extra={"project_id": str(project_id)})
        return
    except Exception as error:
        # SoftTimeLimitExceeded (Celery) is an Exception subclass too.
        timed_out = type(error).__name__ == "SoftTimeLimitExceeded"
        logger.exception("tracking failed", extra={"duration_ms": int((time.monotonic() - started) * 1000)})
        reason = TIMED_OUT if timed_out else RESOURCES_EXHAUSTED if ran_out_of_resources(error) else GENERIC_FAILURE
        _fail(session_factory, project_id, mask_id, reason)
        return

    lost = sum(
        1
        for region in data["regions"].values()
        for segment in region["segments"]
        if segment.get("lost_at") is not None
    )
    with session_factory() as db:
        saved = db.execute(
            update(Mask)
            .where(Mask.id == mask_id, Mask.tracking_status == TRACKING_RUNNING)
            .values(
                tracking_status=TRACKING_DONE,
                tracking_data=data,
                tracking_error=None,
                tracking_progress=Mask.tracking_total,
            )
        ).rowcount
        db.execute(
            update(Project)
            .where(Project.id == project_id, Project.status == ProjectStatus.TRACKING.value)
            .values(status=ProjectStatus.MASKING.value)
        )
        db.commit()
    logger.info(
        "tracking finished" if saved else "tracking result discarded: marks changed",
        extra={
            "project_id": str(project_id),
            "segments_lost": lost,
            "duration_ms": int((time.monotonic() - started) * 1000),
        },
    )


def expire_if_lost(db: Session, project: Project, mask: Mask) -> bool:
    """A job still 'running' long after the worker's hard limit means the worker died. Report it failed."""
    if mask.tracking_status != TRACKING_RUNNING or mask.tracking_started_at is None:
        return False
    started = mask.tracking_started_at
    if started.tzinfo is None:  # SQLite hands back naive datetimes
        started = started.replace(tzinfo=UTC)
    limit = get_settings().tracking_time_limit_seconds + STALE_MARGIN_SECONDS
    if (datetime.now(UTC) - started).total_seconds() < limit:
        return False
    mask.tracking_status = TRACKING_FAILED
    mask.tracking_error = WORKER_LOST
    if project.status == ProjectStatus.TRACKING.value:
        project.status = ProjectStatus.MASKING.value
    db.commit()
    logger.warning("tracking marked lost", extra={"project_id": str(project.id)})
    return True
