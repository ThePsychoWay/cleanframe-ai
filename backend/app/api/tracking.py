"""Mask tracking: start a job, and read its progress and result.

Tracking follows the *saved* marks, so a person has to save before tracking and saving again discards
results. Everything is scoped to the signed-in user (see projects._owned_project).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import update

from app.api.editor import _load_mask
from app.api.projects import Db, _owned_project
from app.auth.deps import CurrentUser
from app.models import Mask, Project, ProjectStatus
from app.models.mask import TRACKING_DONE, TRACKING_NONE, TRACKING_RUNNING
from app.ratelimit import per_user
from app.schemas.tracking import TrackingResponse
from app.services.tracking import expire_if_lost
from app.workers.tasks import EnqueueError, enqueue_tracking

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["tracking"])


def _response(mask: Mask | None, *, with_data: bool) -> TrackingResponse:
    if mask is None:
        return TrackingResponse(status=TRACKING_NONE, frames_done=0, frames_total=0, error=None, started_at=None)
    return TrackingResponse(
        status=mask.tracking_status,
        frames_done=mask.tracking_progress,
        frames_total=mask.tracking_total,
        error=mask.tracking_error,
        started_at=mask.tracking_started_at,
        data=mask.tracking_data if with_data and mask.tracking_status == TRACKING_DONE else None,
    )


@router.post(
    "/{project_id}/track", response_model=TrackingResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[per_user("tracking_start")]
)
def start_tracking(project_id: uuid.UUID, user: CurrentUser, db: Db) -> TrackingResponse:
    project = _owned_project(db, user.id, project_id)
    if project.status == ProjectStatus.TRACKING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Tracking is already running for this project.")
    if project.status != ProjectStatus.MASKING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Tracking can only start while the video is ready for masking.")
    mask = _load_mask(db, project.id)
    if mask is None or not mask.mask_data.get("regions"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Mark at least one area and save before tracking.")

    # Claim atomically so two clicks (or two tabs) can't start two jobs.
    claimed = db.execute(
        update(Project)
        .where(Project.id == project.id, Project.status == ProjectStatus.MASKING.value)
        .values(status=ProjectStatus.TRACKING.value)
    ).rowcount
    if not claimed:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Tracking is already running for this project.")
    mask.tracking_status = TRACKING_RUNNING
    mask.tracking_progress = 0
    mask.tracking_total = 0
    mask.tracking_error = None
    mask.tracking_started_at = datetime.now(UTC)
    mask.tracking_data = None
    db.commit()
    logger.info("tracking requested", extra={"project_id": str(project.id), "user_id": str(user.id)})

    try:
        enqueue_tracking(project.id)
    except EnqueueError:
        logger.exception("couldn't enqueue tracking", extra={"project_id": str(project.id)})
        db.execute(update(Project).where(Project.id == project.id).values(status=ProjectStatus.MASKING.value))
        mask.tracking_status = TRACKING_NONE
        db.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "We couldn't start tracking. Try again in a moment."
        ) from None

    db.expire_all()  # an inline (eager) job may already have finished in another session
    return _response(_load_mask(db, project.id), with_data=False)


@router.get("/{project_id}/tracking", response_model=TrackingResponse)
def get_tracking(project_id: uuid.UUID, user: CurrentUser, db: Db) -> TrackingResponse:
    project = _owned_project(db, user.id, project_id)
    mask = _load_mask(db, project.id)
    if mask is not None:
        expire_if_lost(db, project, mask)
    return _response(mask, with_data=True)

