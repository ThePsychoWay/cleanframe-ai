"""Editor endpoints: signed preview links and the project's mask document.

Everything is scoped to the signed-in user; another person's project is indistinguishable from one
that doesn't exist (see projects._owned_project).
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.projects import Db, Storage, _owned_project
from app.auth.deps import CurrentUser
from app.models import Mask, ProjectStatus
from app.models.mask import TRACKING_NONE
from app.ratelimit import per_user
from app.schemas.mask import MaskDocument, MaskResponse
from app.schemas.project import MediaResponse
from app.storage.s3 import PRESIGN_TTL_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["editor"])


@router.get("/{project_id}/media", response_model=MediaResponse)
def get_media(project_id: uuid.UUID, user: CurrentUser, db: Db, storage: Storage) -> MediaResponse:
    project = _owned_project(db, user.id, project_id)
    if (
        project.proxy_file_key is None
        or project.thumbnail_file_key is None
        or project.proxy_width is None
        or project.proxy_height is None
        or project.fps is None
        or project.frame_count is None
        or project.duration is None
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "The preview for this project isn't ready yet.")

    return MediaResponse(
        proxy_url=storage.presign_get(project.proxy_file_key, expires_in=PRESIGN_TTL_SECONDS),
        thumbnail_url=storage.presign_get(project.thumbnail_file_key, expires_in=PRESIGN_TTL_SECONDS),
        expires_in=PRESIGN_TTL_SECONDS,
        width=project.proxy_width,
        height=project.proxy_height,
        fps=project.fps,
        frame_count=project.frame_count,
        duration=project.duration,
    )


def _load_mask(db: Session, project_id: uuid.UUID) -> Mask | None:
    return db.scalar(select(Mask).where(Mask.project_id == project_id))


@router.get("/{project_id}/mask", response_model=MaskResponse)
def get_mask(project_id: uuid.UUID, user: CurrentUser, db: Db) -> MaskResponse:
    project = _owned_project(db, user.id, project_id)
    mask = _load_mask(db, project.id)
    if mask is None:
        return MaskResponse(mask_data=MaskDocument(), tracking_status="none", updated_at=None)
    return MaskResponse(
        mask_data=MaskDocument.model_validate(mask.mask_data),
        tracking_status=mask.tracking_status,
        updated_at=mask.updated_at,
    )


@router.put("/{project_id}/mask", response_model=MaskResponse, dependencies=[per_user("mask_save")])
def save_mask(project_id: uuid.UUID, document: MaskDocument, user: CurrentUser, db: Db) -> MaskResponse:
    project = _owned_project(db, user.id, project_id)
    if project.status != ProjectStatus.MASKING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Masks can only be edited while the video is ready for masking.")

    last_frame = (project.frame_count or 1) - 1
    _, end = document.frame_span()
    if end > last_frame:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"A keyframe is on frame {end + 1}, but this video has {project.frame_count} frames.",
        )

    start, end = document.frame_span()
    payload = document.model_dump(mode="json")

    def apply(existing: Mask | None) -> Mask:
        if existing is None:
            existing = Mask(project_id=project.id, mask_data=payload, source="manual")
            db.add(existing)
        existing.mask_data = payload
        existing.source = "manual"  # a person has now seen and saved it
        existing.frame_start, existing.frame_end = start, end
        # Edited keyframes invalidate anything tracking produced from the old ones.
        existing.tracking_status = TRACKING_NONE
        existing.tracking_progress = 0
        existing.tracking_total = 0
        existing.tracking_error = None
        existing.tracking_started_at = None
        existing.tracking_data = None
        return existing

    mask = apply(_load_mask(db, project.id))
    try:
        db.commit()
    except IntegrityError:
        # Two first-saves raced; the other one won, so update its row instead.
        db.rollback()
        mask = apply(_load_mask(db, project.id))
        db.commit()
    db.refresh(mask)
    logger.info(
        "mask saved",
        extra={"project_id": str(project.id), "user_id": str(user.id), "regions": len(document.regions)},
    )
    return MaskResponse(
        mask_data=MaskDocument.model_validate(mask.mask_data),
        tracking_status=mask.tracking_status,
        updated_at=mask.updated_at,
    )
