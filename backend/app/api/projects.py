"""Projects and resumable uploads.

Flow: create (or resume) -> sign part URLs -> browser PUTs parts straight to storage ->
complete -> a worker analyses the file. Every route is scoped to the signed-in user; another
person's project is indistinguishable from one that doesn't exist.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import CurrentUser
from app.config import get_settings
from app.db import get_db
from app.disk import InsufficientStorage, ensure_free_space
from app.models import Project, ProjectStatus
from app.ratelimit import per_user
from app.schemas.project import (
    MAX_PART_NUMBER,
    CompleteUploadRequest,
    PartInfo,
    PartUrl,
    PartUrlRequest,
    PartUrlResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpload,
    UploadState,
)
from app.services.cleanup import (
    delete_project_files,
    expire_project,
    extend_retention,
    is_due,
)
from app.storage.s3 import ObjectStorage, StorageError, get_storage
from app.video.probe import SNIFF_BYTES, UNREADABLE_MESSAGE, looks_like_video
from app.workers.tasks import EnqueueError, enqueue_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["projects"])

CONTENT_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".avi": "video/x-msvideo"}
UNSUPPORTED_FORMAT = "Unsupported video format. Please upload an MP4, MOV, WebM or AVI file."
NOT_FOUND = "We couldn't find that project."
EXPIRED_MESSAGE = "This project has expired and its files were deleted. Upload the video again to start a new one."
LIST_LIMIT = 50

Db = Annotated[Session, Depends(get_db)]
Storage = Annotated[ObjectStorage, Depends(get_storage)]


def _clean_filename(raw: str) -> str:
    """Display-only name: drop any directory part and control characters."""
    base = re.split(r"[\\/]", raw)[-1]
    return re.sub(r"[\x00-\x1f\x7f]", "", base).strip()[:255]


def _expire_if_due(db: Session, project: Project) -> None:
    """Retention doesn't wait for the periodic sweep: a project past its deadline is deleted when somebody asks for it
    (unless a worker is still using it). A storage problem is logged and left for the sweep; access stops either way."""
    if not is_due(project):
        return
    try:
        expire_project(db, get_storage(), project)
    except StorageError:
        db.rollback()
        logger.warning("couldn't delete an expired project's files yet", extra={"project_id": str(project.id)}, exc_info=True)


def _owned_project(db: Session, user_id: uuid.UUID, project_id: uuid.UUID, *, allow_expired: bool = False) -> Project:
    project = db.scalar(select(Project).where(Project.id == project_id, Project.user_id == user_id))
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
    _expire_if_due(db, project)
    if project.status == ProjectStatus.EXPIRED.value and not allow_expired:
        raise HTTPException(status.HTTP_410_GONE, EXPIRED_MESSAGE)
    return project


def _total_parts(project: Project) -> int:
    return max(1, math.ceil(project.size_bytes / get_settings().upload_part_size))


def _upload_state(project: Project, storage: ObjectStorage) -> UploadState:
    uploaded: list[PartInfo] = []
    if project.status == ProjectStatus.UPLOADING.value and project.upload_id:
        uploaded = [
            PartInfo(part_number=p.part_number, etag=p.etag, size=p.size)
            for p in storage.list_parts(project.original_file_key, project.upload_id)
        ]
    return UploadState(
        part_size=get_settings().upload_part_size, total_parts=_total_parts(project), uploaded_parts=uploaded
    )


def _require_uploading(project: Project) -> str:
    if project.status != ProjectStatus.UPLOADING.value or not project.upload_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "This upload is already finished.")
    return project.upload_id


@router.post("", response_model=ProjectUpload, dependencies=[per_user("project_create")])
def create_project(
    payload: ProjectCreate, response: Response, user: CurrentUser, db: Db, storage: Storage
) -> ProjectUpload:
    settings = get_settings()
    filename = _clean_filename(payload.filename)
    extension = PurePosixPath(filename).suffix.lower()

    if extension not in CONTENT_TYPES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, UNSUPPORTED_FORMAT)
    if payload.size_bytes > settings.max_upload_size:
        limit_mb = settings.max_upload_size // (1024 * 1024)
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"This file is larger than the {limit_mb} MB upload limit. Choose a smaller file or trim the video first.",
        )
    if math.ceil(payload.size_bytes / settings.upload_part_size) > MAX_PART_NUMBER:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "This file is too large to upload.")
    if settings.storage_backend == "filesystem":  # with object storage the bucket's own capacity applies
        try:
            ensure_free_space(settings.local_data_dir, payload.size_bytes, what="this upload")
        except InsufficientStorage as full:
            raise HTTPException(status.HTTP_507_INSUFFICIENT_STORAGE, full.message) from None

    now = datetime.now(UTC)

    # Same person, same file, still uploading: pick up where it left off instead of starting over.
    existing = db.scalar(
        select(Project).where(
            Project.user_id == user.id,
            Project.original_filename == filename,
            Project.size_bytes == payload.size_bytes,
            Project.status == ProjectStatus.UPLOADING.value,
            Project.expires_at > now,
        )
    )
    if existing is not None:
        return ProjectUpload(project=ProjectResponse.model_validate(existing), upload=_upload_state(existing, storage))

    project_id = uuid.uuid4()
    # Keys are built from server-generated ids only; the person's file name never reaches storage.
    key = f"uploads/{user.id}/{project_id}/original{extension}"
    project = Project(
        id=project_id,
        user_id=user.id,
        name=payload.name or (PurePosixPath(filename).stem or "Untitled video")[:200],
        original_filename=filename,
        size_bytes=payload.size_bytes,
        status=ProjectStatus.UPLOADING.value,
        original_file_key=key,
        upload_id=storage.create_multipart_upload(key, CONTENT_TYPES[extension]),
        expires_at=now + timedelta(hours=settings.video_retention_hours),
    )
    db.add(project)
    db.commit()
    logger.info("upload started", extra={"project_id": str(project.id), "user_id": str(user.id)})

    response.status_code = status.HTTP_201_CREATED
    return ProjectUpload(project=ProjectResponse.model_validate(project), upload=_upload_state(project, storage))


@router.get("", response_model=list[ProjectResponse])
def list_projects(user: CurrentUser, db: Db) -> list[Project]:
    projects = list(
        db.scalars(
            select(Project).where(Project.user_id == user.id).order_by(Project.created_at.desc()).limit(LIST_LIMIT)
        )
    )
    for project in projects:
        _expire_if_due(db, project)
    return projects


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: uuid.UUID, user: CurrentUser, db: Db) -> Project:
    # An expired project is still described (status EXPIRED) so the page can explain what happened.
    return _owned_project(db, user.id, project_id, allow_expired=True)


@router.get("/{project_id}/upload", response_model=UploadState)
def get_upload_state(project_id: uuid.UUID, user: CurrentUser, db: Db, storage: Storage) -> UploadState:
    project = _owned_project(db, user.id, project_id)
    _require_uploading(project)
    return _upload_state(project, storage)


@router.post("/{project_id}/upload/urls", response_model=PartUrlResponse, dependencies=[per_user("upload_urls")])
def sign_part_urls(
    project_id: uuid.UUID, payload: PartUrlRequest, user: CurrentUser, db: Db, storage: Storage
) -> PartUrlResponse:
    project = _owned_project(db, user.id, project_id)
    upload_id = _require_uploading(project)

    total = _total_parts(project)
    if any(n > total for n in payload.part_numbers):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"This upload has {total} parts.")

    return PartUrlResponse(
        urls=[
            PartUrl(part_number=n, url=storage.presign_part(project.original_file_key, upload_id, n))
            for n in sorted(set(payload.part_numbers))
        ]
    )


@router.post("/{project_id}/upload/complete", response_model=ProjectResponse, status_code=status.HTTP_202_ACCEPTED)
def complete_upload(
    project_id: uuid.UUID, payload: CompleteUploadRequest, user: CurrentUser, db: Db, storage: Storage
) -> Project:
    project = _owned_project(db, user.id, project_id)

    if project.status != ProjectStatus.UPLOADING.value:
        # Completing twice (a retry after a dropped response) is harmless.
        return project
    upload_id = _require_uploading(project)

    total = _total_parts(project)
    numbers = sorted(p.part_number for p in payload.parts)
    if numbers != list(range(1, total + 1)):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"The upload is incomplete: expected parts 1 to {total}. Resume the upload to send the rest.",
        )

    # Trust storage, not the browser: the bytes actually received must add up to the declared size.
    received = sum(p.size for p in storage.list_parts(project.original_file_key, upload_id))
    if received != project.size_bytes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "The uploaded data doesn't match the file size. Resume the upload and try again.",
        )

    storage.complete_multipart_upload(
        project.original_file_key, upload_id, [(p.part_number, p.etag) for p in payload.parts]
    )
    project.status = ProjectStatus.UPLOADED.value
    project.upload_id = None
    extend_retention(project)  # a slow upload mustn't eat into the retention period
    # The file name and the browser's MIME type are claims. The file's own first bytes say what it is, so something
    # that isn't an MP4, MOV or WebM is refused here, before any worker spends time on it.
    if not looks_like_video(storage.read_head(project.original_file_key, SNIFF_BYTES)):
        project.status = ProjectStatus.FAILED.value
        project.error = UNREADABLE_MESSAGE
        db.commit()
        logger.info("upload refused: not a video", extra={"project_id": str(project.id), "user_id": str(user.id)})
        db.refresh(project)
        return project
    db.commit()
    logger.info("upload completed", extra={"project_id": str(project.id), "user_id": str(user.id)})

    _start_analysis(project)
    db.refresh(project)
    return project


@router.post("/{project_id}/analyze", response_model=ProjectResponse, status_code=status.HTTP_202_ACCEPTED)
def retry_analysis(project_id: uuid.UUID, user: CurrentUser, db: Db) -> Project:
    project = _owned_project(db, user.id, project_id)
    if project.status not in {ProjectStatus.UPLOADED.value, ProjectStatus.FAILED.value}:
        raise HTTPException(status.HTTP_409_CONFLICT, "This project isn't waiting for analysis.")

    project.status = ProjectStatus.UPLOADED.value
    project.error = None
    db.commit()

    _start_analysis(project)
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: uuid.UUID, user: CurrentUser, db: Db, storage: Storage) -> Response:
    project = _owned_project(db, user.id, project_id, allow_expired=True)

    # Files go first. If storage fails the row stays, so the person can simply try again. A render output key is
    # deterministic per job, including a job still running (a worker that finishes after this finds its job gone and
    # removes what it uploaded).
    delete_project_files(db, storage, project)

    db.delete(project)
    db.commit()
    logger.info("project deleted", extra={"project_id": str(project_id), "user_id": str(user.id)})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _start_analysis(project: Project) -> None:
    try:
        enqueue_analysis(project.id)
    except EnqueueError:
        logger.exception("couldn't enqueue analysis", extra={"project_id": str(project.id)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Your upload finished, but we couldn't start analyzing it. Try again in a moment.",
        ) from None
