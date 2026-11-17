"""The finished video: links to preview it next to the original, and to download it.

The files stay in private storage. The browser only ever gets time-limited signed links, issued here after the
ownership check, so a link can't be guessed or reused for somebody else's project.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.projects import Db, Storage, _owned_project
from app.auth.deps import CurrentUser
from app.models import Job, Project
from app.models.job import JOB_COMPLETED
from app.ratelimit import per_user
from app.schemas.result import ExportResponse, PreviewResponse
from app.storage.s3 import PRESIGN_TTL_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["result"])

DOWNLOAD_TTL_SECONDS = 300  # the browser starts the download as soon as it has the link
NO_RESULT = "There's no finished video to show yet."
FILE_GONE = "The finished video is no longer available. Process the video again to make a new one."
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def download_name(project_name: str, container: str) -> str:
    """A file name for the download built from the project's name: letters, digits, spaces, dots, dashes and
    underscores only, so it can't carry quotes, slashes or control characters into a header."""
    ascii_only = unicodedata.normalize("NFKD", project_name).encode("ascii", "ignore").decode()
    stem = _UNSAFE.sub("", ascii_only).strip(" .-_")
    stem = re.sub(r"\s+", " ", stem)[:80].strip(" .-_") or "video"
    extension = container if container in ("mp4", "webm") else "mp4"
    return f"{stem}-restored.{extension}"


def _result_job(db: Db, project: Project) -> Job | None:
    """The job that produced the project's current result."""
    return db.scalar(
        select(Job)
        .where(Job.project_id == project.id, Job.status == JOB_COMPLETED, Job.output_file_key == project.output_file_key)
        .order_by(Job.completed_at.desc())
        .limit(1)
    )


def _require_result(db: Db, project: Project, storage: Storage) -> tuple[Job, int | None]:
    if not project.output_file_key:
        raise HTTPException(status.HTTP_409_CONFLICT, NO_RESULT)
    job = _result_job(db, project)
    if job is None or not job.result:
        raise HTTPException(status.HTTP_409_CONFLICT, NO_RESULT)
    size = storage.object_size(project.output_file_key)
    if size is None:
        raise HTTPException(status.HTTP_410_GONE, FILE_GONE)
    return job, size


@router.get("/{project_id}/preview", response_model=PreviewResponse)
def get_preview(project_id: uuid.UUID, user: CurrentUser, db: Db, storage: Storage) -> PreviewResponse:
    project = _owned_project(db, user.id, project_id)
    job, size = _require_result(db, project, storage)
    assert project.output_file_key is not None
    if not project.proxy_file_key:
        raise HTTPException(status.HTTP_409_CONFLICT, "The preview copy of the original isn't ready.")
    result = job.result or {}
    return PreviewResponse(
        result_url=storage.presign_get(project.output_file_key, expires_in=PRESIGN_TTL_SECONDS),
        original_url=storage.presign_get(project.proxy_file_key, expires_in=PRESIGN_TTL_SECONDS),
        expires_in=PRESIGN_TTL_SECONDS,
        width=int(result["width"]),
        height=int(result["height"]),
        fps=float(result["fps"]),
        frame_count=int(result["frames"]),
        container=str(result.get("container", "mp4")),
        size_bytes=size,
        audio=result.get("audio"),
        development_only=job.development_only,
        warnings=list(job.warnings or []),
        finished_at=job.completed_at,
        facts=result,
    )


@router.post("/{project_id}/export", response_model=ExportResponse, dependencies=[per_user("export_link")])
def export_result(project_id: uuid.UUID, user: CurrentUser, db: Db, storage: Storage) -> ExportResponse:
    project = _owned_project(db, user.id, project_id)
    job, size = _require_result(db, project, storage)
    assert project.output_file_key is not None
    filename = download_name(project.name, str((job.result or {}).get("container", "mp4")))
    url = storage.presign_get(project.output_file_key, expires_in=DOWNLOAD_TTL_SECONDS, download_as=filename)
    logger.info(
        "export link issued",
        extra={"project_id": str(project.id), "user_id": str(user.id), "job_id": str(job.id), "bytes": size},
    )
    return ExportResponse(url=url, filename=filename, expires_in=DOWNLOAD_TTL_SECONDS, size_bytes=size)
