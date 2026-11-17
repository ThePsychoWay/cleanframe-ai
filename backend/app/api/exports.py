"""Additional exports of a finished video: another size or format, made from the finished file (no AI work is repeated).

Only technically valid choices are offered: a size smaller than the finished video, and formats this server can encode.
Every route is scoped to the signed-in user, like the rest of a project.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.projects import Db, Storage, _owned_project
from app.api.result import DOWNLOAD_TTL_SECONDS, _require_result, download_name
from app.auth.deps import CurrentUser
from app.config import get_settings
from app.models import Job
from app.models.job import ACTIVE_JOB_STATUSES, JOB_COMPLETED, JOB_EXPORT, JOB_FAILED, JOB_QUEUED
from app.ratelimit import per_user
from app.schemas.export import ExportJobOut, ExportList, ExportOptions, ExportRequest, FormatOption, SizeOption
from app.schemas.result import ExportResponse
from app.services.exporting import export_filename_suffix
from app.services.rendering import expire_if_lost
from app.video.encoders import LABELS, choose_encoder, codec_available
from app.video.frames import output_size
from app.video.resolution import resolution_class, target_label, valid_targets
from app.workers.tasks import EnqueueError, enqueue_export

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["exports"])

MAX_EXPORTS_PER_PROJECT = 12
FORMATS = (("mp4", "h264"), ("mp4", "hevc"), ("webm", "vp9"))


def _out(job: Job) -> ExportJobOut:
    percent = None
    if job.status == JOB_COMPLETED:
        percent = 100.0
    elif job.frames_total > 0 and job.status in ACTIVE_JOB_STATUSES:
        percent = round(100.0 * min(job.frames_done, job.frames_total) / job.frames_total, 1)
    return ExportJobOut(
        id=str(job.id), status=job.status, stage=job.stage, frames_done=job.frames_done, frames_total=job.frames_total,
        percent=percent, error=job.error, warnings=list(job.warnings or []), settings=dict(job.settings), result=job.result,
        created_at=job.created_at, completed_at=job.completed_at,
    )  # fmt: skip


def _jobs(db: Db, project_id: uuid.UUID) -> list[Job]:
    jobs = list(
        db.scalars(select(Job).where(Job.project_id == project_id, Job.type == JOB_EXPORT).order_by(Job.created_at.desc()).limit(MAX_EXPORTS_PER_PROJECT))
    )
    for job in jobs:
        expire_if_lost(db, job)
    return jobs


@router.get("/{project_id}/exports/options", response_model=ExportOptions)
def export_options(project_id: uuid.UUID, user: CurrentUser, db: Db, storage: Storage) -> ExportOptions:
    project = _owned_project(db, user.id, project_id)
    job, _ = _require_result(db, project, storage)
    result = job.result or {}
    out_w, out_h = int(result["width"]), int(result["height"])
    src_w = int(result.get("source_width") or project.width or out_w)
    src_h = int(result.get("source_height") or project.height or out_h)
    sizes = []
    for target in valid_targets(out_w, out_h):
        width, height = output_size(out_w, out_h, target)
        sizes.append(SizeOption(height=target, label=target_label(target), width_px=width, height_px=height))
    formats = [
        FormatOption(container=container, codec=codec, label=f"{LABELS[codec]} in {container.upper()}")  # type: ignore[arg-type]
        for container, codec in FORMATS
        if codec_available(codec)
    ]
    encoder = choose_encoder("mp4", "h264", "balanced", get_settings().video_encoder)
    return ExportOptions(
        original_width=src_w, original_height=src_h, original_label=resolution_class(src_w, src_h)[1],
        output_width=out_w, output_height=out_h, output_label=resolution_class(out_w, out_h)[1],
        codec=result.get("codec"), container=str(result.get("container", "mp4")),
        sizes=sizes, formats=formats, hardware_encoder=encoder.hardware,
    )  # fmt: skip


@router.get("/{project_id}/exports", response_model=ExportList)
def list_exports(project_id: uuid.UUID, user: CurrentUser, db: Db) -> ExportList:
    project = _owned_project(db, user.id, project_id)
    return ExportList(exports=[_out(job) for job in _jobs(db, project.id)])


@router.post(
    "/{project_id}/exports", response_model=ExportJobOut, status_code=status.HTTP_202_ACCEPTED, dependencies=[per_user("render_start")]
)
def create_export(project_id: uuid.UUID, body: ExportRequest, user: CurrentUser, db: Db, storage: Storage) -> ExportJobOut:
    project = _owned_project(db, user.id, project_id)
    source, _ = _require_result(db, project, storage)
    result = source.result or {}
    out_w, out_h = int(result["width"]), int(result["height"])
    if body.height is not None and body.height not in valid_targets(out_w, out_h):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"The finished video is {out_w}×{out_h}; it can only be exported at a smaller size, not enlarged.",
        )
    codec = body.codec or {"mp4": "h264", "webm": "vp9"}[body.container]
    if not codec_available(codec):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"This server can't encode {LABELS[codec]}.")
    if body.height is None and codec == result.get("codec") and body.container == result.get("container"):
        raise HTTPException(status.HTTP_409_CONFLICT, "That is the format of the finished video. Download it directly.")

    settings = {
        "height": body.height, "container": body.container, "codec": codec, "quality": body.quality,
        "source_key": project.output_file_key, "source_job_id": str(source.id), "source_container": str(result.get("container", "mp4")),
        "frames": int(result["frames"]),
    }  # fmt: skip
    wanted = {k: settings[k] for k in ("height", "container", "codec", "quality", "source_key")}
    for existing in _jobs(db, project.id):  # asking twice for the same thing gives the same export
        if existing.status in (*ACTIVE_JOB_STATUSES, JOB_COMPLETED) and {k: existing.settings.get(k) for k in wanted} == wanted:
            return _out(existing)

    job = Job(
        project_id=project.id, user_id=user.id, type=JOB_EXPORT, status=JOB_QUEUED, queue="cpu",
        frames_total=int(result["frames"]), settings=settings, engine="ffmpeg",
    )  # fmt: skip
    db.add(job)
    db.commit()
    logger.info("export requested", extra={"job_id": str(job.id), "project_id": str(project.id), "user_id": str(user.id), "export": settings})
    try:
        enqueue_export(job.id)
    except EnqueueError:
        job.status, job.error = JOB_FAILED, "The job queue isn't reachable."
        db.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "We couldn't start the export. Try again in a moment.") from None
    db.expire_all()
    return _out(db.get(Job, job.id))  # type: ignore[arg-type]


@router.post("/{project_id}/exports/{job_id}/link", response_model=ExportResponse, dependencies=[per_user("export_link")])
def export_link(project_id: uuid.UUID, job_id: uuid.UUID, user: CurrentUser, db: Db, storage: Storage) -> ExportResponse:
    project = _owned_project(db, user.id, project_id)
    job = db.scalar(select(Job).where(Job.id == job_id, Job.project_id == project.id, Job.type == JOB_EXPORT))
    if job is None or job.status != JOB_COMPLETED or not job.output_file_key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That export isn't ready.")
    size = storage.object_size(job.output_file_key)
    if size is None:
        raise HTTPException(status.HTTP_410_GONE, "This export is no longer available. Make it again.")
    result = job.result or {}
    base = download_name(project.name, str(result.get("container", "mp4")))
    stem, _, extension = base.rpartition(".")
    filename = f"{stem}-{export_filename_suffix(job.settings, int(result.get('height', 0)))}.{extension}"
    url = storage.presign_get(job.output_file_key, expires_in=DOWNLOAD_TTL_SECONDS, download_as=filename)
    logger.info("export link issued", extra={"project_id": str(project.id), "user_id": str(user.id), "job_id": str(job.id), "bytes": size})
    return ExportResponse(url=url, filename=filename, expires_in=DOWNLOAD_TTL_SECONDS, size_bytes=size)
