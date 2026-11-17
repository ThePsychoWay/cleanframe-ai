"""How a job is shown: the list of stages, speed and time remaining. Pure functions of the job record and the clock."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Job, Project
from app.models.job import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    STAGE_AUDIO,
    STAGE_MEASURING,
    STAGE_PREPARING,
    STAGE_RECONSTRUCTING,
    STAGE_SAVING,
    STAGE_VERIFYING,
    STAGES,
)
from app.schemas.render import EngineOut, JobOut, StageOut
from app.services.rendering import aware

MIN_FRAMES_FOR_ESTIMATE = 3

_LABELS = {
    STAGE_PREPARING: "Fetching the original",
    STAGE_MEASURING: "Measuring the overlay",
    STAGE_RECONSTRUCTING: "Rebuilding and encoding the video",
    STAGE_AUDIO: "Restoring audio",
    STAGE_VERIFYING: "Checking the result",
    STAGE_SAVING: "Saving the result",
}


def speed_and_eta(job: Job, now: datetime | None = None) -> tuple[float | None, float | None]:
    started = aware(job.reconstruct_started_at)
    if job.status != JOB_RUNNING or job.stage != STAGE_RECONSTRUCTING or started is None:
        return None, None
    if job.frames_done < MIN_FRAMES_FOR_ESTIMATE:
        return None, None
    elapsed = ((now or datetime.now(UTC)) - started).total_seconds()
    if elapsed <= 0:
        return None, None
    speed = job.frames_done / elapsed
    remaining = max(0, job.frames_total - job.frames_done)
    return round(speed, 2), round(remaining / speed, 1)


def _wants_audio(job: Job, project: Project) -> bool:
    return bool(project.audio_codec) and bool(job.settings.get("keep_audio", True))


def build_stages(job: Job, project: Project) -> list[StageOut]:
    """The steps of the job in order, each marked done, active, pending, skipped, failed or stopped by what the worker reached."""
    earlier = [
        StageOut(key="analyze", label="Analyzing video", state="done"),
        StageOut(key="mask", label="Creating mask", state="done"),
        StageOut(
            key="track",
            label="Tracking watermark",
            state="done" if job.tracking_used else "skipped",
            detail=None if job.tracking_used else "Not used: the marks stay where you placed them",
        ),
    ]
    reached = STAGES.index(job.stage) if job.stage in STAGES else -1
    stages = list(earlier)
    for index, key in enumerate(STAGES):
        detail: str | None = None
        if key == STAGE_MEASURING and job.engine != "auto":
            state = "skipped"
            detail = "Only the automatic engine studies the video first"
        elif key == STAGE_AUDIO and not _wants_audio(job, project):
            state = "skipped"
            detail = "The original has no audio" if not project.audio_codec else "Audio removal was chosen"
        elif job.status == JOB_COMPLETED:
            state = "done"
        elif job.status == JOB_QUEUED:
            state = "pending"
        elif index < reached:
            state = "done"
        elif index == reached:
            state = {JOB_RUNNING: "active", JOB_FAILED: "failed", JOB_CANCELLED: "stopped"}[job.status]
        else:
            state = "pending"
        if key == STAGE_RECONSTRUCTING and job.frames_total and state in ("active", "done", "failed", "stopped"):
            detail = f"{job.frames_done} of {job.frames_total} frames"
        stages.append(StageOut(key=key, label=_LABELS[key], state=state, detail=detail))  # type: ignore[arg-type]
    return stages


def jobs_ahead(db: Session, job: Job) -> int | None:
    if job.status != JOB_QUEUED:
        return None
    return int(
        db.scalar(
            select(func.count()).select_from(Job).where(Job.status == JOB_QUEUED, Job.queue == job.queue, Job.created_at < job.created_at)
        )
        or 0
    )


def to_out(db: Session, job: Job, project: Project) -> JobOut:
    speed, eta = speed_and_eta(job)
    if job.status == JOB_COMPLETED:
        percent: float | None = 100.0
    elif job.status == JOB_RUNNING and job.stage == STAGE_RECONSTRUCTING and job.frames_total:
        percent = round(min(100.0, 100.0 * job.frames_done / job.frames_total), 1)
    else:
        percent = None
    return JobOut(
        id=str(job.id),
        project_id=str(job.project_id),
        status=job.status,  # type: ignore[arg-type]
        stage=job.stage,
        stages=build_stages(job, project),
        frames_done=job.frames_done,
        frames_total=job.frames_total,
        percent=percent,
        speed_fps=speed,
        eta_seconds=eta,
        jobs_ahead=jobs_ahead(db, job),
        error=job.error,
        warnings=list(job.warnings or []),
        engine=EngineOut(provider=job.engine, development_only=job.development_only),
        settings=dict(job.settings or {}),
        result=job.result,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
