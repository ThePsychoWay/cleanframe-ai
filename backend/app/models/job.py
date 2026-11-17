import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


JOB_RENDER = "render"
JOB_EXPORT = "export"  # the finished video re-encoded at another size or codec (no AI work)

# Job.status
JOB_QUEUED = "queued"  # waiting for a worker
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
ACTIVE_JOB_STATUSES = (JOB_QUEUED, JOB_RUNNING)

# Job.stage: where a running (or stopped) job is. Set by the worker as it really gets there.
STAGE_QUEUED = "queued"
STAGE_PREPARING = "preparing"  # fetching the original
STAGE_MEASURING = "measuring"  # an engine that studies the video first (the survey pass) is doing so
STAGE_RECONSTRUCTING = "reconstructing"  # rebuilding the picture and encoding it, frame by frame
STAGE_AUDIO = "audio"
STAGE_VERIFYING = "verifying"
STAGE_SAVING = "saving"
STAGES = (STAGE_PREPARING, STAGE_MEASURING, STAGE_RECONSTRUCTING, STAGE_AUDIO, STAGE_VERIFYING, STAGE_SAVING)


class Job(Base):
    """One background job for a project. Today only renders; the type column keeps room for others.

    Progress is what the worker actually did: `frames_done` counts frames that came out of the reconstruction
    engine, `stage` is the step the worker has reached. Nothing here is estimated.
    """

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(20), default=JOB_RENDER)
    status: Mapped[str] = mapped_column(String(20), default=JOB_QUEUED, index=True)
    stage: Mapped[str] = mapped_column(String(20), default=STAGE_QUEUED)
    queue: Mapped[str] = mapped_column(String(20), default="cpu")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    frames_done: Mapped[int] = mapped_column(Integer, default=0)
    frames_total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(500), default=None)

    # What the person asked for, and what ran. Fixed when the job is created.
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    engine: Mapped[str] = mapped_column(String(20), default="")
    development_only: Mapped[bool] = mapped_column(Boolean, default=False)
    tracking_used: Mapped[bool] = mapped_column(Boolean, default=False)

    # Filled in when the job finishes.
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    output_file_key: Mapped[str | None] = mapped_column(String(512), default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    reconstruct_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), onupdate=_utcnow
    )
