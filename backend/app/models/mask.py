import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, deferred, mapped_column

from app.models.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Values of Mask.tracking_status.
TRACKING_NONE = "none"  # not tracked, or the marks changed since it was
TRACKING_RUNNING = "running"
TRACKING_DONE = "done"
TRACKING_FAILED = "failed"


class Mask(Base):
    """The regions a person marked on one project. One document per project, replaced on save.

    `mask_data` follows app.schemas.mask.MaskDocument: regions, each with keyframes, in coordinates
    normalised to the frame (0 to 1) so they're independent of proxy or source resolution.
    """

    __tablename__ = "masks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), unique=True, index=True
    )
    mask_data: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    # First and last keyframe across all regions, so later stages can skip untouched frames.
    frame_start: Mapped[int] = mapped_column(Integer, default=0)
    frame_end: Mapped[int] = mapped_column(Integer, default=0)
    # "auto": found by the detector; "manual": drawn or changed by a person. Saving from the editor makes it manual.
    source: Mapped[str] = mapped_column(String(10), default="manual", server_default="manual")
    # See TRACKING_* above. Saving new marks always returns this to "none".
    tracking_status: Mapped[str] = mapped_column(String(20), default=TRACKING_NONE, server_default=TRACKING_NONE)
    tracking_progress: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    tracking_total: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    tracking_error: Mapped[str | None] = mapped_column(String(500), default=None)
    tracking_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # The tracked motion (app.tracking.shapes), which can be megabytes: only loaded when asked for.
    tracking_data: Mapped[dict | None] = deferred(
        mapped_column(JSON().with_variant(JSONB(), "postgresql"), default=None)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), onupdate=_utcnow
    )
