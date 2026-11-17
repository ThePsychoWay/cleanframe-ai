import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ProjectStatus(StrEnum):
    UPLOADING = "UPLOADING"  # created; the browser is still sending parts
    UPLOADED = "UPLOADED"  # all bytes are in storage; analysis not finished
    ANALYZING = "ANALYZING"
    MASKING = "MASKING"  # analysed and ready for the person to mark overlays
    TRACKING = "TRACKING"
    PROCESSING = "PROCESSING"
    ENCODING = "ENCODING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    # Shown to the person only. Never used to build a storage key or a filesystem path.
    original_filename: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(20), default=ProjectStatus.UPLOADING.value, index=True)
    error: Mapped[str | None] = mapped_column(String(500), default=None)

    # Storage keys, never URLs: signed links are created on demand.
    original_file_key: Mapped[str] = mapped_column(String(512))
    output_file_key: Mapped[str | None] = mapped_column(String(512), default=None)
    upload_id: Mapped[str | None] = mapped_column(String(255), default=None)

    # Editor preview, made during analysis. Keys only; links are signed on request.
    proxy_file_key: Mapped[str | None] = mapped_column(String(512), default=None)
    thumbnail_file_key: Mapped[str | None] = mapped_column(String(512), default=None)
    proxy_width: Mapped[int | None] = mapped_column(Integer, default=None)
    proxy_height: Mapped[int | None] = mapped_column(Integer, default=None)

    # Filled in by analysis, from the file's real contents.
    container: Mapped[str | None] = mapped_column(String(40), default=None)
    codec: Mapped[str | None] = mapped_column(String(40), default=None)
    audio_codec: Mapped[str | None] = mapped_column(String(40), default=None)
    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)
    fps: Mapped[float | None] = mapped_column(Float, default=None)
    duration: Mapped[float | None] = mapped_column(Float, default=None)
    bitrate: Mapped[int | None] = mapped_column(BigInteger, default=None)
    frame_count: Mapped[int | None] = mapped_column(Integer, default=None)
    # What the automatic pipeline found and did: see app.services.automation. None for a project made before it existed.
    automation: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), default=None)

    # Set in Python (microseconds) so "newest first" is deterministic on every database;
    # the server default only covers rows inserted outside the app.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), onupdate=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
