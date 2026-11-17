import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_PART_NUMBER = 10_000  # S3 limit

Filename = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
ProjectName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ProjectCreate(BaseModel):
    filename: Filename
    size_bytes: int = Field(gt=0)
    name: ProjectName | None = None


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    original_filename: str
    size_bytes: int
    status: str
    error: str | None
    container: str | None
    codec: str | None
    audio_codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    duration: float | None
    bitrate: int | None
    frame_count: int | None
    automation: dict | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class MediaResponse(BaseModel):
    """Time-limited links for the editor. Anyone holding one can read that file until it expires."""

    proxy_url: str
    thumbnail_url: str
    expires_in: int
    # The proxy's size; masks are saved in fractions of the frame so they apply to the source unchanged.
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


class PartInfo(BaseModel):
    part_number: int
    etag: str
    size: int


class UploadState(BaseModel):
    part_size: int
    total_parts: int
    uploaded_parts: list[PartInfo]


class ProjectUpload(BaseModel):
    project: ProjectResponse
    upload: UploadState


class PartUrlRequest(BaseModel):
    part_numbers: list[Annotated[int, Field(ge=1, le=MAX_PART_NUMBER)]] = Field(min_length=1, max_length=20)


class PartUrl(BaseModel):
    part_number: int
    url: str


class PartUrlResponse(BaseModel):
    urls: list[PartUrl]


class CompletedPart(BaseModel):
    part_number: int = Field(ge=1, le=MAX_PART_NUMBER)
    etag: str = Field(min_length=1, max_length=128)


class CompleteUploadRequest(BaseModel):
    parts: list[CompletedPart] = Field(min_length=1, max_length=MAX_PART_NUMBER)
