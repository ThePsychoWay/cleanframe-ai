from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ExportRequest(BaseModel):
    """An additional copy of the finished video. `height` is the shorter side of the new picture (2160 = 4K, 1080 = Full HD...);
    None keeps the finished video's size and only changes the format."""

    model_config = ConfigDict(extra="forbid")

    height: Literal[720, 1080, 1440, 2160] | None = None
    container: Literal["mp4", "webm"] = "mp4"
    codec: Literal["h264", "hevc", "vp9"] | None = None
    quality: Literal["fast", "balanced", "high"] = "balanced"

    @model_validator(mode="after")
    def _codec_fits_container(self) -> ExportRequest:
        allowed = {"mp4": ("h264", "hevc"), "webm": ("vp9",)}[self.container]
        if self.codec is not None and self.codec not in allowed:
            raise ValueError(f"{self.codec.upper()} can't be stored in {self.container.upper()}.")
        return self


class SizeOption(BaseModel):
    height: int  # the shorter side
    label: str
    width_px: int
    height_px: int


class FormatOption(BaseModel):
    container: Literal["mp4", "webm"]
    codec: Literal["h264", "hevc", "vp9"]
    label: str


class ExportOptions(BaseModel):
    """What can honestly be exported from the finished video."""

    original_width: int
    original_height: int
    original_label: str
    output_width: int
    output_height: int
    output_label: str
    codec: str | None
    container: str
    sizes: list[SizeOption]
    formats: list[FormatOption]
    hardware_encoder: bool  # NVENC is in use for new encodes


class ExportJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    stage: str
    frames_done: int
    frames_total: int
    percent: float | None
    error: str | None
    warnings: list[str]
    settings: dict[str, Any]
    result: dict[str, Any] | None
    created_at: datetime
    completed_at: datetime | None


class ExportList(BaseModel):
    exports: list[ExportJobOut]
