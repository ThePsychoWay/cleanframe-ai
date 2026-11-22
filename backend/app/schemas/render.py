from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from app.video.containers import CONTAINER_VIDEO_CODECS
from pydantic import BaseModel, ConfigDict, model_validator


class RenderRequest(BaseModel):
    """What the person chose for the final render. Unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")

    quality: Literal["fast", "balanced", "high"] = "balanced"
    container: Literal["mp4", "mov", "webm", "mkv"] = "mp4"
    output_height: Literal[720, 1080, 1440, 2160] | None = None  # None = original size. The shorter side is capped at this; only ever scales down.
    codec: Literal["h264", "hevc", "vp9", "av1"] | None = None  # None = the container's first codec: H.264 in MP4, MOV and MKV, VP9 in WebM
    keep_audio: bool = True
    # Required when the configured engine sends frames to a third party (the response says so first).
    acknowledge_third_party: bool = False

    @model_validator(mode="after")
    def _codec_fits_container(self) -> RenderRequest:
        allowed = CONTAINER_VIDEO_CODECS[self.container]
        if self.codec is not None and self.codec not in allowed:
            raise ValueError(f"{self.codec.upper()} can't be stored in {self.container.upper()}.")
        return self


class StageOut(BaseModel):
    key: str
    label: str
    state: Literal["done", "active", "pending", "skipped", "failed", "stopped"]
    detail: str | None = None


class EngineOut(BaseModel):
    provider: str
    development_only: bool


class JobOut(BaseModel):
    """A render job as the interface shows it. Everything here was measured or recorded by the worker."""

    id: str
    project_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    stage: str
    stages: list[StageOut]
    frames_done: int
    frames_total: int
    # Share of frames finished while frames are being rebuilt; null in every other stage, where there is no
    # honest fraction to show. 100 once the job is complete.
    percent: float | None
    speed_fps: float | None  # frames finished per second so far, while rebuilding
    eta_seconds: float | None  # remaining frames at that speed: the rebuilding step only, not the steps after it
    jobs_ahead: int | None  # queued jobs on the same queue created earlier, while waiting
    error: str | None
    warnings: list[str]
    engine: EngineOut
    settings: dict[str, Any]
    result: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class RenderStatusResponse(BaseModel):
    job: JobOut | None
