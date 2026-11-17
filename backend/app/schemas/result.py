from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PreviewResponse(BaseModel):
    """Time-limited links to the restored video and to the preview copy of the original, for the compare view.

    Anyone holding a link can read that file until it expires."""

    result_url: str
    original_url: str
    expires_in: int
    # The restored video's real size and format. The original shown beside it is the lower-resolution preview copy
    # (silent, same frames), so the compare view fits both to the same picture box.
    width: int
    height: int
    fps: float
    frame_count: int
    container: str
    size_bytes: int | None
    audio: str | None
    development_only: bool
    warnings: list[str]
    finished_at: datetime | None
    facts: dict[str, Any]


class ExportResponse(BaseModel):
    """A short-lived link that downloads the restored video."""

    url: str
    filename: str
    expires_in: int
    size_bytes: int | None
