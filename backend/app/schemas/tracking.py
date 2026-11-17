from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class TrackingResponse(BaseModel):
    """Where tracking stands. `data` (the tracked motion, see app.tracking.shapes) comes only with 'done'."""

    status: str  # none | running | done | failed
    frames_done: int
    frames_total: int
    error: str | None
    started_at: datetime | None
    data: dict[str, Any] | None = None
