"""Which inpainting engine is configured, and whether it can run.

The interface uses this to label development output as such, to warn before picture data is sent to a
third-party service, and to set expectations about speed (GPU or CPU)."""

from __future__ import annotations

from fastapi import APIRouter

from app.ai.registry import describe_engine
from app.auth.deps import CurrentUser
from app.schemas.engine import DeviceResponse, EngineResponse

router = APIRouter(prefix="/api", tags=["engine"])


@router.get("/engine", response_model=EngineResponse)
def engine_status(_: CurrentUser) -> EngineResponse:
    info = describe_engine()
    return EngineResponse(
        provider=info.provider,
        name=info.name,
        available=info.available,
        development_only=info.development_only,
        sends_frames_to_third_party=info.sends_frames_to_third_party,
        device=None if info.device is None else DeviceResponse(kind=info.device.kind, name=info.device.name, note=info.device.note),
        reason=info.reason,
        expectation=info.expectation,
    )
