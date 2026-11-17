from pydantic import BaseModel


class DeviceResponse(BaseModel):
    kind: str  # cuda | cpu | remote
    name: str
    note: str


class EngineResponse(BaseModel):
    """What the app may say about the inpainting engine. Never includes paths, URLs or keys."""

    provider: str
    name: str
    available: bool
    development_only: bool
    sends_frames_to_third_party: bool
    device: DeviceResponse | None
    reason: str | None
    expectation: str
