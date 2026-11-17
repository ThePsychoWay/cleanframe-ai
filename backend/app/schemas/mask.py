"""The mask document: what a person marked for removal, in a form later stages can rasterise.

Regions carry keyframes, each holding the shape at one source frame. Coordinates are fractions of the
frame (0 to 1 from the top-left), so they don't depend on the proxy's or the source's resolution.
Brush radii are fractions of the frame *width*.

  rectangle  keyframe.rect = {x, y, w, h}
  polygon    keyframe.points = [[x, y], ...]            (3 or more)
  brush      keyframe.strokes = [{radius, points}, ...] (painted area)

Any keyframe may also carry `erase` strokes, which are subtracted from that keyframe's shape, or be `hidden`: the
region isn't on screen from that frame until its next keyframe.

The limits below keep a saved document small enough to store, load and rasterise predictably.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

MAX_REGIONS = 20
MAX_KEYFRAMES_PER_REGION = 200
MAX_POLYGON_POINTS = 500
MAX_STROKES_PER_KEYFRAME = 500
MAX_POINTS_PER_STROKE = 2000
MAX_TOTAL_POINTS = 50_000
MIN_RADIUS, MAX_RADIUS = 0.001, 0.25

_SHAPE_FIELDS = frozenset({"rect", "points", "strokes"})
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

Coordinate = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Point = Annotated[list[Coordinate], Field(min_length=2, max_length=2)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Stroke(_Strict):
    radius: float = Field(ge=MIN_RADIUS, le=MAX_RADIUS, allow_inf_nan=False)
    points: list[Point] = Field(min_length=1, max_length=MAX_POINTS_PER_STROKE)


class Rect(_Strict):
    x: Coordinate
    y: Coordinate
    w: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    h: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _inside_frame(self) -> Self:
        if self.x + self.w > 1.0 + 1e-6 or self.y + self.h > 1.0 + 1e-6:
            raise ValueError("A rectangle must lie inside the frame.")
        return self


class Keyframe(_Strict):
    frame: int = Field(ge=0)
    rect: Rect | None = None
    points: list[Point] | None = Field(default=None, min_length=3, max_length=MAX_POLYGON_POINTS)
    strokes: list[Stroke] | None = Field(default=None, min_length=1, max_length=MAX_STROKES_PER_KEYFRAME)
    erase: list[Stroke] = Field(default_factory=list, max_length=MAX_STROKES_PER_KEYFRAME)
    # The region isn't on screen from this frame until its next keyframe (an overlay that comes and goes). The keyframe
    # still carries a shape, which is not drawn.
    hidden: bool = False

    @model_serializer(mode="wrap")
    def _omit_unused_shapes(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep documents minimal: one kind of shape per keyframe, and no empty erase list."""
        data: dict[str, Any] = handler(self)
        data = {k: v for k, v in data.items() if not (k in _SHAPE_FIELDS and v is None)}
        if not data.get("hidden"):
            data.pop("hidden", None)
        if not data.get("erase"):
            data.pop("erase", None)
        return data


class Region(_Strict):
    id: str
    name: str = Field(min_length=1, max_length=60)
    type: Literal["rectangle", "polygon", "brush"]
    keyframes: list[Keyframe] = Field(min_length=1, max_length=MAX_KEYFRAMES_PER_REGION)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("Region ids may only contain letters, numbers, dashes and underscores.")
        return value

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = _CONTROL.sub("", value).strip()
        if not cleaned:
            raise ValueError("A region needs a name.")
        return cleaned

    @model_validator(mode="after")
    def _shapes_match_type(self) -> Self:
        shape_field = {"rectangle": "rect", "polygon": "points", "brush": "strokes"}[self.type]
        frames = [k.frame for k in self.keyframes]
        if any(b <= a for a, b in zip(frames, frames[1:], strict=False)):
            raise ValueError("Keyframes must be in order, one per frame.")
        for keyframe in self.keyframes:
            present = {f for f in ("rect", "points", "strokes") if getattr(keyframe, f) is not None}
            if present != {shape_field}:
                raise ValueError(f"A {self.type} region's keyframes must contain only '{shape_field}'.")
        return self


class MaskDocument(_Strict):
    version: Literal[1] = 1
    regions: list[Region] = Field(default_factory=list, max_length=MAX_REGIONS)

    @model_validator(mode="after")
    def _unique_and_bounded(self) -> Self:
        ids = [r.id for r in self.regions]
        if len(set(ids)) != len(ids):
            raise ValueError("Region ids must be unique.")
        if self.total_points() > MAX_TOTAL_POINTS:
            raise ValueError("This mask is too detailed to save. Simplify it or remove some keyframes.")
        return self

    def total_points(self) -> int:
        total = 0
        for region in self.regions:
            for k in region.keyframes:
                total += len(k.points or [])
                total += sum(len(s.points) for s in (k.strokes or []))
                total += sum(len(s.points) for s in k.erase)
        return total

    def frame_span(self) -> tuple[int, int]:
        frames = [k.frame for r in self.regions for k in r.keyframes]
        return (min(frames), max(frames)) if frames else (0, 0)


class MaskResponse(BaseModel):
    mask_data: MaskDocument
    tracking_status: str
    # None until the first save: an empty document is returned for a project with no mask yet.
    updated_at: datetime | None
