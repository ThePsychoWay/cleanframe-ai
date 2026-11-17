"""What a region looks like on any frame: keyframes, blending between them, and tracked motion.

This is the single definition of "the mask at frame N" that later stages (inpainting, export) use.
The editor has a TypeScript mirror (frontend/lib/tracking.ts); tests/fixtures/shape_cases.json is run
against both so they can't drift apart.

Tracking stores, for each frame after a keyframe, one small similarity transform of that keyframe's
shape: [dx, dy, scale, angle]. Distances are in *frame-width units* (x and y both divided by the
frame width) so a rotation doesn't stretch on a non-square frame. The transform acts about the centre
of the keyframe's bounding box:

    p' = (centre + (dx, dy)) + scale * R(angle) * (p - centre)

Rectangles are tracked without rotation (angle is always 0) so a tracked rectangle stays a rectangle
and can be edited like any other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.schemas.mask import Keyframe, Region

Point = tuple[float, float]
IDENTITY = (0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class Stroke:
    radius: float
    points: tuple[Point, ...]


@dataclass(frozen=True)
class Shape:
    """One region's geometry on one frame. Exactly one of rect, points or strokes is set."""

    rect: tuple[float, float, float, float] | None = None  # x, y, w, h
    points: tuple[Point, ...] | None = None
    strokes: tuple[Stroke, ...] | None = None
    erase: tuple[Stroke, ...] = field(default_factory=tuple)
    hidden: bool = False  # not on screen: the geometry is kept but isn't drawn


def shape_from_keyframe(keyframe: Keyframe) -> Shape:
    def strokes(items: list[Any]) -> tuple[Stroke, ...]:
        return tuple(Stroke(s.radius, tuple((p[0], p[1]) for p in s.points)) for s in items)

    return Shape(
        rect=(keyframe.rect.x, keyframe.rect.y, keyframe.rect.w, keyframe.rect.h) if keyframe.rect else None,
        points=tuple((p[0], p[1]) for p in keyframe.points) if keyframe.points else None,
        strokes=strokes(keyframe.strokes) if keyframe.strokes else None,
        erase=strokes(keyframe.erase),
        hidden=keyframe.hidden,
    )


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def blended_shape(region: Region, frame: int) -> Shape:
    """Keyframes only, no tracking. Mirrors shapeAt in frontend/lib/masks.ts."""
    keyframes = region.keyframes
    previous = keyframes[0]
    following: Keyframe | None = None
    for keyframe in keyframes:
        if keyframe.frame <= frame:
            previous = keyframe
        else:
            following = keyframe
            break
    base = shape_from_keyframe(previous)
    if frame <= previous.frame or following is None or base.hidden or following.hidden:
        return base  # a region that appears or vanishes does so at its keyframe: nothing is blended with "not there"

    t = (frame - previous.frame) / (following.frame - previous.frame)
    if region.type == "rectangle" and base.rect and following.rect:
        a, b = base.rect, (following.rect.x, following.rect.y, following.rect.w, following.rect.h)
        return Shape(rect=tuple(_lerp(a[i], b[i], t) for i in range(4)), erase=base.erase)  # type: ignore[arg-type]
    if region.type == "polygon" and base.points and following.points and len(base.points) == len(following.points):
        points = tuple(
            (_lerp(p[0], q[0], t), _lerp(p[1], q[1], t)) for p, q in zip(base.points, following.points, strict=True)
        )
        return Shape(points=points, erase=base.erase)
    return base


def shape_center(shape: Shape) -> Point:
    """Centre of the shape's bounding box in normalised coordinates (brush radius isn't included)."""
    if shape.rect:
        x, y, w, h = shape.rect
        return (x + w / 2, y + h / 2)
    if shape.points:
        points: list[Point] = list(shape.points)
    else:
        points = [p for s in (shape.strokes or ()) for p in s.points]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def _transform_point(p: Point, centre: Point, params: tuple[float, ...], aspect: float) -> Point:
    dx, dy, scale, angle = params
    cos, sin = math.cos(angle), math.sin(angle)
    # Work in frame-width units: y is scaled by the frame's height/width ratio.
    rx, ry = (p[0] - centre[0]), (p[1] - centre[1]) * aspect
    x = centre[0] + dx + scale * (cos * rx - sin * ry)
    y = (centre[1] * aspect + dy + scale * (sin * rx + cos * ry)) / aspect
    return (x, y)


def transform_shape(shape: Shape, params: tuple[float, ...], aspect: float) -> Shape:
    """Move a keyframe's shape by one tracked transform (see the module docstring)."""
    if tuple(params) == IDENTITY:
        return shape
    centre = shape_center(shape)
    dx, dy, scale, _ = params

    def strokes(items: tuple[Stroke, ...]) -> tuple[Stroke, ...]:
        return tuple(
            Stroke(s.radius * scale, tuple(_transform_point(p, centre, params, aspect) for p in s.points)) for s in items
        )

    if shape.rect:
        x, y, w, h = shape.rect
        cx, cy = centre[0] + dx, centre[1] + dy / aspect
        nw, nh = w * scale, h * scale
        # A tracked rectangle that drifts partly off screen is clipped to the frame.
        left, top = max(0.0, cx - nw / 2), max(0.0, cy - nh / 2)
        right, bottom = min(1.0, cx + nw / 2), min(1.0, cy + nh / 2)
        return Shape(rect=(left, top, max(0.0, right - left), max(0.0, bottom - top)), erase=strokes(shape.erase))
    if shape.points:
        return Shape(
            points=tuple(_transform_point(p, centre, params, aspect) for p in shape.points),
            erase=strokes(shape.erase),
        )
    return Shape(strokes=strokes(shape.strokes or ()), erase=strokes(shape.erase))


# --- tracking results ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """Tracked motion for one region, starting from the keyframe on `anchor`.

    `step` is +1 for frames after the anchor, -1 for frames before it. `until` is the last frame the
    segment is responsible for (just before the next keyframe, or the end/start of the video).
    transforms[i] is the transform at frame anchor + step * (i + 1). When tracking is lost,
    `transforms` stops early, `lost_at` says where, and the last transform holds up to `until`.
    """

    anchor: int
    step: int
    until: int
    transforms: tuple[tuple[float, float, float, float], ...]
    lost_at: int | None = None
    reason: str | None = None

    def covers(self, frame: int) -> bool:
        distance = (frame - self.anchor) * self.step
        return 1 <= distance <= abs(self.until - self.anchor)

    def transform_at(self, frame: int) -> tuple[float, float, float, float]:
        distance = (frame - self.anchor) * self.step
        if not self.transforms:
            return IDENTITY
        return self.transforms[min(distance, len(self.transforms)) - 1]


def segments_from_json(data: dict[str, Any] | None) -> dict[str, list[Segment]]:
    """Parse the stored tracking document into segments per region id."""
    result: dict[str, list[Segment]] = {}
    if not data:
        return result
    for region_id, entry in (data.get("regions") or {}).items():
        result[region_id] = [
            Segment(
                anchor=s["anchor"],
                step=s["step"],
                until=s["until"],
                transforms=tuple(tuple(t) for t in s["transforms"]),  # type: ignore[misc]
                lost_at=s.get("lost_at"),
                reason=s.get("reason"),
            )
            for s in entry.get("segments", [])
        ]
    return result


def shape_at(region: Region, frame: int, segments: list[Segment] | None, aspect: float) -> Shape:
    """The region's shape on `frame`: a keyframe exactly, else tracked motion, else blended keyframes."""
    for keyframe in region.keyframes:
        if keyframe.frame == frame:
            return shape_from_keyframe(keyframe)
    for segment in segments or ():
        if segment.covers(frame):
            anchor = next(k for k in region.keyframes if k.frame == segment.anchor)
            return transform_shape(shape_from_keyframe(anchor), segment.transform_at(frame), aspect)
    return blended_shape(region, frame)
