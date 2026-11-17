"""Turn a Shape into a binary mask at a given resolution (255 inside the region, 0 outside)."""

from __future__ import annotations

import cv2
import numpy as np

from app.tracking.shapes import Shape, Stroke

_SHIFT = 4  # sub-pixel precision for OpenCV's fixed-point drawing
_ONE = 1 << _SHIFT


def _fixed(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return round(x * width * _ONE), round(y * height * _ONE)


def _draw_stroke(mask: np.ndarray, stroke: Stroke, value: int) -> None:
    height, width = mask.shape
    radius = max(0.5, stroke.radius * width)
    radius_fixed = round(radius * _ONE)
    points = [_fixed(p[0], p[1], width, height) for p in stroke.points]
    for x, y in points:
        cv2.circle(mask, (x, y), radius_fixed, value, thickness=-1, lineType=cv2.LINE_8, shift=_SHIFT)
    for a, b in zip(points, points[1:], strict=False):
        # Round caps are already covered by the circles; the line fills the gap between them.
        cv2.line(mask, a, b, value, thickness=max(1, round(radius * 2)), lineType=cv2.LINE_8, shift=_SHIFT)


def rasterize(shape: Shape, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if shape.hidden:
        return mask
    if shape.rect:
        x, y, w, h = shape.rect
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        polygon = np.array([_fixed(px, py, width, height) for px, py in corners], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_8, shift=_SHIFT)
    elif shape.points:
        polygon = np.array([_fixed(px, py, width, height) for px, py in shape.points], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_8, shift=_SHIFT)
    elif shape.strokes:
        for stroke in shape.strokes:
            _draw_stroke(mask, stroke, 255)
    for stroke in shape.erase:
        _draw_stroke(mask, stroke, 0)
    return mask
