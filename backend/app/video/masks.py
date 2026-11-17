"""The mask for every frame of the final render, at the video's real resolution.

This is the same "mask at frame N" the editor shows: tracked motion where tracking has run, otherwise the person's
keyframes blended. Shapes are stored as fractions of the frame, so rasterising at full resolution is exact and needs
no scaling from the preview. Regions are combined into one mask per frame ("process masks together").
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from app.schemas.mask import MaskDocument
from app.tracking.raster import rasterize
from app.tracking.shapes import Segment, Shape, segments_from_json, shape_at


class MaskSequence:
    def __init__(
        self,
        document: MaskDocument,
        tracking: dict[str, Any] | None,
        *,
        width: int,
        height: int,
        frame_count: int,
    ) -> None:
        if width < 2 or height < 2 or frame_count < 1:
            raise ValueError("a mask sequence needs a real frame size and at least one frame")
        self.document = document
        self.width, self.height, self.frame_count = width, height, frame_count
        self._segments: dict[str, list[Segment]] = segments_from_json(tracking)
        # Tracked transforms are in the units of the frame they were computed on. Use that frame's aspect ratio.
        aspect = (tracking or {}).get("aspect")
        self._aspect = float(aspect) if isinstance(aspect, (int, float)) and aspect > 0 else height / width
        self._cache: dict[tuple[Shape, ...], np.ndarray] = {}

    @property
    def uses_tracking(self) -> bool:
        return any(self._segments.values())

    def shapes_at(self, frame: int) -> tuple[Shape, ...]:
        return tuple(shape_at(region, frame, self._segments.get(region.id), self._aspect) for region in self.document.regions)

    def mask(self, frame: int) -> np.ndarray:
        """uint8 HxW, 255 where something is to be removed on this frame. Don't modify the result: static marks share it."""
        shapes = self.shapes_at(frame)
        cached = self._cache.get(shapes)
        if cached is not None:
            return cached
        combined = np.zeros((self.height, self.width), np.uint8)
        for shape in shapes:
            np.maximum(combined, rasterize(shape, self.width, self.height), out=combined)
        if len(self._cache) >= 4:  # a static mark repeats every frame; a moving one never does
            self._cache.pop(next(iter(self._cache)))
        self._cache[shapes] = combined
        return combined

    def __iter__(self) -> Iterator[np.ndarray]:
        for frame in range(self.frame_count):
            yield self.mask(frame)
