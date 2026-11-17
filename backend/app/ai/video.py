"""Engines that reconstruct a short clip at a time (video inpainting models).

A video model looks at several frames together, so what it fills in is consistent from frame to frame without
help. Long videos are cut into overlapping windows; where two windows overlap their results are cross-faded, so
there is no visible step where one window ends and the next begins.

All frames of a window are cropped to the *same* box (the union of where the marks are in that window), because
models expect a fixed size across time.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from app.ai.base import (
    Frame,
    InpaintingEngine,
    InpaintingError,
    InpaintingOptions,
    Mask,
    paired,
)
from app.ai.consistency import match_colour
from app.ai.roi import PreparedMask, Roi, align_roi, blend_into, prepare_mask

logger = logging.getLogger(__name__)


@dataclass
class _Result:
    frame: Frame
    bounds: Roi | None  # where this frame differs from its source; None when nothing was marked


class VideoInpaintingEngine(InpaintingEngine):
    clip_frames: int = 16  # frames per model call
    overlap_frames: int = 4  # frames shared by consecutive windows
    input_multiple: int = 8

    def __init__(self, clip_frames: int | None = None, overlap_frames: int | None = None) -> None:
        if clip_frames is not None:
            self.clip_frames = clip_frames
        if overlap_frames is not None:
            self.overlap_frames = overlap_frames
        if self.overlap_frames < 0 or self.clip_frames < 2 * self.overlap_frames + 1:
            raise ValueError("clip_frames must be at least 2 * overlap_frames + 1")

    @abstractmethod
    def inpaint_clip(self, crops: list[Frame], masks: list[Mask], options: InpaintingOptions) -> list[Frame]:
        """Return the crops with the marked pixels reconstructed, consistently across the clip.
        Same count and size, uint8 BGR. A mask may be empty (all zero) for frames with nothing to remove."""

    def process(
        self,
        frames: Iterable[Frame],
        masks: Iterable[Mask],
        options: InpaintingOptions,
    ) -> Iterator[Frame]:
        self.require_available()
        pairs = paired(frames, masks)
        carried: list[tuple[Frame, Mask]] = []  # the overlap, taken from the end of the previous window
        tail: list[_Result] = []  # the previous window's results for those frames, not yet emitted
        first_index = 0
        try:
            while True:
                window = list(carried)
                fresh = 0
                exhausted = False
                while len(window) < self.clip_frames:
                    item = next(pairs, None)
                    if item is None:
                        exhausted = True
                        break
                    window.append((item[1], item[2]))
                    fresh += 1
                if fresh == 0:
                    # Everything left was already inside the previous window.
                    yield from (r.frame for r in tail)
                    return
                results = self._run_window(window, options, first_index)
                for i, previous in enumerate(tail):  # cross-fade the overlap
                    results[i] = _crossfade(previous, results[i], (i + 1) / (len(tail) + 1))
                if exhausted:
                    yield from (r.frame for r in results)
                    return
                keep = self.overlap_frames
                yield from (r.frame for r in results[: len(results) - keep])
                first_index += len(results) - keep
                carried = window[len(window) - keep :] if keep else []
                tail = results[len(results) - keep :] if keep else []
        finally:
            self.close()

    def close(self) -> None:
        """Release anything held between windows. Called when processing ends."""

    def _run_window(self, window: list[tuple[Frame, Mask]], options: InpaintingOptions, first_index: int) -> list[_Result]:
        prepared: list[PreparedMask | None] = [prepare_mask(mask, options) for _, mask in window]
        active = [p for p in prepared if p is not None]
        if not active:
            return [_Result(frame.copy(), None) for frame, _ in window]

        height, width = window[0][0].shape[:2]
        padding = max(options.context_padding_px, options.feather_px + 1)
        box = active[0].bounds
        for p in active[1:]:
            box = box.union(p.bounds)
        roi = align_roi(box.expanded(padding, width, height), self.input_multiple, width, height)

        crops = [np.ascontiguousarray(frame[roi.slices]) for frame, _ in window]
        crop_masks = [
            np.ascontiguousarray(p.grown[roi.slices]) if p is not None else np.zeros((roi.height, roi.width), np.uint8)
            for p in prepared
        ]
        try:
            filled = self.inpaint_clip(crops, crop_masks, options)
        except InpaintingError:
            raise
        except Exception as error:
            logger.exception("video inpainting failed in the window starting at frame %d (%s)", first_index, self.provider)
            raise InpaintingError(f"Reconstruction failed around frame {first_index + 1}.") from error
        if not isinstance(filled, list) or len(filled) != len(window):
            raise InpaintingError("The inpainting engine returned the wrong number of frames.")

        results: list[_Result] = []
        for (frame, _), p, crop, crop_mask, patch in zip(window, prepared, crops, crop_masks, filled, strict=True):
            if not isinstance(patch, np.ndarray) or patch.dtype != np.uint8 or patch.shape != crop.shape:
                raise InpaintingError("The inpainting engine returned an image that doesn't match its input.")
            output = frame.copy()
            if p is not None:
                blend_into(output, match_colour(patch, crop, crop_mask), p.alpha, roi)
            results.append(_Result(output, None if p is None else p.bounds))
        return results


def _crossfade(before: _Result, after: _Result, weight: float) -> _Result:
    """Both results were made from the same source frame, so they only differ where the marks are."""
    if before.bounds is None or after.bounds is None:
        return after
    region = before.bounds.union(after.bounds).slices
    mixed = before.frame[region].astype(np.float32) * (1.0 - weight) + after.frame[region].astype(np.float32) * weight
    result = after.frame.copy()
    result[region] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return _Result(result, after.bounds)
