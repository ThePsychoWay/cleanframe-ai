"""Work only where the mask is.

A logo covers a few percent of a frame. Giving a model the whole frame costs time and invites it to
change things nobody asked about, so engines reconstruct *regions of interest*: the masked area plus some
context around it, cropped out, processed, and blended back. Everything here is plain NumPy/OpenCV and shared
by every engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.ai.base import MASK_THRESHOLD, Frame, InpaintingOptions, Mask

# Merging stops being worth the bookkeeping beyond this many separate pieces (e.g. a speckled brush mask):
# they are treated as one region.
MAX_REGIONS = 32


@dataclass(frozen=True)
class Roi:
    """A rectangle in frame pixels. x1 and y1 are exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def expanded(self, pixels: int, width: int, height: int) -> Roi:
        return Roi(max(0, self.x0 - pixels), max(0, self.y0 - pixels), min(width, self.x1 + pixels), min(height, self.y1 + pixels))

    def intersects(self, other: Roi) -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1

    def union(self, other: Roi) -> Roi:
        return Roi(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))


@dataclass(frozen=True)
class PreparedMask:
    """A frame's mask, grown a little, with the soft weights used to blend the reconstruction back."""

    grown: Mask  # full frame, uint8 0/255: what the engine is asked to reconstruct
    alpha: np.ndarray  # full frame, float32 0..1: how much of the reconstruction to use at each pixel
    bounds: Roi  # smallest box containing every pixel with alpha > 0


def _bounds_of(binary: np.ndarray) -> Roi | None:
    rows = np.flatnonzero(binary.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(binary.any(axis=0))
    return Roi(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def prepare_mask(mask: Mask, options: InpaintingOptions) -> PreparedMask | None:
    """Grow the mask by edge_dilate_px and add a feathered rim. None when nothing is marked."""
    binary = mask > MASK_THRESHOLD
    box = _bounds_of(binary)
    if box is None:
        return None
    height, width = mask.shape
    reach = options.reach_px
    window = box.expanded(reach + 2, width, height)  # all work happens inside this box

    sub = binary[window.slices].astype(np.uint8) * 255
    if options.edge_dilate_px > 0:
        size = 2 * options.edge_dilate_px + 1
        sub = cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    alpha_sub = (sub > 0).astype(np.float32)
    if options.feather_px > 0:
        # Distance (in pixels) from each pixel outside the grown mask to the mask: 1 at the edge, 0 at feather+1.
        distance = cv2.distanceTransform(255 - sub, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        rim = np.clip(1.0 - distance / (options.feather_px + 1), 0.0, 1.0)
        alpha_sub = np.where(sub > 0, 1.0, rim).astype(np.float32)

    grown = np.zeros(mask.shape, np.uint8)
    alpha = np.zeros(mask.shape, np.float32)
    grown[window.slices] = sub
    alpha[window.slices] = alpha_sub
    bounds = _bounds_of(alpha > 0)
    assert bounds is not None  # the mask had pixels, so the grown mask does too
    return PreparedMask(grown=grown, alpha=alpha, bounds=bounds)


def align_roi(roi: Roi, multiple: int, width: int, height: int) -> Roi:
    """Grow a box so its size is a multiple (models often need that), staying inside the frame when it can."""
    if multiple <= 1:
        return roi

    def fit(start: int, stop: int, limit: int) -> tuple[int, int]:
        size = stop - start
        wanted = -(-size // multiple) * multiple
        if wanted > limit:  # the frame itself isn't a multiple: take all of it, the engine will pad
            return 0, limit
        extra = wanted - size
        start -= extra // 2
        stop = start + wanted
        if start < 0:
            start, stop = 0, wanted
        if stop > limit:
            start, stop = limit - wanted, limit
        return start, stop

    x0, x1 = fit(roi.x0, roi.x1, width)
    y0, y1 = fit(roi.y0, roi.y1, height)
    return Roi(x0, y0, x1, y1)


def plan_regions(prepared: PreparedMask, options: InpaintingOptions, *, multiple: int = 1) -> list[Roi]:
    """Boxes (mask plus context) to reconstruct. Marks that are far apart get separate boxes, so a corner
    logo and a timestamp don't force the model to look at the whole frame. Overlapping boxes are merged."""
    height, width = prepared.grown.shape
    padding = max(options.context_padding_px, options.feather_px + 1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(prepared.grown, connectivity=8)
    boxes = [
        Roi(int(x), int(y), int(x + w), int(y + h))
        for x, y, w, h in stats[1:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
    ]
    # The feathered rim extends past the grown mask, so the boxes must too.
    boxes = [b.expanded(options.feather_px + 1, width, height) for b in boxes]
    padded = [b.expanded(padding, width, height) for b in boxes]
    if len(padded) > MAX_REGIONS:
        padded = [prepared.bounds.expanded(padding, width, height)]

    merged = True
    while merged and len(padded) > 1:
        merged = False
        for i in range(len(padded)):
            for j in range(i + 1, len(padded)):
                if padded[i].intersects(padded[j]):
                    padded[i] = padded[i].union(padded[j])
                    del padded[j]
                    merged = True
                    break
            if merged:
                break
    return [align_roi(roi, multiple, width, height) for roi in padded]


def blend_into(output: Frame, filled: Frame, alpha: np.ndarray, roi: Roi) -> None:
    """Write a reconstructed crop into `output` in place, weighted by alpha. Where alpha is 0 the output is
    not touched at all, so pixels away from the mask stay bit-identical to the input."""
    window = roi.slices
    a = alpha[window]
    touched = a > 0
    if not touched.any():
        return
    target = output[window]
    full = a >= 1.0
    target[full] = filled[full]
    partial = touched & ~full
    if partial.any():
        weight = a[partial][:, None]
        mixed = target[partial].astype(np.float32) * (1.0 - weight) + filled[partial].astype(np.float32) * weight
        target[partial] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
