"""Colour and temporal consistency, shared by every engine.

Two separate problems:

* **Colour/lighting**: a model can return a patch that is a little brighter, darker or tinted compared with
  what surrounds it. `match_colour` compares the edge of the patch with the pixels just outside the mask and
  removes a systematic difference. Small differences are left alone, because correcting noise would add flicker.

* **Temporal**: engines that fill each frame on its own produce a slightly different patch every frame, which
  shows as shimmer. `TemporalStabilizer` finds how the picture moved between two frames (from the pixels around
  the mask, which are real), moves the previous result along with it, and blends it into the new one. That damps
  frame-to-frame variation without smearing the patch across a moving picture. It does not invent motion
  inside the patch: if something moves *behind* the logo, only a model that understands video can restore it.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.base import Frame
from app.ai.roi import PreparedMask

# ---------------------------------------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------------------------------------

BAND_PX = 4  # how far in from the mask edge, and out from it, the two sides are compared
MIN_BAND_PIXELS = 24
COLOUR_DEADBAND = 6.0  # average difference (0-255 levels) below which nothing is changed
MAX_COLOUR_SHIFT = 40.0


def match_colour(filled: Frame, source: Frame, grown: np.ndarray) -> Frame:
    """Remove a constant colour offset between the edge of the reconstruction and its surroundings.

    `filled` and `source` are crops of the same area; `grown` is the crop's 0/255 mask. Returns `filled`
    itself when no correction is needed."""
    mask = grown > 0
    if not mask.any() or mask.all():
        return filled
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_PX + 1, 2 * BAND_PX + 1))
    inner = mask & ~(cv2.erode(grown, kernel) > 0)  # the edge of the patch, inside the mask
    outer = (cv2.dilate(grown, kernel) > 0) & ~mask  # the real pixels just outside it
    if inner.sum() < MIN_BAND_PIXELS or outer.sum() < MIN_BAND_PIXELS:
        return filled

    shift = source[outer].mean(axis=0) - filled[inner].mean(axis=0)  # per channel, BGR
    if abs(float(shift.mean())) < COLOUR_DEADBAND and float(np.abs(shift).max()) < 2 * COLOUR_DEADBAND:
        return filled
    shift = np.clip(shift, -MAX_COLOUR_SHIFT, MAX_COLOUR_SHIFT)
    corrected = filled.copy()
    corrected[mask] = np.clip(np.rint(filled[mask].astype(np.float32) + shift), 0, 255).astype(np.uint8)
    return corrected


SEAM_PAD = 2  # px of untouched picture added round the patch so the solver always has real pixels at its border


def match_seam(filled: Frame, source: Frame, grown: np.ndarray) -> Frame:
    """Make a generated patch continuous with the real pixels around it, keeping the texture the model made inside.

    A generated patch is often a little lighter, darker or tinted against its surroundings, and not by one constant amount, so
    its outline shows as a faint box. This solves for the patch whose *gradients* are the generated ones and whose edge equals
    the real pixels just outside it (Poisson blending): a smooth correction that is zero-effort for the interior and exact at
    the edge. `filled` is the crop with the patch painted in, `source` the same crop without it, `grown` the patch's 0/255 mask.
    Falls back to `match_colour` where the patch touches the crop's border (no real pixels to match there)."""
    mask = grown > 0
    if not mask.any() or mask.all():
        return filled
    height, width = mask.shape
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if x < SEAM_PAD + 1 or y < SEAM_PAD + 1 or x + w > width - SEAM_PAD - 1 or y + h > height - SEAM_PAD - 1:
        return match_colour(filled, source, grown)
    pad = SEAM_PAD
    patch = cv2.copyMakeBorder(np.ascontiguousarray(filled[y : y + h, x : x + w]), pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    cut = cv2.copyMakeBorder(np.ascontiguousarray(grown[y : y + h, x : x + w]), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    try:
        blended = cv2.seamlessClone(patch, np.ascontiguousarray(source), np.where(cut > 0, 255, 0).astype(np.uint8), (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
    except cv2.error:
        return match_colour(filled, source, grown)
    result = filled.copy()
    result[mask] = blended[mask]
    return result


# ---------------------------------------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------------------------------------

MOTION_WIDTH = 480  # motion is estimated on a copy this wide: cheap, and enough for a global shift
NEAR_PX = 160  # use pixels within this distance of the mask to estimate motion (at MOTION_WIDTH scale)
MIN_MOTION_POINTS = 10
STATIC_DIFFERENCE = 6.0  # mean difference between consecutive frames below which the picture is "still"
CONTEXT_PX = 20  # real pixels around the mask used to check that the motion estimate is right
MIN_CONTEXT_PIXELS = 80


class TemporalStabilizer:
    """Blends each reconstructed patch with the previous one, carried along with the picture's motion."""

    def __init__(self, *, strength: float = 0.6, max_context_error: float = 8.0) -> None:
        if not 0.0 <= strength < 1.0:
            raise ValueError("strength must be in [0, 1)")
        self.strength = strength
        self.max_context_error = max_context_error
        self.reset()

    def reset(self) -> None:
        self._gray: np.ndarray | None = None  # previous frame, small, grayscale
        self._output: Frame | None = None  # previous result, full size
        self._alpha: np.ndarray | None = None  # previous blend weights, full size (None = nothing was filled)
        self._mask_small: np.ndarray | None = None

    # -- public -----------------------------------------------------------------------------------------

    def skip(self, source: Frame) -> None:
        """A frame with nothing to remove: remember it so the next frame can reuse its real pixels."""
        small, _ = self._small_gray(source)
        self._remember(small, source, None, None)

    def stabilize(self, source: Frame, prepared: PreparedMask, output: Frame) -> Frame:
        """Return `output` with the patch blended towards the previous frame's. `output` is modified in place."""
        small, scale = self._small_gray(source)
        previous_ok = (
            self._gray is not None
            and self._output is not None
            and self._output.shape == output.shape
            and self._gray.shape == small.shape
        )
        if previous_ok:
            matrix = self._estimate_motion(small, scale, prepared)
            if matrix is not None:
                self._blend(source, prepared, output, matrix)
        self._remember(small, output, prepared.alpha, prepared.grown)
        return output

    # -- internals --------------------------------------------------------------------------------------

    def _remember(self, small: np.ndarray, output: Frame, alpha: np.ndarray | None, grown: np.ndarray | None) -> None:
        self._gray = small
        self._output = output.copy()
        self._alpha = alpha
        self._mask_small = None if grown is None else cv2.resize(grown, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _small_gray(frame: Frame) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        scale = min(1.0, MOTION_WIDTH / width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
        return gray, scale

    def _estimate_motion(self, current: np.ndarray, scale: float, prepared: PreparedMask) -> np.ndarray | None:
        """2x3 similarity mapping previous-frame pixels to current-frame pixels, or None if the two frames
        can't be related (a cut, or too little to follow). Only pixels away from the marks are used."""
        assert self._gray is not None
        previous = self._gray
        height, width = current.shape
        exclude = cv2.resize(prepared.grown, (width, height), interpolation=cv2.INTER_NEAREST)
        if self._mask_small is not None:
            exclude = cv2.max(exclude, self._mask_small)
        exclude = cv2.dilate(exclude, np.ones((7, 7), np.uint8))

        # Look near the marks: that is the motion that matters for the patch.
        near = np.zeros_like(exclude)
        box = prepared.bounds
        x0, y0 = int(box.x0 * scale), int(box.y0 * scale)
        x1, y1 = int(np.ceil(box.x1 * scale)), int(np.ceil(box.y1 * scale))
        near[max(0, y0 - NEAR_PX) : y1 + NEAR_PX, max(0, x0 - NEAR_PX) : x1 + NEAR_PX] = 255
        allowed = cv2.bitwise_and(near, cv2.bitwise_not(exclude))

        difference = float(np.mean(cv2.absdiff(previous, current)[allowed > 0])) if (allowed > 0).any() else 255.0
        points = cv2.goodFeaturesToTrack(previous, maxCorners=200, qualityLevel=0.01, minDistance=5, mask=allowed)
        if points is not None and len(points) >= MIN_MOTION_POINTS:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(15, 15), maxLevel=3)
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, winSize=(15, 15), maxLevel=3)
            good = (
                (status.ravel() == 1)
                & (back_status.ravel() == 1)
                & (np.linalg.norm(back - points, axis=2).ravel() < 1.2)
            )
            if good.sum() >= MIN_MOTION_POINTS:
                matrix, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=1.0)
                if matrix is not None and inliers is not None and int(inliers.sum()) >= MIN_MOTION_POINTS:
                    # Convert to full-resolution pixel units: x_full = x_small / scale.
                    full = matrix.copy()
                    full[:, 2] /= scale
                    return full
        # Nothing to follow. A still picture (flat sky, black bars) is fine as it is; anything else is a cut.
        if difference < STATIC_DIFFERENCE:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return None

    def _blend(self, source: Frame, prepared: PreparedMask, output: Frame, matrix: np.ndarray) -> None:
        assert self._output is not None
        height, width = output.shape[:2]
        area = prepared.bounds.expanded(CONTEXT_PX, width, height)
        # Moving the previous result into the current frame, only over the area we need.
        shifted = matrix.copy()
        shifted[:, 2] -= (area.x0, area.y0)
        size = (area.width, area.height)
        moved = cv2.warpAffine(self._output, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if self._alpha is not None:
            moved_alpha = cv2.warpAffine(self._alpha, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            moved_alpha = np.zeros((area.height, area.width), np.float32)
        # Out-of-frame sources would be smeared edge pixels: don't trust them.
        inside = cv2.warpAffine(
            np.full(self._output.shape[:2], 255, np.uint8), shifted, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        current_alpha = prepared.alpha[area.slices]
        # Is the motion estimate right? Compare real pixels around the mask: the moved previous frame against this one.
        context = (current_alpha == 0) & (moved_alpha < 0.01) & (inside > 0)
        if int(context.sum()) < MIN_CONTEXT_PIXELS:
            return
        error = float(np.abs(moved[context].astype(np.float32) - source[area.slices][context].astype(np.float32)).mean())
        if error > self.max_context_error:
            return

        weight = (self.strength * current_alpha * (inside > 0))[..., None]
        target = output[area.slices]
        active = weight[..., 0] > 0
        mixed = target[active].astype(np.float32) * (1.0 - weight[active]) + moved[active].astype(np.float32) * weight[active]
        target[active] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def flicker(frames: list[Frame], region: np.ndarray) -> float:
    """Mean absolute change, per pixel and channel, of `region` (bool HxW) between consecutive frames.
    A measure of shimmer; compare it with the same number for the clean original footage."""
    if len(frames) < 2 or not region.any():
        return 0.0
    total = 0.0
    for before, after in zip(frames, frames[1:], strict=False):
        total += float(np.abs(after[region].astype(np.float32) - before[region].astype(np.float32)).mean())
    return total / (len(frames) - 1)

