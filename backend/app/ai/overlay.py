"""Taking a semi-transparent overlay off by inverting it, which restores the real pixels.

A logo laid over a video at partial opacity changes every pixel under it in a known way:

    shown = A * W + (1 - A) * original            A: opacity at that pixel, W: the logo's colour

If A and W are known, the original is `(shown - A * W) / (1 - A)`: exact, nothing invented, and it works on a still
camera, where no other frame shows what is behind the logo. A and W are not known, but they are the same in every
frame while the picture behind them changes, and that is enough to measure them from the clip itself (the idea
of Dekel et al., "On the Effectiveness of Visible Watermarks"):

1. The median over time of the picture's gradients is the logo's own edges: the changing picture averages out.
2. Integrating those edges (a small Poisson solve, anchored to pixels known to be clean) gives the logo's
   contrast against the picture, Q = A * (W - median picture).
3. With a candidate colour W, opacity follows per pixel from Q and the median frame. The right W is the one for
   which the inverted frames show no trace of the logo's edges and stay inside the 0..255 range.

What this needs, and checks before it touches anything:
* the mark is fixed and the picture under it changes over the sampled frames (a still title card, or a picture that
  is the same in every frame, can't be separated from the logo, so the render stops and says so);
* the overlay is one colour (white, black, grey or tinted) at constant opacity with soft edges; logos with several
  colours, gradients or full opacity fail the checks and the render stops (a fully opaque logo hides the picture, nothing can be inverted);
* the whole logo lies inside the marked area.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice

import cv2
import numpy as np

from app.ai.base import (
    DeviceInfo,
    Frame,
    InpaintingEngine,
    InpaintingError,
    InpaintingOptions,
    Mask,
    paired,
)
from app.ai.roi import Roi, prepare_mask

logger = logging.getLogger(__name__)

SAMPLE_BYTES = 256 * 1024 * 1024  # memory for the frames held while the overlay is measured
MIN_SAMPLE, MAX_SAMPLE = 16, 60
MARGIN_PX = 24  # clean pixels around the mark that the measurement is anchored to
CLEAN_GAP_PX = 6  # pixels this far outside the mark are taken to be free of the logo
MAX_ROI_PIXELS = 1_500_000
BYTES_PER_ROI_PIXEL_FRAME = 64  # measured: about 3.6 KB per ROI pixel for 60 frames (a handful of float32 stacks)
DEFAULT_MEMORY_MB = 1024
MIN_PICTURE_CHANGE = 4.0  # median temporal standard deviation (levels) under the mark: below this the logo can't be told from the picture
MIN_OPACITY, MAX_OPACITY = 0.06, 0.90
MAX_EDGE_RATIO = 2.5  # inverted frames may have at most this much more edge energy along the logo's outline than clean areas have
MAX_LEFTOVER = 0.30  # share of the logo's edge signal that may remain after inverting it
LAST_CHECK: dict[str, float] = {}  # the last validation's readings, for tests and calibration
GAIN_GRID = np.linspace(0.0, 1.6, 9)  # how strong the logo is in a frame, relative to the measured model (1 = as measured)
GAIN_SNAP_ONE = 0.12  # a frame within this of 1 is inverted exactly as measured (so a constant logo stays exact)
GAIN_SNAP_ZERO = 0.10  # at or below this the logo is taken to be absent from the frame and the frame is left alone
MAX_APPLIED_OPACITY = 0.95
LUMA = np.array([0.114, 0.587, 0.299], np.float32)  # BGR weights
MAX_CLIPPED = 0.01  # share of inverted pixels allowed outside 0..255 (a wrong overlay model pushes pixels out of range)
SEARCH_FRAMES = 16
EVIDENCE_LEVELS = 4.0  # contrast (levels) at which the measured logo contrast counts as real
MIN_EVIDENCE_COVERED = 0.9
APPLY_THRESHOLD = 0.02  # opacity below this is treated as "no overlay here"


@dataclass(frozen=True)
class OverlayModel:
    roi: Roi
    mask: np.ndarray  # full-frame binary mask the model was measured for
    alpha: np.ndarray  # ROI-sized float32 opacity per pixel
    colour: np.ndarray  # the overlay's colour (B, G, R), 0..255
    opacity: float  # the strongest opacity found
    edge_ratio: float
    clipped: float
    outline: np.ndarray | None = None  # ROI-sized bool: pixels around the logo's edges, where its presence shows


class OverlayRejected(Exception):
    """The overlay couldn't be measured well enough to invert. The message says why, in plain words."""


# -- the measurement ------------------------------------------------------------------------------------


def _grad(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return q[:, 1:] - q[:, :-1], q[1:, :] - q[:-1, :]


def _grad_t(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    out = np.zeros((gy.shape[0] + 1, gx.shape[1] + 1), np.float32)
    out[:, 1:] += gx
    out[:, :-1] -= gx
    out[1:, :] += gy
    out[:-1, :] -= gy
    return out


def integrate_gradients(gx: np.ndarray, gy: np.ndarray, pinned: np.ndarray, *, weight: float = 1000.0, iterations: int = 1500) -> np.ndarray:
    """Least-squares integration of a gradient field, pinned to zero where `pinned` is true (conjugate gradients)."""
    anchor = pinned.astype(np.float32) * weight
    b = _grad_t(gx, gy)
    x = np.zeros_like(b)

    def apply(q: np.ndarray) -> np.ndarray:
        return _grad_t(*_grad(q)) + anchor * q

    r = b - apply(x)
    p = r.copy()
    rs = float((r * r).sum())
    stop = 1e-12 * max(float((b * b).sum()), 1e-9)
    for _ in range(iterations):
        if rs < stop:
            break
        ap = apply(p)
        step = rs / max(float((p * ap).sum()), 1e-20)
        x += step * p
        r -= step * ap
        rs_new = float((r * r).sum())
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _median_over_time(stack: np.ndarray) -> np.ndarray:
    """Median along the first axis. Sorting that axis is about 3x faster than np.median on stacks of a few dozen
    frames (the same result when nothing is NaN, which is the case for pictures)."""
    ordered = np.sort(stack, axis=0)
    n = stack.shape[0]
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _opacity(q: np.ndarray, median: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Per-pixel opacity implied by the logo's contrast Q, the median picture and a candidate overlay colour."""
    denominator = colour - median + q
    ratio = np.where(np.abs(denominator) > 1.0, q / np.where(denominator == 0, 1.0, denominator), 0.0)
    return np.clip(ratio.mean(axis=-1), 0.0, 0.95).astype(np.float32)


def _invert(frames: np.ndarray, alpha: np.ndarray, colour: np.ndarray) -> np.ndarray:
    return (frames - (alpha[..., None] * colour)[None]) / (1.0 - alpha)[None, :, :, None]


def _edge_energy(frames: np.ndarray, region: np.ndarray) -> float:
    inside = region[:-1, :-1]
    if not inside.any():
        return float("nan")
    weight = inside.astype(np.float32)  # a weighted sum is much faster than indexing every frame with a boolean mask
    gx = np.abs(np.diff(frames, axis=2)).mean(axis=-1)[:, :-1, :].sum(axis=0)
    gy = np.abs(np.diff(frames, axis=1)).mean(axis=-1)[:, :, :-1].sum(axis=0)
    return float(((gx + gy) * weight).sum() / (weight.sum() * frames.shape[0]))


def _ghost_level(restored: np.ndarray, region: np.ndarray) -> float:
    """Mean size of the median-over-time gradient inside `region`: the picture changes, so a logo that is still there
    is what is left, as edges that stay put."""
    rows, cols = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
    if rows.size == 0:
        return float("nan")
    box = (slice(max(int(rows[0]) - 1, 0), int(rows[-1]) + 2), slice(max(int(cols[0]) - 1, 0), int(cols[-1]) + 2))
    part, inside = restored[:, box[0], box[1]], region[box]
    gx = np.abs(_median_over_time(np.diff(part, axis=2))).mean(axis=-1)
    gy = np.abs(_median_over_time(np.diff(part, axis=1))).mean(axis=-1)
    return float(np.concatenate([gx[inside[:, :-1]], gy[inside[:-1]]]).mean())


def _ghost(restored: np.ndarray, region: np.ndarray, reference_level: float) -> float:
    return _ghost_level(restored, region) / max(reference_level, 1e-6)


def _score(
    colour: np.ndarray, q: np.ndarray, median: np.ndarray, frames: np.ndarray, clean_reference: float, evidence: np.ndarray, ghost_reference: float
) -> tuple[float, float, float, np.ndarray] | None:
    alpha = _opacity(q, median, colour)
    support = (alpha > 0.03).astype(np.uint8)
    if int(support.sum()) < 20:
        return None
    if float((support > 0)[evidence].mean()) < MIN_EVIDENCE_COVERED:
        return None  # this colour can't account for the contrast the logo was seen to have (wrong sign or size)
    outline = (cv2.dilate(support, np.ones((5, 5), np.uint8)) - cv2.erode(support, np.ones((5, 5), np.uint8))) > 0
    restored = _invert(frames, alpha, colour)
    energy = _edge_energy(restored, outline)
    clipped = float(((restored < -6.0) | (restored > 261.0))[:, support > 0].mean())
    ratio = energy / max(clean_reference, 1e-6)
    # The ghost measure is what tells a right colour from a nearly right one, so it decides between candidates.
    ghost = _ghost(restored, outline | (cv2.dilate(support, np.ones((9, 9), np.uint8)) > 0), ghost_reference)
    return ghost + 0.2 * ratio + 50.0 * clipped, ratio, clipped, alpha


MIN_REFINE_FRAMES = 12  # frames that must still show the logo for the measurement to be repeated without the ones that don't


def _outline_of(alpha: np.ndarray) -> np.ndarray:
    support = (alpha > 0.03).astype(np.uint8)
    ring = np.ones((5, 5), np.uint8)
    return (cv2.dilate(support, ring) - cv2.erode(support, ring)) > 0


def estimate_overlay(crops: list[np.ndarray], clean: np.ndarray, pinned: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Measure the overlay (see _estimate_once). Frames in which the first measurement finds the logo absent would
    pull the median towards "no logo", so when there are such frames the measurement is repeated without them."""
    first = _estimate_once(crops, clean, pinned)
    alpha, colour = first[0], first[1]
    outline = _outline_of(alpha)
    present = [crop for crop in crops if _gain_of(crop.astype(np.float32), alpha, colour, outline) > 0.0]
    if MIN_REFINE_FRAMES <= len(present) < len(crops):
        try:
            return _estimate_once(present, clean, pinned)
        except OverlayRejected:
            return first
    return first


def _estimate_once(crops: list[np.ndarray], clean: np.ndarray, pinned: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Measure the overlay from ROI crops (uint8 BGR, same size) of frames that show it.

    `clean` marks ROI pixels known to be free of the logo; `pinned` is the thin band along the sides of the ROI that
    are not the picture's edge, where the integration is anchored (pinning every clean pixel instead lets small
    biases in the median gradients pile up inside the logo). Returns (opacity map, overlay colour, strongest opacity, edge
    ratio, clipped share) or raises OverlayRejected."""
    frames = np.stack(crops).astype(np.float32)
    count, height, width, _ = frames.shape
    if count < MIN_SAMPLE:
        raise OverlayRejected("The clip is too short to measure an overlay (it needs at least 16 frames showing it).")
    if height * width > MAX_ROI_PIXELS:
        raise OverlayRejected("The marked area is too large to measure an overlay in.")
    if int(clean.sum()) < 200:
        raise OverlayRejected("There isn't enough clean picture around the mark to measure against. Mark a smaller area.")

    inside = ~clean
    change = float(np.median(frames.mean(axis=-1).std(axis=0)[inside]))
    if change < MIN_PICTURE_CHANGE:
        raise OverlayRejected(
            "The picture behind the mark hardly changes over the clip, so the mark can't be told apart from it. "
            "An overlay can only be measured while what is behind it moves."
        )

    gx = _median_over_time(np.diff(frames, axis=2))
    gy = _median_over_time(np.diff(frames, axis=1))
    q = np.stack([integrate_gradients(gx[..., c].copy(), gy[..., c].copy(), pinned) for c in range(3)], axis=-1)
    median = _median_over_time(frames)
    contrast = np.abs(q).mean(axis=-1)
    # What counts as a real logo edge is set by how large the integration noise is on picture known to be clean.
    noise_ceiling = max(EVIDENCE_LEVELS, 1.2 * float(np.percentile(contrast[clean], 99.5)))
    evidence = (contrast > noise_ceiling) & ~clean
    if int(evidence.sum()) < 20:
        raise OverlayRejected("No overlay could be found in the marked area.")

    subset = frames[np.linspace(0, count - 1, min(count, SEARCH_FRAMES)).astype(int)]
    reference_region = clean & (cv2.erode(clean.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    reference = _edge_energy(subset, reference_region)
    ghost_reference = _ghost_level(subset, reference_region)
    if not np.isfinite(reference) or reference <= 0:
        raise OverlayRejected("There isn't enough clean picture around the mark to measure against. Mark a smaller area.")

    # Candidate colours. A grey (white, black, grey) overlay is one number. A tinted one is found from how the measured
    # contrast splits across the channels: Q_c = A * (W_c - picture_c), so W_c follows from one free level.
    typical = np.median(median[clean], axis=0)
    points = q[evidence]
    lead = int(np.argmax(np.abs(points).mean(axis=0)))
    usable = np.abs(points[:, lead]) > noise_ceiling
    ratios = np.median(points[usable] / points[usable][:, lead : lead + 1], axis=0) if usable.sum() >= 20 else np.ones(3)

    def candidate(family: str, level: float) -> np.ndarray:
        if family == "grey":
            return np.full(3, level, np.float32)
        return np.clip(typical + ratios * (level - typical[lead]), 0.0, 255.0).astype(np.float32)

    best: tuple[float, str, float, float, float, np.ndarray, np.ndarray] | None = None  # total, family, level, ratio, clipped, alpha, colour

    def consider(family: str, level: float) -> None:
        nonlocal best
        colour = candidate(family, level)
        scored = _score(colour, q, median, subset, reference, evidence, ghost_reference)
        if scored is not None and (best is None or scored[0] < best[0]):
            best = (scored[0], family, level, scored[1], scored[2], scored[3], colour)

    for family in ("grey", "tinted"):
        for level in range(0, 256, 8):
            consider(family, float(level))
    if best is None:
        raise OverlayRejected("No overlay could be found in the marked area.")
    _, family, centre = best[0], best[1], best[2]
    for level in np.arange(max(0.0, centre - 8), min(255.0, centre + 8) + 0.5, 2.0):
        consider(family, float(level))

    assert best is not None
    _, _, _, ratio, clipped, alpha, grey = best
    opacity = float(alpha.max())
    if opacity < MIN_OPACITY:
        raise OverlayRejected("No overlay could be found in the marked area.")
    if opacity > MAX_OPACITY:
        raise OverlayRejected("The mark is too close to opaque to take off: it hides the picture, so there is nothing to restore it from.")
    support = (alpha > 0.03).astype(np.uint8)
    outline = _outline_of(alpha)
    # Frames where the logo is faded (or absent) are inverted with their own strength, as the render will do.
    full = np.empty_like(frames)
    for n in range(count):
        gain = _gain_of(frames[n], alpha, grey, outline)
        full[n] = frames[n] if gain == 0.0 else _invert(frames[n : n + 1], np.minimum(alpha * gain, MAX_APPLIED_OPACITY), grey)[0]
    clipped = float(((full < -6.0) | (full > 261.0))[:, alpha > APPLY_THRESHOLD].mean())
    ratio = _edge_energy(full[np.linspace(0, count - 1, min(count, SEARCH_FRAMES)).astype(int)], outline) / reference
    # A logo that is still there shows up as edges that stay put while the picture changes: the median over time of
    # the inverted frames' gradients is then far from the noise level it has on clean picture.
    region = outline | (cv2.dilate(support, np.ones((9, 9), np.uint8)) > 0)
    floor = _ghost_level(full, reference_region)  # what the measure reads on picture with no logo
    after, before = _ghost_level(full, region), _ghost_level(frames, region)
    ghost = after / max(floor, 1e-6)
    # Share of the logo's own edge signal that is still there after inverting. Video compression leaves a little of
    # it (the logo is encoded as part of the picture); a wrong overlay model leaves much more.
    leftover = max(after - floor, 0.0) / max(before - floor, 1e-6)
    LAST_CHECK.update(ghost=ghost, leftover=leftover, ratio=ratio, clipped=clipped)
    if clipped > MAX_CLIPPED or ratio > MAX_EDGE_RATIO or leftover > MAX_LEFTOVER:
        raise OverlayRejected(
            "The mark doesn't behave like a single-colour, semi-transparent overlay (inverting it leaves traces of it or "
            "pushes colours out of range), so it won't be taken off this way."
        )
    return alpha, grey, opacity, ratio, clipped


# -- logos that fade, appear or disappear -----------------------------------------------------------------


def _frame_gain(crop: np.ndarray, model: OverlayModel) -> float:
    return _gain_of(crop, model.alpha, model.colour, model.outline)


def _gain_of(crop: np.ndarray, base_alpha: np.ndarray, base_colour: np.ndarray, outline: np.ndarray | None) -> float:
    """How strong the logo is in this one frame, as a multiple of the measured model (0 = absent, 1 = as measured).

    Inverting with the right strength leaves the least trace of the logo's outline; too little leaves the logo, too
    much leaves it inverted. So the strength is the one that minimises the edge energy along the outline of the
    inverted picture (brightness only). Frames close to 1 snap to exactly 1, and frames close to 0 to absent."""
    if outline is None or not outline.any():
        return 1.0
    luma = crop @ LUMA
    colour = float(base_colour @ LUMA)
    inside = outline[:-1, :-1].astype(np.float32)
    inside_count = float(inside.sum())

    def energy(gain: float) -> float:
        a = np.minimum(base_alpha * gain, MAX_APPLIED_OPACITY)
        inverted = (luma - a * colour) / (1.0 - a)
        gx = np.abs(np.diff(inverted, axis=1))[:-1, :]
        gy = np.abs(np.diff(inverted, axis=0))[:, :-1]
        return float(((gx + gy) * inside).sum() / inside_count)

    scores = [energy(g) for g in GAIN_GRID]
    best = int(np.argmin(scores))
    low, high = float(GAIN_GRID[max(best - 1, 0)]), float(GAIN_GRID[min(best + 1, len(GAIN_GRID) - 1)])
    ratio = 0.6180339887
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    fa, fb = energy(a), energy(b)
    for _ in range(7):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - ratio * (high - low)
            fa = energy(a)
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = energy(b)
    gain = 0.5 * (low + high)
    if abs(gain - 1.0) <= GAIN_SNAP_ONE:
        return 1.0
    return 0.0 if gain <= GAIN_SNAP_ZERO else gain


# -- the engine -----------------------------------------------------------------------------------------


class OverlayInversionEngine(InpaintingEngine):
    provider = "overlay"
    name = "Overlay removal (exact inversion)"
    is_development_only = False
    sends_frames_to_third_party = False

    def __init__(self, sample_frames: int | None = None, memory_mb: int = DEFAULT_MEMORY_MB) -> None:
        self._sample_frames = sample_frames
        self.memory_mb = memory_mb
        self._model: OverlayModel | None = None
        self._gains = {"full": 0, "partial": 0, "absent": 0}

    def is_available(self) -> bool:
        return True

    def device(self) -> DeviceInfo:
        return DeviceInfo(kind="cpu", name="CPU", note="Runs on the CPU and is fast once the overlay has been measured.")

    def expectation(self) -> str:
        return (
            "For a fixed, semi-transparent, single-colour logo over a picture that changes: the logo is measured from the "
            "video and taken off exactly, restoring the real pixels. If the mark isn't like that, the render stops and says why."
        )

    def notes(self) -> list[str]:
        model = self._model
        if model is None:
            return []
        spread = float(model.colour.max() - model.colour.min())
        level = float(model.colour.mean())
        colour = "coloured" if spread > 40 else "white" if level >= 215 else "black" if level <= 40 else "grey"
        notes = [
            (
                f"A fixed {colour} overlay (up to {model.opacity:.0%} opaque) was measured from the video itself and "
                "inverted: the original pixels under it were restored, not invented."
            )
        ]
        counts = self._gains
        if counts["partial"] or counts["absent"]:
            notes.append(
                f"The overlay's strength changed during the video: full strength in {counts['full']} frames, partly faded "
                f"in {counts['partial']} and not visible in {counts['absent']}. Each frame was inverted with its own "
                "estimate of the strength, so frames with a faded logo are a little less exact."
            )
        return notes

    # -- processing ---------------------------------------------------------------------------------------

    def process(self, frames: Iterable[Frame], masks: Iterable[Mask], options: InpaintingOptions) -> Iterator[Frame]:
        self.require_available()
        pairs = paired(frames, masks)
        first = next(pairs, None)
        if first is None:
            return
        limit = self.sample_size(first[1])
        sample = [first, *islice(pairs, limit - 1)]
        yield from self.run(sample, pairs, options, measure(sample, options, self.memory_mb * 1024 * 1024))

    @staticmethod
    def sample_frames_for(frame_nbytes: int) -> int:
        return int(np.clip(SAMPLE_BYTES // max(frame_nbytes, 1), MIN_SAMPLE, MAX_SAMPLE))

    def sample_size(self, frame: Frame) -> int:
        return self._sample_frames or self.sample_frames_for(frame.nbytes)

    def run(
        self,
        sample: list[tuple[int, Frame, Mask]],
        rest: Iterator[tuple[int, Frame, Mask]],
        options: InpaintingOptions,
        model: OverlayModel,
    ) -> Iterator[Frame]:
        """Apply an already measured overlay to the sampled frames and then to the rest of the stream."""
        self._model = model
        self._gains = {"full": 0, "partial": 0, "absent": 0}
        for index, frame, mask in chain(sample, rest):
            yield self._apply(model, index, frame, mask, options)

    def _apply(self, model: OverlayModel, index: int, frame: Frame, mask: Mask, options: InpaintingOptions) -> Frame:
        prepared = prepare_mask(mask, options)
        if prepared is None:
            return frame.copy()
        if not np.array_equal(mask > 127, model.mask):
            raise InpaintingError(f"The mark changes at frame {index + 1}. Overlay removal needs a mark that stays where it is.")
        window = model.roi.slices
        crop = frame[window].astype(np.float32)
        gain = _frame_gain(crop, model)
        self._gains["full" if gain == 1.0 else "absent" if gain == 0.0 else "partial"] += 1
        if gain == 0.0:
            return frame.copy()  # the logo isn't in this frame, so there is nothing to take off
        alpha = np.minimum(model.alpha * gain, MAX_APPLIED_OPACITY)
        restored = np.clip(_invert(crop[None], alpha, model.colour)[0], 0.0, 255.0)
        weight = np.clip(alpha / 0.03, 0.0, 1.0) * (prepared.alpha[window] > 0)
        output = frame.copy()
        output[window] = np.clip(np.rint(crop * (1.0 - weight[..., None]) + restored * weight[..., None]), 0, 255).astype(np.uint8)
        return output


def measure(sample: list[tuple[int, Frame, Mask]], options: InpaintingOptions, memory_bytes: int = DEFAULT_MEMORY_MB * 1024 * 1024) -> OverlayModel:
    """Measure the overlay from the first frames. Raises InpaintingError (safe to show) when it can't be trusted."""
    shown = [(f, m) for _, f, m in sample if (m > 127).any()]
    if not shown:
        raise InpaintingError("There is nothing marked in the first frames, so there is no overlay to measure.")
    reference = shown[0][1] > 127
    if any(not np.array_equal(m > 127, reference) for _, m in shown):
        raise InpaintingError("The mark moves or changes shape. Overlay removal needs a mark that stays where it is.")
    height, width = reference.shape
    rows, cols = np.flatnonzero(reference.any(axis=1)), np.flatnonzero(reference.any(axis=0))
    roi = Roi(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1).expanded(MARGIN_PX + options.reach_px, width, height)
    mask_roi = reference[roi.slices].astype(np.uint8)
    gap = CLEAN_GAP_PX + options.reach_px
    clean = cv2.dilate(mask_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))) == 0
    pinned = np.zeros(mask_roi.shape, bool)
    if roi.y0 > 0:
        pinned[:2, :] = True
    if roi.y1 < height:
        pinned[-2:, :] = True
    if roi.x0 > 0:
        pinned[:, :2] = True
    if roi.x1 < width:
        pinned[:, -2:] = True
    if not pinned.any():
        raise InpaintingError("The marked area fills the picture, so there is no clean picture to measure the overlay against.")
    # The measurement holds several float32 stacks of the marked area for every frame it studies, so the number of
    # frames follows the memory it is allowed (never fewer than MIN_SAMPLE, spread evenly over the ones offered).
    area = (roi.x1 - roi.x0) * (roi.y1 - roi.y0)
    affordable = max(MIN_SAMPLE, memory_bytes // (area * BYTES_PER_ROI_PIXEL_FRAME))
    if len(shown) > affordable:
        shown = [shown[int(i)] for i in np.linspace(0, len(shown) - 1, affordable)]
    crops = [np.ascontiguousarray(frame[roi.slices]) for frame, _ in shown]
    try:
        alpha, grey, opacity, ratio, clipped = estimate_overlay(crops, clean, pinned)
    except OverlayRejected as rejection:
        raise InpaintingError(str(rejection)) from rejection
    # Only pixels inside the marked area (grown a little) are ever changed.
    allowed = cv2.dilate(mask_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * options.reach_px + 1,) * 2)) > 0
    alpha = np.where(allowed, alpha, 0.0).astype(np.float32)
    logger.info("overlay measured", extra={"colour": [round(float(c)) for c in grey], "opacity": round(opacity, 2), "edge_ratio": round(ratio, 2)})
    outline = _outline_of(alpha)
    return OverlayModel(roi=roi, mask=reference, alpha=alpha, colour=grey, opacity=opacity, edge_ratio=ratio, clipped=clipped, outline=outline)
