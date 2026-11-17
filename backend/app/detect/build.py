"""Per-frame findings -> the mask document the renderer already understands.

Each overlay becomes one polygon region whose shape is the overlay's own outline (from the template), moved and scaled
by its measured pose. A region that comes and goes gets `hidden` keyframes, so the mask is empty on frames where the
overlay isn't there. Keyframes sit where the motion actually changes (a straight run needs two), and a jump from one
place to another is kept as a jump instead of being smeared across the frames between.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.detect.survey import Survey
from app.detect.types import Overlay
from app.schemas.mask import (
    MAX_KEYFRAMES_PER_REGION,
    MAX_REGIONS,
    Keyframe,
    MaskDocument,
    Region,
)

GROW = 3  # px (survey) added around the outline so the soft edge of a logo is inside the mask
MAX_POINTS = 40
TOLERANCE = 1.2  # px: how far a straight-line pose may stray from the measured one before it gets a keyframe


def outline(overlay: Overlay) -> np.ndarray:
    """The overlay's outline in template pixels: the shape of its structure (letters joined into one body), grown a little,
    with at most MAX_POINTS corners. A compact logo gets the hull of its structure; lettering keeps its own outline, so
    the mask doesn't take in the picture between and above the letters."""
    support = overlay.support
    h, w = support.shape
    pad = GROW + 2
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    canvas[pad : pad + h, pad : pad + w] = support
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1))
    grown = cv2.dilate(canvas, kernel)
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    hull = cv2.convexHull(np.vstack(contours))
    bx, by, bw, bh = cv2.boundingRect(hull)
    if cv2.contourArea(hull) < 0.8 * bw * bh:  # a hull of a few arcs of a round logo leaves its corners out: take the whole box
        hull = np.array([[[bx, by]], [[bx + bw, by]], [[bx + bw, by + bh]], [[bx, by + bh]]], np.int32)
    shape = hull
    if len(contours) > 1 or cv2.contourArea(hull) > 0:
        joined = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * GROW + 5, 3)))
        parts, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(parts) == 1 and cv2.contourArea(parts[0]) >= 0.35 * cv2.contourArea(hull):
            body = parts[0]
            if cv2.contourArea(body) < 0.92 * cv2.contourArea(hull):  # a real concavity: keep it
                shape = body
    epsilon = 0.6
    approx = cv2.approxPolyDP(shape, epsilon, True)
    while len(approx) > MAX_POINTS:
        epsilon *= 1.3
        approx = cv2.approxPolyDP(shape, epsilon, True)
    if len(approx) < 4:  # a hull that collapsed to a sliver is no use as a mask
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return approx.reshape(-1, 2).astype(np.float32) - pad


def _simplify(poses: np.ndarray, tolerance: float) -> list[int]:
    """Indices of the poses worth a keyframe (Ramer-Douglas-Peucker over (x, y, size) per sample)."""
    n = len(poses)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = (np.arange(a + 1, b) - a) / (b - a)
        line = poses[a][None, :] * (1 - t)[:, None] + poses[b][None, :] * t[:, None]
        error = np.abs(poses[a + 1 : b] - line).max(axis=1)
        k = int(error.argmax())
        if error[k] > tolerance:
            keep.add(a + 1 + k)
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return sorted(keep)


def runs_of(overlay: Overlay) -> list[tuple[int, int]]:
    """Samples [a, b] where the overlay is present without a jump or gap."""
    present = overlay.present
    h, w = overlay.template.shape
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(len(present)):
        if present[i] and start is None:
            start = i
        elif start is not None:
            ended = not present[i]
            jumped = (not ended) and math.hypot(overlay.x[i] - overlay.x[i - 1], overlay.y[i] - overlay.y[i - 1]) > max(10.0, 0.6 * min(w, h) * float(overlay.s[i]))
            if ended or jumped:
                runs.append((start, i - 1))
                start = None if ended else i
    if start is not None:
        runs.append((start, len(present) - 1))
    return runs


def region_for(overlay: Overlay, survey: Survey, region_id: str, name: str) -> Region:
    """One overlay as a region, with keyframes in source frame numbers."""
    width, height = survey.width, survey.height
    if overlay.shapes:
        return _region_from_shapes(overlay, survey, region_id, name)
    shape = outline(overlay)
    stride = survey.stride
    frames = survey.frames
    last_frame = survey.source_frames - 1

    def points_at(i: int) -> list[list[float]]:
        s = float(overlay.s[i])
        pts = shape * s + np.array([overlay.x[i], overlay.y[i]], np.float32)
        return [[float(np.clip(px / width, 0, 1)), float(np.clip(py / height, 0, 1))] for px, py in pts]

    tolerance = TOLERANCE
    while True:
        keys: list[Keyframe] = []
        runs = runs_of(overlay)
        for a, b in runs:
            poses = np.stack([overlay.x[a : b + 1], overlay.y[a : b + 1], overlay.s[a : b + 1] * max(overlay.template.shape)], axis=1).astype(np.float64)
            chosen = _simplify(poses, tolerance)
            first_frame = max(0, int(frames[a]) - (stride - 1))  # across a gap between samples, err on the side of covering
            for k, local in enumerate(chosen):
                i = a + local
                frame = first_frame if k == 0 else int(frames[i])
                keys.append(Keyframe(frame=frame, points=points_at(i)))
            end_frame = int(frames[b]) + stride  # from here on it isn't there, until the next run says otherwise
            if end_frame <= last_frame:
                keys.append(Keyframe(frame=end_frame, points=points_at(b), hidden=True))
        if runs and keys and keys[0].frame > 0:
            keys.insert(0, Keyframe(frame=0, points=keys[0].points, hidden=True))
        keys = _ordered(keys)
        if len(keys) <= MAX_KEYFRAMES_PER_REGION or tolerance > 40:
            break
        tolerance *= 1.6
    return Region(id=region_id, name=name, type="polygon", keyframes=keys[:MAX_KEYFRAMES_PER_REGION])


def _region_from_shapes(overlay: Overlay, survey: Survey, region_id: str, name: str) -> Region:
    """A mark whose shape was measured on each sample: a keyframe per sample (thinned only when there are too many)."""
    assert overlay.shapes is not None
    width, height = survey.width, survey.height
    samples = sorted(overlay.shapes)
    stride = survey.stride
    frames = survey.frames
    step = 1
    while -(-len(samples) // step) > MAX_KEYFRAMES_PER_REGION - 3:
        step += 1
    chosen = samples[::step]
    if chosen[-1] != samples[-1]:
        chosen.append(samples[-1])
    keys: list[Keyframe] = []
    for k, i in enumerate(chosen):
        pts = [[float(np.clip(px / width, 0, 1)), float(np.clip(py / height, 0, 1))] for px, py in overlay.shapes[i]]
        frame = max(0, int(frames[i]) - (stride - 1)) if k == 0 else int(frames[i])
        keys.append(Keyframe(frame=frame, points=pts))
    end_frame = int(frames[chosen[-1]]) + stride
    if end_frame <= survey.source_frames - 1:
        keys.append(Keyframe(frame=end_frame, points=keys[-1].points, hidden=True))
    if keys[0].frame > 0:
        keys.insert(0, Keyframe(frame=0, points=keys[0].points, hidden=True))
    return Region(id=region_id, name=name, type="polygon", keyframes=_ordered(keys))


def _ordered(keys: list[Keyframe]) -> list[Keyframe]:
    """Strictly increasing frames; where two keyframes land on one frame the visible one wins."""
    by_frame: dict[int, Keyframe] = {}
    for key in keys:
        old = by_frame.get(key.frame)
        if old is None or (old.hidden and not key.hidden):
            by_frame[key.frame] = key
    return [by_frame[f] for f in sorted(by_frame)]


def document_for(overlays: list[Overlay], survey: Survey) -> MaskDocument:
    regions = [
        region_for(o, survey, f"auto-{o.id}", f"{'Text' if o.kind == 'text' else 'Logo'} {i}")
        for i, o in enumerate(overlays[:MAX_REGIONS], start=1)
    ]
    return MaskDocument(version=1, regions=regions)
