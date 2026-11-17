"""Find one overlay on every frame: where it is, how big, and whether it is there at all.

The overlay's template is matched (normalised cross-correlation on the detail picture, so a transparent logo over any
background still correlates) around where it was last seen, at a few nearby sizes. When it isn't found there, the
whole frame is searched at many sizes: that is how a mark that jumps to another corner, grows, or comes back after
being gone is picked up again. A frame where nothing correlates well enough is a frame where the overlay is absent:
never a guess.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from app.detect.types import Overlay

ACCEPT = 0.5  # normalised correlation needed to call the overlay present where it was just seen
ACCEPT_ELSEWHERE = 0.6  # ... and anywhere else on the frame, where a coincidence is far likelier
GLOBAL_SCALES = (0.7, 0.85, 1.0, 1.2, 1.45)
LOCAL_SCALES = (1.0, 0.93, 1.075)  # the current size first: a change of size has to earn its place
SCALE_RANGE = (0.5, 2.0)  # an overlay that changes size does so within this range of what it first looked like
SCALE_GAIN = 0.03  # a different size must beat the current one by this much to be believed
GLOBAL_EVERY = 3  # while absent, search the whole frame on every n-th frame only
MIN_TEMPLATE = 14  # px, after scaling, on the shorter side: smaller templates match anything


class Matcher:
    """Match templates against a clip's detail pictures."""

    def __init__(self, hp: np.ndarray) -> None:
        self.hp = hp  # N x H x W float16
        self.n, self.h, self.w = hp.shape
        self._half: dict[int, np.ndarray] = {}

    def half(self, i: int) -> np.ndarray:
        cached = self._half.get(i)
        if cached is None:
            cached = cv2.resize(self.hp[i].astype(np.float32), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if len(self._half) > 64:
                self._half.pop(next(iter(self._half)))
            self._half[i] = cached
        return cached

    @staticmethod
    def _scaled(template: np.ndarray, s: float) -> np.ndarray | None:
        h, w = template.shape
        nh, nw = round(h * s), round(w * s)
        if min(nh, nw) < MIN_TEMPLATE:
            return None
        if (nh, nw) == (h, w):
            return template
        return cv2.resize(template, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    @staticmethod
    def _best(image: np.ndarray, template: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int, int]:
        """Best normalised correlation; with a mask only the overlay's own pixels count, not what lies beside them."""
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            return -1.0, 0, 0
        if mask is not None and mask.shape != template.shape:
            mask = cv2.resize(mask, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask) if mask is not None else cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, best, _, at = cv2.minMaxLoc(result)
        return float(best), int(at[0]), int(at[1])

    def local(
        self, i: int, template: np.ndarray, x: float, y: float, s: float, radius: int, mask: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        """Best (score, x, y, s) within `radius` px of (x, y) at sizes around s."""
        frame = self.hp[i]
        best = (-1.0, x, y, s)
        for factor in LOCAL_SCALES:
            if not SCALE_RANGE[0] <= s * factor <= SCALE_RANGE[1]:
                continue
            scaled = self._scaled(template, s * factor)
            if scaled is None:
                continue
            th, tw = scaled.shape
            x0, y0 = max(0, int(x) - radius), max(0, int(y) - radius)
            x1, y1 = min(self.w, int(x) + tw + radius), min(self.h, int(y) + th + radius)
            if x1 - x0 < tw or y1 - y0 < th:
                continue
            score, dx, dy = self._best(frame[y0:y1, x0:x1].astype(np.float32), scaled, mask)
            if score > best[0] + (SCALE_GAIN if factor != 1.0 and best[0] > 0 else 0.0):
                best = (score, float(x0 + dx), float(y0 + dy), s * factor)
        return best

    def everywhere(self, i: int, template: np.ndarray, scales: tuple[float, ...] = GLOBAL_SCALES) -> list[tuple[float, float, float, float]]:
        """Best (score, x, y, s) for each size over the whole frame, found at half resolution and refined at full."""
        frame = self.half(i)
        hits = []
        for s in scales:
            if not SCALE_RANGE[0] <= s <= SCALE_RANGE[1]:
                continue
            scaled = self._scaled(template, s * 0.5)
            if scaled is None:
                continue
            score, dx, dy = self._best(frame, scaled)
            if score > 0:
                hits.append((score, dx * 2.0, dy * 2.0, s))
        return hits

    def refine(
        self, i: int, template: np.ndarray, hit: tuple[float, float, float, float], mask: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        score, x, y, s = hit
        return self.local(i, template, x, y, s, radius=4, mask=mask)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)), max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def follow(
    matcher: Matcher, overlay: Overlay, seed: int, pose: tuple[float, float, float], others: Sequence[Overlay] = ()
) -> None:
    """Fill `overlay`'s per-sample position, size and presence by walking out from the frame where it was found."""
    n = matcher.n
    template = overlay.template
    overlay.present[:] = False
    overlay.match[:] = 0.0
    h0, w0 = template.shape
    mask = cv2.dilate(overlay.support, np.ones((3, 3), np.uint8)).astype(np.float32)

    def taken(i: int, hit: tuple[float, float, float, float]) -> bool:
        """Is this hit the place another overlay already holds on this frame?"""
        box = (hit[1], hit[2], w0 * hit[3], h0 * hit[3])
        return any(o.present[i] and _iou(box, o.box_at(i)) > 0.3 for o in others)

    for direction in (1, -1):
        x, y, s = pose
        ghost: tuple[float, float, float] | None = None
        missing = 0
        i = seed if direction == 1 else seed - 1
        while 0 <= i < n:
            found = None
            if ghost is None:
                score, nx, ny, ns = matcher.local(i, template, x, y, s, radius=max(10, int(0.05 * matcher.w)), mask=mask)
                if score >= ACCEPT and not taken(i, (score, nx, ny, ns)):
                    found = (score, nx, ny, ns)
            else:
                gx, gy, gs = ghost
                score, nx, ny, ns = matcher.local(i, template, gx, gy, gs, radius=6, mask=mask)
                if score >= ACCEPT and not taken(i, (score, nx, ny, ns)):
                    found = (score, nx, ny, ns)
                elif missing % GLOBAL_EVERY == 0:  # an absent overlay is looked for everywhere, now and then
                    hits = matcher.everywhere(i, template)
                    hits = [hit for hit in hits if not taken(i, hit)]
                    if hits:
                        best = max(hits)
                        if best[0] >= ACCEPT_ELSEWHERE - 0.05:
                            refined = matcher.refine(i, template, best, mask)
                            if refined[0] >= ACCEPT_ELSEWHERE and not taken(i, refined):
                                found = refined
            if found is None and ghost is None:  # lost where it was: look everywhere before calling it gone
                hits = [hit for hit in matcher.everywhere(i, template) if not taken(i, hit)]
                if hits:
                    best = max(hits)
                    refined = matcher.refine(i, template, best, mask)
                    if refined[0] >= ACCEPT_ELSEWHERE and not taken(i, refined):
                        found = refined
            if found is not None:
                score, x, y, s = found
                overlay.present[i], overlay.match[i] = True, score
                overlay.x[i], overlay.y[i], overlay.s[i] = x, y, s
                if ghost is not None and missing > 1:  # skipped frames just before the hit may already show it again
                    back = i - direction
                    score_b, xb, yb, sb = matcher.local(back, template, x, y, s, radius=6, mask=mask)
                    if score_b >= ACCEPT and not overlay.present[back]:
                        overlay.present[back], overlay.match[back] = True, score_b
                        overlay.x[back], overlay.y[back], overlay.s[back] = xb, yb, sb
                ghost, missing = None, 0
            else:
                if ghost is None:
                    ghost = (x, y, s)
                missing += 1
            i += direction
    _tidy(overlay)


def _tidy(overlay: Overlay, min_run: int = 3, max_gap: int = 2) -> None:
    """One missed frame inside a long presence is noise, and a presence of one or two frames is a coincidence."""
    present = overlay.present
    n = len(present)
    i = 0
    while i < n:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n and not present[j]:
            j += 1
        near = 0 < i and j < n and abs(overlay.x[i - 1] - overlay.x[j]) + abs(overlay.y[i - 1] - overlay.y[j]) <= 8
        if near and j - i <= max_gap:  # fill a short gap by walking between the neighbours' poses; a jump is not a gap
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                overlay.present[k] = True
                overlay.x[k] = overlay.x[i - 1] * (1 - t) + overlay.x[j] * t
                overlay.y[k] = overlay.y[i - 1] * (1 - t) + overlay.y[j] * t
                overlay.s[k] = overlay.s[i - 1] * (1 - t) + overlay.s[j] * t
                overlay.match[k] = min(overlay.match[i - 1], overlay.match[j])
        i = j
    i = 0
    while i < n:
        if not present[i]:
            i += 1
            continue
        j = i
        while j < n and present[j]:
            j += 1
        if j - i < min_run and n >= 2 * min_run:
            overlay.present[i:j] = False
            overlay.match[i:j] = 0.0
        i = j
