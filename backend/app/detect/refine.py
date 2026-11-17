"""After a mark is accepted: make its mask cover all of it, and hold it steady.

* `grow_extent` — the template is cut from the strongest part of the mark. The weak end of a transparent mark (a letter
  or two over busy footage) can be left out of it, and a mask that stops short leaves that end on screen. The mark's
  frames, cut out at the measured positions, are stacked; wherever the stack still shows structure that stays the same
  from frame to frame next to the template, the template box grows to take it in.
* `settle_static` — a mark that never moves is measured with a pixel or so of noise on each frame. Its position is set
  to the median, so the mask doesn't wobble, and short dropouts of the match (a frame or two where compression hid it)
  are filled, so the mask doesn't blink. A gap long enough to be the mark really leaving is kept.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.detect.follow import Matcher
from app.detect.types import Overlay

GROW_PAD = 0.35  # look this far (share of the box side, at most MAX_PAD px) beyond the template
MAX_PAD = 28
GROW_COHERENCE = 0.6  # grown pixels must keep the same sign and strength on most frames (noise averages to ~0.1)
DIM_SHARE = 0.5  # a frame where the mark's surroundings carry less than this share of the usual detail says nothing about it
DIM_MATCH = 0.25  # ... and where there is detail, a match at least this good keeps the mark present
GROW_LEVEL = 0.10  # structure counts when its evidence is this share of the template's own (90th percentile)
GROW_FLOOR = 1.5  # ... and never less than this (grey levels), so noise isn't mistaken for structure
MAX_GROWTH_TEXT = (0.4, 0.15)  # a line of text may grow this share of its width along the line, and of its height across it
MAX_GROWTH_LOGO = (0.12, 0.12)  # a compact mark's own outline is where it ends: it may only grow a little
STATIC_WANDER = 4.0  # px (survey): the measured position stays this close to the median for a mark that doesn't move
MAX_BRIDGE_SECONDS = 0.3


def _stack(overlay: Overlay, hp: np.ndarray, pad_x: int, pad_y: int, limit: int = 60) -> np.ndarray | None:
    h, w = overlay.template.shape
    idx = np.flatnonzero(overlay.present)
    if len(idx) < 3:
        return None
    idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), limit)).astype(int)]
    patches = []
    for i in idx:
        s = float(overlay.s[i])
        pw, ph = max(2, round((w + 2 * pad_x) * s)), max(2, round((h + 2 * pad_y) * s))
        x0, y0 = int(round(overlay.x[i] - pad_x * s)), int(round(overlay.y[i] - pad_y * s))
        canvas = np.zeros((ph, pw), np.float32)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(hp.shape[2], x0 + pw), min(hp.shape[1], y0 + ph)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = hp[i, sy0:sy1, sx0:sx1].astype(np.float32)
        patches.append(cv2.resize(canvas, (w + 2 * pad_x, h + 2 * pad_y), interpolation=cv2.INTER_AREA) if (pw, ph) != (w + 2 * pad_x, h + 2 * pad_y) else canvas)
    return np.stack(patches) if len(patches) >= 3 else None


def grow_extent(overlay: Overlay, hp: np.ndarray) -> bool:
    """Widen the overlay's box (and template) to the whole mark. True when it changed."""
    h, w = overlay.template.shape
    pad_x, pad_y = min(MAX_PAD, int(GROW_PAD * w)), min(MAX_PAD, int(GROW_PAD * h))
    stack = _stack(overlay, hp, pad_x, pad_y)
    if stack is None:
        return False
    mean, mean_abs = stack.mean(axis=0), np.abs(stack).mean(axis=0)
    evidence = mean * mean / (mean_abs + 1.0)
    core = evidence[pad_y : pad_y + h, pad_x : pad_x + w]
    level = max(GROW_FLOOR, GROW_LEVEL * float(np.percentile(core, 90)))
    coherence = np.abs(mean) / (mean_abs + 1.0)
    mask = ((evidence > level) & (coherence > GROW_COHERENCE)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    inside = np.unique(labels[pad_y : pad_y + h, pad_x : pad_x + w])
    keep = [int(i) for i in inside if i != 0 and stats[i, cv2.CC_STAT_AREA] >= 6]
    if not keep:
        return False
    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in keep)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in keep)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in keep)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in keep)
    gx, gy = MAX_GROWTH_TEXT if w >= 2.5 * h else MAX_GROWTH_LOGO
    x0, y0 = max(x0, pad_x - int(gx * w)), max(y0, pad_y - int(gy * h))
    x1, y1 = min(x1, pad_x + w + int(gx * w)), min(y1, pad_y + h + int(gy * h))
    x0, y0, x1, y1 = min(x0, pad_x), min(y0, pad_y), max(x1, pad_x + w), max(y1, pad_y + h)  # never smaller than before
    if (x0, y0, x1, y1) == (pad_x, pad_y, pad_x + w, pad_y + h):
        return False
    grown = np.zeros_like(mask)
    for i in keep:
        grown[labels == i] = 1
    overlay.template = mean[y0:y1, x0:x1].copy()
    support = cv2.dilate(grown[y0:y1, x0:x1], np.ones((3, 3), np.uint8))
    support[pad_y - y0 : pad_y - y0 + h, pad_x - x0 : pad_x - x0 + w] |= overlay.support  # what was matched stays
    overlay.support = support.astype(np.uint8)
    on = overlay.present
    overlay.x[on] = overlay.x[on] + (x0 - pad_x) * overlay.s[on]
    overlay.y[on] = overlay.y[on] + (y0 - pad_y) * overlay.s[on]
    overlay.notes.append("extent grown to the whole mark")
    return True


def settle_static(overlay: Overlay, fps: float, stride: int) -> bool:
    """Hold a mark that doesn't move at one position, and fill short dropouts. True when it was static."""
    idx = np.flatnonzero(overlay.present)
    if len(idx) < 3:
        return False
    mx, my, ms = float(np.median(overlay.x[idx])), float(np.median(overlay.y[idx])), float(np.median(overlay.s[idx]))
    wander = max(float(np.abs(overlay.x[idx] - mx).max()), float(np.abs(overlay.y[idx] - my).max()))
    if wander > STATIC_WANDER or float(np.abs(overlay.s[idx] - ms).max()) > 0.12:
        return False
    max_gap = max(2, round(MAX_BRIDGE_SECONDS * fps / stride))
    present = overlay.present
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if 1 < b - a <= max_gap + 1:
            present[a + 1 : b] = True
            overlay.match[a + 1 : b] = min(overlay.match[a], overlay.match[b])
    on = np.flatnonzero(present)
    overlay.x[on], overlay.y[on], overlay.s[on] = mx, my, ms
    overlay.notes.append("holds still")
    return True


def fill_dim_frames(matcher: Matcher, overlay: Overlay) -> int:
    """A mark that holds still and lasts most of the clip is on every frame, unless a frame shows it is not.

    A frame can't show that when the picture there is washed out, black or blurred: the mark's own detail is gone with
    everything else's. So a frame is left without the mark only when the place carries the usual amount of detail and
    still doesn't match. Returns how many frames were added.
    """
    idx = np.flatnonzero(overlay.present)
    if len(idx) < 3:
        return 0
    x, y, s = float(overlay.x[idx[0]]), float(overlay.y[idx[0]]), float(overlay.s[idx[0]])
    h, w = overlay.template.shape
    mask = cv2.dilate(overlay.support, np.ones((3, 3), np.uint8)).astype(np.float32)
    inside = overlay.support.astype(bool)
    x0, y0 = int(round(x)), int(round(y))
    if x0 < 0 or y0 < 0 or x0 + w > matcher.w or y0 + h > matcher.h or not inside.any():
        return 0
    energy = np.array([float(np.abs(matcher.hp[i, y0 : y0 + h, x0 : x0 + w].astype(np.float32))[inside].mean()) for i in range(matcher.n)])
    usual = float(np.median(energy[idx]))
    # A frame that is washed out loses detail everywhere, not only under the mark. A frame where the mark is simply gone
    # loses it under the mark alone, so the band around the mark is what tells the two apart.
    pad = max(4, max(h, w) // 2)
    bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
    bx1, by1 = min(matcher.w, x0 + w + pad), min(matcher.h, y0 + h + pad)
    band = np.ones((by1 - by0, bx1 - bx0), bool)
    band[y0 - by0 : y0 - by0 + h, x0 - bx0 : x0 - bx0 + w] = False
    if not band.any():
        return 0
    around = np.array([float(np.abs(matcher.hp[i, by0:by1, bx0:bx1].astype(np.float32))[band].mean()) for i in range(matcher.n)])
    around_usual = float(np.median(around[idx]))
    added = 0
    for i in range(matcher.n):
        if overlay.present[i]:
            continue
        score = matcher.local(i, overlay.template, x, y, s, radius=2, mask=mask)[0]
        washed_out = energy[i] < DIM_SHARE * usual and around[i] < DIM_SHARE * around_usual
        if score >= DIM_MATCH or washed_out:
            overlay.present[i] = True
            overlay.match[i] = max(score, 0.0)
            overlay.x[i], overlay.y[i], overlay.s[i] = x, y, s
            added += 1
    if added:
        overlay.notes.append("kept on frames where the picture was too washed out to show it")
    return added
