"""Marks that drift across the picture.

A mark that moves can't be found the way a fixed one is (a structure that stays put while the picture changes). Where
part of the picture is still — a card, a slide, a freeze — the mark shows up there plainly: it is what differs from the
still background, frame after frame, and it moves. Following those differences from frame to frame and keeping the ones
that travel on a smooth path gives the mark, with a clean look of it.

That look is then carried on along the same path into the parts of the picture that do move, where the mark alone
could never have been found: the mark is searched for only near where its path says it should be, and a stretch where
it can't be seen is bridged only when it is seen again on the other side.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.detect.follow import Matcher
from app.detect.types import Overlay

STILL_DEVIATION = 2.5  # grey levels: a place whose picture rarely departs from its usual look this far is "still"
STILL_SHARE = 0.06  # at least this share of the frame must be still, or there is nothing to difference against
RESIDUAL = 9.0  # grey levels a pixel must differ from the still background to count as the mark
MIN_W, MAX_W = 0.06, 0.6  # mark width as a share of the frame width
MAX_H = 0.07  # ... and height as a share of the frame height (a line of text, a small logo)
LINK_DISTANCE = 9.0  # px (survey) a mark may move between consecutive samples
MAX_GAP = 3
MIN_TRAVEL = 0.05  # the path must cover this share of the frame width, or it isn't moving
MIN_SECONDS = 0.5
FOLLOW_ACCEPT = 0.3  # match needed near the predicted position (a coincidence there is unlikely)
FOLLOW_RADIUS = 4
MAX_COAST = 10  # samples a path is carried along without seeing the mark


def still_background(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(background, still mask) from the whole clip: the usual look of each place, and where it hardly ever departs from it."""
    sample = grey[:: max(1, grey.shape[0] // 100)]
    background = np.median(sample, axis=0)
    deviation = np.median(np.abs(sample - background), axis=0)
    still = (deviation < STILL_DEVIATION).astype(np.uint8)
    still = cv2.morphologyEx(still, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return background, cv2.erode(still, np.ones((3, 3), np.uint8))


def _detections(grey: np.ndarray, background: np.ndarray, still: np.ndarray) -> tuple[list[list[tuple[float, float, float, float]]], dict[tuple[int, int], tuple[float, float, float, float, float]]]:
    n, height, width = grey.shape
    join = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    out: list[list[tuple[float, float, float, float]]] = []
    rects: dict[tuple[int, int], tuple[float, float, float, float, float]] = {}  # (t, k) -> centre x, y, length, height, angle
    for t in range(n):
        flagged = ((np.abs(grey[t] - background) > RESIDUAL) & (still > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(flagged, join), connectivity=8)
        boxes = []
        for i in range(1, count):
            own = (labels == i) & (flagged > 0)
            if own.sum() < 10:
                continue
            ys, xs = np.nonzero(own)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            w, h = x1 - x0, y1 - y0
            if MIN_W * width <= w <= MAX_W * width and 5 <= h <= MAX_H * height and own.sum() >= 0.06 * w * h:
                (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
                if rw < rh:
                    rw, rh, ang = rh, rw, ang + 90.0
                ang = (ang + 90.0) % 180.0 - 90.0
                rects[(t, len(boxes))] = (float(rcx), float(rcy), float(rw), float(rh), float(ang))
                boxes.append((float(x0), float(y0), float(w), float(h)))
        out.append(boxes)
    return out, rects


def _link(detections: list[list[tuple[float, float, float, float]]], width: int) -> list[list[tuple[int, tuple[float, float, float, float], int]]]:
    tracks: list[list[tuple[int, tuple[float, float, float, float], int]]] = []
    for t, boxes in enumerate(detections):
        used: set[int] = set()
        for track in tracks:
            last_t, last = track[-1][0], track[-1][1]
            if t - last_t > MAX_GAP + 1:
                continue
            best, best_d = None, LINK_DISTANCE * (t - last_t)
            for k, b in enumerate(boxes):
                if k in used or not 0.6 <= b[2] / last[2] <= 1.6:
                    continue
                d = float(np.hypot(b[0] + b[2] / 2 - last[0] - last[2] / 2, b[1] + b[3] / 2 - last[1] - last[3] / 2))
                if d < best_d:
                    best, best_d = k, d
            if best is not None:
                track.append((t, boxes[best], best))
                used.add(best)
        for k, b in enumerate(boxes):
            if k not in used:
                tracks.append([(t, b, k)])
    return tracks


def find_drifting(grey: np.ndarray, hp: np.ndarray, fps: float, stride: int) -> list[Overlay]:
    """Overlays that travel across a still part of the picture, with their poses on the frames where they were seen."""
    n, height, width = grey.shape
    if n < 8:
        return []
    background, still = still_background(grey)
    if float(still.mean()) < STILL_SHARE:
        return []
    min_len = max(6, round(MIN_SECONDS * fps / stride))
    found: list[Overlay] = []
    detections, rects = _detections(grey, background, still)
    for track in sorted(_link(detections, width), key=len, reverse=True):
        if len(track) < min_len:
            break
        first, last = track[0][1], track[-1][1]
        travel = float(np.hypot(first[0] + first[2] / 2 - last[0] - last[2] / 2, first[1] + first[3] / 2 - last[1] - last[3] / 2))
        if travel < MIN_TRAVEL * width:
            continue
        w = int(round(np.median([b[2] for _, b, _ in track])))
        h = int(round(np.median([b[3] for _, b, _ in track])))
        overlay = Overlay(
            id=0, kind="text" if w >= 2.2 * h else "logo", template=np.zeros((h, w), np.float32), support=np.ones((h, w), np.uint8),
            score=0.0, activity=0.0, present=np.zeros(n, bool), x=np.zeros(n, np.float32), y=np.zeros(n, np.float32),
            s=np.ones(n, np.float32), match=np.zeros(n, np.float32), basis="drift",
        )  # fmt: skip
        for t, b, _ in track:
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            overlay.present[t], overlay.match[t] = True, 1.0
            overlay.x[t], overlay.y[t] = cx - w / 2, cy - h / 2
        _fill_between(overlay)
        overlay.shapes = _smooth_shapes(track, rects, n, fps, stride)
        found.append(overlay)
        if len(found) >= 3:
            break
    return found


SHAPE_MARGIN = 4.0  # px (survey) added around the measured outline: the soft edge of lettering is wider than its strokes
SMOOTH_SECONDS = 0.25
EXTRAPOLATE_SECONDS = 0.35


def _smooth_shapes(track: list, rects: dict, n: int, fps: float, stride: int) -> dict[int, np.ndarray]:
    """One rotated rectangle per sample along the path. Measured on each sighting from the pixels that differ from the
    still background (the weak end of a letter may be missed on any one frame), then smoothed over time: the size and
    tilt of a mark change slowly, so their running median is truer than any single frame."""
    seen = np.array([t for t, _, _ in track])
    data = np.array([rects[(t, k)] for t, _, k in track])  # cx, cy, L, H, angle
    win = max(5, round(SMOOTH_SECONDS * fps / stride) | 1)
    first, last = int(seen[0]), int(seen[-1])
    ts = np.arange(first, last + 1)
    cols = [np.interp(ts, seen, data[:, c]) for c in range(5)]
    out: dict[int, np.ndarray] = {}
    half = win // 2
    ext = max(1, round(EXTRAPOLATE_SECONDS * fps / stride))

    def rect(cx: float, cy: float, length: float, height: float, angle: float) -> np.ndarray:
        box = cv2.boxPoints(((cx, cy), (length + 2 * SHAPE_MARGIN, height + 2 * SHAPE_MARGIN), angle))
        order = sorted(box.tolist(), key=lambda p: p[0])
        left, right = sorted(order[:2], key=lambda p: p[1]), sorted(order[2:], key=lambda p: p[1])
        return np.array([left[0], left[1], right[1], right[0]], np.float32)  # top-left, bottom-left, bottom-right, top-right

    steady = [np.array([np.median(cols[c][max(0, i - half) : i + half + 1]) for i in range(len(ts))]) for c in (2, 3, 4)]
    centre = [np.array([np.mean(cols[c][max(0, i - half // 2) : i + half // 2 + 1]) for i in range(len(ts))]) for c in (0, 1)]
    for i, t in enumerate(ts):
        out[int(t)] = rect(centre[0][i], centre[1][i], steady[0][i], steady[1][i], steady[2][i])
    # a short stretch beyond each end, along the heading the path had there
    for direction, edge in ((-1, 0), (1, len(ts) - 1)):
        back = max(0, edge - 8) if direction == 1 else edge
        front = edge if direction == 1 else min(len(ts) - 1, edge + 8)
        span = max(1, front - back)
        v = np.array([centre[0][front] - centre[0][back], centre[1][front] - centre[1][back]]) / span
        for k in range(1, ext + 1):
            t = int(ts[edge]) + direction * k
            if not 0 <= t < n:
                break
            cx, cy = centre[0][edge] + direction * v[0] * k, centre[1][edge] + direction * v[1] * k
            out[t] = rect(cx, cy, steady[0][edge], steady[1][edge], steady[2][edge])
    return out


def _fill_between(overlay: Overlay) -> None:
    """Straight-line poses on the samples between sightings of one path."""
    idx = np.flatnonzero(overlay.present)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if b - a > 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            overlay.x[a + 1 : b] = overlay.x[a] * (1 - t) + overlay.x[b] * t
            overlay.y[a + 1 : b] = overlay.y[a] * (1 - t) + overlay.y[b] * t
            overlay.present[a + 1 : b] = True
            overlay.match[a + 1 : b] = 0.5


def learn_template(overlay: Overlay, hp: np.ndarray) -> bool:
    """The mark's look: the detail picture on the frames it was seen, cut out at its own positions and averaged."""
    h, w = overlay.template.shape
    idx = np.flatnonzero(overlay.present)
    idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), 60)).astype(int)]
    patches = []
    for i in idx:
        x, y = int(round(overlay.x[i])), int(round(overlay.y[i]))
        if x < 0 or y < 0 or x + w > hp.shape[2] or y + h > hp.shape[1]:
            continue
        patches.append(hp[i, y : y + h, x : x + w].astype(np.float32))
    if len(patches) < 3:
        return False
    stack = np.stack(patches)
    mean, mean_abs = stack.mean(axis=0), np.abs(stack).mean(axis=0)
    coherence = np.abs(mean) / (mean_abs + 1.0)
    keep = (coherence > 0.8) & (np.abs(mean) > 0.25 * float(np.abs(mean).max()))
    if keep.sum() < 24:
        return False
    overlay.template, overlay.support = mean, keep.astype(np.uint8)
    return True


def carry_along(matcher: Matcher, overlay: Overlay) -> None:
    """Follow the path beyond where the mark was seen, into parts of the picture that move, looking only near where the
    path says the mark must be. Samples where it can't be seen are bridged only if it is seen again after them."""
    n = matcher.n
    mask = cv2.dilate(overlay.support, np.ones((3, 3), np.uint8)).astype(np.float32)
    seen = np.flatnonzero(overlay.present)
    span = max(1, int(seen[-1] - seen[0]))
    forward = np.array([overlay.x[seen[-1]] - overlay.x[seen[0]], overlay.y[seen[-1]] - overlay.y[seen[0]]], np.float64) / span
    for direction in (1, -1):
        edge = int(seen[-1] if direction == 1 else seen[0])
        recent = seen[-10:] if direction == 1 else seen[:10]
        gap = max(1, int(recent[-1] - recent[0]))
        v = np.array([overlay.x[recent[-1]] - overlay.x[recent[0]], overlay.y[recent[-1]] - overlay.y[recent[0]]], np.float64) / gap
        if float(np.hypot(*v)) < 0.3 * float(np.hypot(*forward)):  # a short stretch says little about the heading: use the whole path
            v = forward.copy()
        speed = float(np.hypot(*v))
        last_i, last_pos = edge, np.array([overlay.x[edge], overlay.y[edge]], np.float64)
        pos = last_pos.copy()
        i = edge + direction
        while 0 <= i < n:
            guess = pos + direction * v
            score, nx, ny, _ = matcher.local(i, overlay.template, float(guess[0]), float(guess[1]), 1.0, radius=FOLLOW_RADIUS, mask=mask)
            found = np.array([nx, ny], np.float64)
            moved = float(np.hypot(*(found - last_pos))) / abs(i - last_i)
            if score >= FOLLOW_ACCEPT and 0.5 * speed <= moved <= 2.0 * speed:  # a mark keeps moving; something that doesn't is the picture
                for j in range(min(last_i, i) + 1, max(last_i, i)):  # bridge what couldn't be seen
                    t = (j - last_i) / (i - last_i)
                    overlay.present[j], overlay.match[j] = True, FOLLOW_ACCEPT
                    overlay.x[j], overlay.y[j] = (last_pos * (1 - t) + found * t)
                overlay.present[i], overlay.match[i] = True, score
                overlay.x[i], overlay.y[i] = nx, ny
                v = 0.9 * v + 0.1 * (found - last_pos) * direction / abs(i - last_i)
                last_i, last_pos, pos = i, found, found
            else:
                pos = guess
                if abs(i - last_i) > MAX_COAST:
                    break
            i += direction
