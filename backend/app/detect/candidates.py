"""From an evidence map to boxes: the places where something looks stuck on the picture."""

from __future__ import annotations

import cv2
import numpy as np

from app.detect.evidence import (
    ACTIVITY_MIN,
    EVIDENCE_FLOOR,
    near_activity,
    window_evidence,
)
from app.detect.types import Box, Candidate

MIN_SIDE = 6
MAX_AREA = 0.35  # an "overlay" covering a third of the frame is the picture, not something on it
MIN_FILL = 0.12
PAD = 3


def _texty(strokes: np.ndarray, box: Box) -> bool:
    """Text is a row of small separate marks of similar height."""
    x, y, w, h = box
    if w < 2.2 * h:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(strokes[y : y + h, x : x + w], connectivity=8)
    heights = [int(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if len(heights) < 3:
        return False
    return float(np.median(heights)) >= 0.45 * h


def find_candidates(hp: np.ndarray, grey: np.ndarray, window: tuple[int, int], appearance: bool = True) -> list[Candidate]:
    """Overlay-like structures in samples [a, b). `hp`/`grey` are the whole clip's stacks."""
    a, b = window
    evidence, activity, mean_detail, detail = window_evidence(hp[a:b], grey[a:b])
    height, width = evidence.shape
    gate = near_activity(activity) >= ACTIVITY_MIN
    gated = np.where(gate, evidence, 0.0)
    if gated.max() < EVIDENCE_FLOOR:
        return []
    floor = max(EVIDENCE_FLOOR, 0.12 * float(np.percentile(gated[gated > 0], 99.5)))
    strokes = (gated > floor).astype(np.uint8)
    joined = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 7)))  # letters join into words, words into a line
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    found: list[Candidate] = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if w < MIN_SIDE or h < MIN_SIDE or w * h > MAX_AREA * width * height or area < MIN_FILL * w * h:
            continue
        inside = labels == i
        own = strokes.astype(bool) & inside
        if own.sum() < 12:
            continue
        ring = cv2.dilate(inside.astype(np.uint8), np.ones((17, 17), np.uint8)) - inside.astype(np.uint8)
        # Tighten to the strong core, so a logo that sits beside a scene edge doesn't take the edge into its template.
        core = own & (evidence > 0.4 * float(np.percentile(evidence[own], 95)))
        ys, xs = np.nonzero(core)
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        x0, y0 = max(0, x - PAD), max(0, y - PAD)
        x1, y1 = min(width, x + w + PAD), min(height, y + h + PAD)
        box: Box = (x0, y0, x1 - x0, y1 - y0)
        around = float(np.percentile(activity[ring > 0], 80)) if ring.any() else 0.0  # part of the ring may be a still border
        if around < ACTIVITY_MIN:
            continue
        template = mean_detail[y0:y1, x0:x1].copy()
        peak = float(np.abs(template).max())
        if peak < 1.0:
            continue
        support = (np.abs(template) > 0.2 * peak).astype(np.uint8)
        found.append(
            Candidate(
                box=box, score=float(evidence[own].mean()), activity=around, window=window,
                coherence=float(np.median((np.abs(mean_detail) / (detail + 1.0))[own])),
                template=template, support=support, texty=_texty(strokes, box),
            )
        )  # fmt: skip
    if appearance:
        found += _by_appearance(evidence, activity, mean_detail, detail, window)
    found.sort(key=lambda c: c.score * min(c.activity, 20.0 if c.basis == "motion" else 1.0), reverse=True)
    return found


MARGIN = 0.22  # a mark sits within this share of the frame from an edge
APPEARANCE_AREA = 0.03
ISOLATION = 2.5  # detail inside the box must be this many times the detail just around it
CRISP = 12.0  # the sharpest 5 % of the mark's detail must reach this (grey levels)


def _by_appearance(
    evidence: np.ndarray, activity: np.ndarray, mean_detail: np.ndarray, detail: np.ndarray, window: tuple[int, int]
) -> list[Candidate]:
    """Marks on a part of the picture that doesn't move, where there is no motion to be evidence.

    Nothing in the pictures separates a logo from the wall behind it there, so the call rests on how marks look: small,
    near an edge, crisp, and alone (plain picture all around them). These are reported as such and ranked below
    candidates that moving pictures proved.
    """
    height, width = evidence.shape
    still = near_activity(activity) < ACTIVITY_MIN
    ungated = np.where(still, evidence, 0.0)
    if ungated.max() < 2 * EVIDENCE_FLOOR:
        return []
    strokes = (ungated > max(EVIDENCE_FLOOR, 0.15 * float(ungated.max()))).astype(np.uint8)
    joined = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 7)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    found: list[Candidate] = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if w < 10 or h < 10 or w * h > APPEARANCE_AREA * width * height or area < 0.3 * w * h:
            continue
        cx, cy = x + w / 2, y + h / 2
        if min(cx, width - cx) > MARGIN * width and min(cy, height - cy) > MARGIN * height:
            continue
        reach = int(0.6 * max(w, h)) + 4
        x0, y0, x1, y1 = max(0, x - reach), max(0, y - reach), min(width, x + w + reach), min(height, y + h + reach)
        inside = np.zeros((height, width), bool)
        inside[y : y + h, x : x + w] = True
        around = np.zeros((height, width), bool)
        around[y0:y1, x0:x1] = True
        around &= ~inside
        if not around.any() or float(detail[inside].mean()) < ISOLATION * float(detail[around].mean() + 0.5):
            continue
        if float(np.percentile(np.abs(mean_detail[y : y + h, x : x + w]), 95)) < CRISP:
            continue
        template = mean_detail[y : y + h, x : x + w].copy()
        peak = float(np.abs(template).max())
        found.append(
            Candidate(
                box=(x, y, w, h), score=float(evidence[inside].mean()), activity=0.0, window=window, template=template,
                support=(np.abs(template) > 0.2 * peak).astype(np.uint8), texty=_texty(strokes, (x, y, w, h)), basis="appearance",
            )
        )  # fmt: skip
    return found
