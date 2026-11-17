"""Does a template look like lettering or a logo, rather than a soft blob, a single edge or a bar?

A watermark's own pixels are strokes: several separate ones, running in more than one direction. Scene structure that
happens to stay put (a highlight on teeth, the edge of a door frame, a horizon line) is smooth, or has one dominant
direction. These two numbers keep that difference, so a mark that cannot be matched perfectly on every frame of real,
compressed footage can still be told apart from scene structure.
"""

from __future__ import annotations

import cv2
import numpy as np

STROKE_LEVEL = 0.35  # a stroke pixel is this share of the template's peak
MIN_STROKE_AREA = 4  # px (survey)
MIN_STROKES = 6
MAX_ORIENTATION = 0.7  # 1 = every edge runs the same way (a line or bar), 0 = edges in every direction (lettering)


def strokes(template: np.ndarray) -> int:
    """How many separate strokes the template holds."""
    peak = float(np.abs(template).max())
    if peak <= 0:
        return 0
    mask = (np.abs(template) > STROKE_LEVEL * peak).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_STROKE_AREA))


def orientation_coherence(template: np.ndarray) -> float:
    """Dominance of one edge direction in the template, 0..1 (structure-tensor coherence)."""
    peak = float(np.abs(template).max())
    if peak <= 0:
        return 1.0
    unit = (template / peak).astype(np.float32)
    gx = cv2.Sobel(unit, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(unit, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    return float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-6))


def lettering_like(template: np.ndarray) -> bool:
    return strokes(template) >= MIN_STROKES and orientation_coherence(template) <= MAX_ORIENTATION
