"""Where does a structure stay put while the picture under it changes?

The picture is split into a smooth part and a detail part (the "high-pass"). An overlay is detail that keeps the same
sign and strength on every frame of a window; a moving scene's detail does not. Coherence = |mean| / mean|.|, which is
close to 1 for a fixed structure whatever its colour or transparency and close to 0 for changing detail. A logo that is
only 30 % opaque still leaves a coherent trace, which a plain "is this pixel constant" test would lose in the noise.

A fixed structure is only evidence of an overlay if the scene around it moves: a still picture cannot tell a logo from
a poster on the wall. So every score is gated by how much the picture changes nearby.
"""

from __future__ import annotations

import cv2
import numpy as np

HP_SIGMA = 2.5  # detail finer than this is "structure"; coarser is lighting and shape
ACTIVITY_MIN = 2.0  # grey levels of change per frame near the structure, below which the scene is "still"
EVIDENCE_FLOOR = 8.0  # weakest structure worth looking at (grey levels)
MAX_FRAMES = 24


def high_pass(grey: np.ndarray) -> np.ndarray:
    """N x H x W float32 -> the detail part of every frame, as float16 to keep a long clip's copy small."""
    out = np.empty(grey.shape, np.float16)
    for i in range(grey.shape[0]):
        out[i] = grey[i] - cv2.GaussianBlur(grey[i], (0, 0), HP_SIGMA)
    return out


def picture_activity(grey: np.ndarray) -> np.ndarray:
    """H x W: how much each place changes from one frame to the next, averaged over the clip and smoothed."""
    if grey.shape[0] < 2:
        return np.zeros(grey.shape[1:], np.float32)
    starts = np.unique(np.linspace(0, grey.shape[0] - 2, min(grey.shape[0] - 1, MAX_FRAMES)).astype(int))
    diff = np.zeros(grey.shape[1:], np.float32)
    for i in starts:  # consecutive pairs, spread over the window
        diff += np.abs(grey[i + 1] - grey[i])
    return cv2.GaussianBlur(diff / len(starts), (0, 0), 4)


def window_evidence(hp: np.ndarray, grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(evidence, activity, mean detail, detail energy) for one window. All H x W float32."""
    activity = picture_activity(grey)
    if hp.shape[0] > MAX_FRAMES:  # a long window says no more with 24 evenly spaced frames than with all of them
        pick = np.linspace(0, hp.shape[0] - 1, MAX_FRAMES).astype(int)
        hp = hp[pick]
    window = hp.astype(np.float32)
    mean = window.mean(axis=0)
    mean_abs = np.abs(window).mean(axis=0)
    evidence = mean * mean / (mean_abs + 1.0)  # |mean| x coherence: strong and steady
    return evidence, activity, mean, mean_abs


def near_activity(activity: np.ndarray, radius: int = 12) -> np.ndarray:
    """The most the picture moves within `radius` of each place: a structure's own pixels barely change by definition."""
    size = 2 * radius + 1
    return cv2.dilate(activity, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
