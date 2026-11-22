"""After a render: look at the result, not at what the engine said it did.

Two clips are read at low resolution, the original and the restored one, and each marked area is compared on both:

  residual   how much of the overlay's steady structure is still there. It is the evidence the detector works from
             (detail that stays put frame after frame), measured inside the mask before and after. Near 0 = gone.
  flicker    how much the filled area changes between frames compared with the picture just around it. A fill that
             shimmers scores well above 1.

Frame count, size, frame rate and audio are checked separately, by the render itself (video/checks). Nothing here
changes the video; it only decides what the person is told. A check that can't run is reported as not run, never as passed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.detect.evidence import high_pass, window_evidence
from app.detect.survey import read_survey
from app.schemas.mask import MaskDocument
from app.video.masks import MaskSequence
from app.video.source import inspect_source

logger = logging.getLogger(__name__)

PRESERVE_PSNR_WARN = 30.0  # dB: outside the marked areas the result should look like the original (it was re-encoded, so not identical)
PRESERVE_SSIM_WARN = 0.9
RESIDUAL_WARN = 0.5  # more than this share of the overlay's structure still standing
FLICKER_WARN = 2.5  # the fill changes this many times more than its surroundings


def _region_masks(document: MaskDocument, tracking: dict[str, Any] | None, survey_w: int, survey_h: int, frames: np.ndarray, total: int) -> dict[str, np.ndarray]:
    """Per region: bool [samples, H, W] of where it is masked."""
    out: dict[str, np.ndarray] = {}
    for region in document.regions:
        single = MaskDocument(version=1, regions=[region])
        seq = MaskSequence(single, tracking, width=survey_w, height=survey_h, frame_count=total)
        out[region.name] = np.stack([seq.mask(int(f)) > 0 for f in frames])
    return out


def psnr(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> float | None:
    """Peak signal-to-noise ratio in dB over the pixels where `keep` is True (all pixels when None). None if no pixels."""
    diff = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    values = diff[keep] if keep is not None else diff.ravel()
    if values.size == 0:
        return None
    mse = float(values.mean())
    return 99.0 if mse < 1e-10 else float(10.0 * np.log10(255.0**2 / mse))


def ssim(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> float | None:
    """Mean structural similarity of two grey images (11x11 Gaussian window, sigma 1.5, the usual constants) over `keep`."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    index = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    values = index[keep] if keep is not None else index.ravel()
    return None if values.size == 0 else float(values.mean())


def preservation(before: np.ndarray, after: np.ndarray, marked: np.ndarray) -> dict[str, Any]:
    """How much of the picture outside every marked area survived. `before`/`after` are [N,H,W] grey, `marked` [N,H,W] bool.

    The mask is grown a little first so the soft edge of a filled area isn't counted against the untouched picture.
    """
    grown = np.stack([cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) for m in marked])
    outside = ~grown
    psnrs, ssims = [], []
    for i in range(before.shape[0]):
        if outside[i].sum() < 100:
            continue
        p, q = psnr(before[i], after[i], outside[i]), ssim(before[i], after[i], outside[i])
        if p is not None and q is not None:
            psnrs.append(p)
            ssims.append(q)
    if not psnrs:
        return {"checked": False}
    worst_psnr, mean_ssim = float(min(psnrs)), float(np.mean(ssims))
    problems = []
    if worst_psnr < PRESERVE_PSNR_WARN:
        problems.append("parts of the video outside the marked areas changed more than encoding alone explains")
    if mean_ssim < PRESERVE_SSIM_WARN:
        problems.append("the picture outside the marked areas lost structure")
    return {"checked": True, "psnr_db": round(worst_psnr, 1), "ssim": round(mean_ssim, 4), "frames": len(psnrs), "problems": problems}


def check_result(original: Path, restored: Path, document: MaskDocument, tracking: dict[str, Any] | None, frame_count: int) -> dict[str, Any]:
    started = time.monotonic()
    before = read_survey(original, inspect_source(original), frame_count)
    after = read_survey(restored, inspect_source(restored), frame_count)
    if before.grey.shape != after.grey.shape:
        return {"checked": False, "reason": "the restored video and the original didn't line up for comparison", "regions": []}
    masks = _region_masks(document, tracking, before.width, before.height, before.frames, before.source_frames)
    hp_before, hp_after = high_pass(before.grey), high_pass(after.grey)
    ev_before = window_evidence(hp_before, before.grey)[0]
    ev_after = window_evidence(hp_after, after.grey)[0]
    regions = []
    for name, stack in masks.items():
        any_mask = stack.any(axis=0)
        if not stack.any():
            regions.append({"name": name, "frames_masked": 0, "residual": None, "flicker": None, "verdict": "not_masked"})
            continue
        frames_masked = int(stack.reshape(stack.shape[0], -1).any(axis=1).sum())
        sample = any_mask & (ev_before > 6.0)  # where the overlay's structure was
        residual = float(ev_after[sample].sum() / max(1e-6, ev_before[sample].sum())) if sample.any() else None
        flicker = _flicker(after.grey, stack)
        problems = []
        if residual is not None and residual > RESIDUAL_WARN:
            problems.append("part of the overlay may still be visible")
        if flicker is not None and flicker > FLICKER_WARN:
            problems.append("the filled area may flicker")
        regions.append(
            {
                "name": name, "frames_masked": frames_masked, "of_frames": int(stack.shape[0]),
                "residual": None if residual is None else round(residual, 3), "flicker": None if flicker is None else round(flicker, 2),
                "verdict": "check" if problems else "clean", "problems": problems,
            }
        )  # fmt: skip
    union = np.stack(list(masks.values())).any(axis=0) if masks else np.zeros(before.grey.shape, dtype=bool)
    outside = preservation(before.grey, after.grey, union)
    return {"checked": True, "regions": regions, "outside": outside, "seconds": round(time.monotonic() - started, 2)}


def _flicker(grey: np.ndarray, stack: np.ndarray) -> float | None:
    """Frame-to-frame change inside the mask divided by the same just outside it, over frames where both were masked."""
    ratios = []
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i in range(1, stack.shape[0]):
        if not (stack[i].any() and stack[i - 1].any()):
            continue
        inside = stack[i] & stack[i - 1]
        ring = cv2.dilate(inside.astype(np.uint8), ring_kernel).astype(bool) & ~cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if inside.sum() < 20 or ring.sum() < 20:
            continue
        diff = np.abs(grey[i] - grey[i - 1])
        ratios.append((float(diff[inside].mean()), float(diff[ring].mean())))
    if len(ratios) < 3:
        return None
    inner, outer = np.mean([r[0] for r in ratios]), np.mean([r[1] for r in ratios])
    return float(inner / max(outer, 1.0))  # the floor keeps a still picture's compression noise from inflating the ratio


def severity(report: dict[str, Any]) -> float:
    """One number for how bad a check report is, so two renders of the same clip can be compared. 0 = nothing flagged.

    A check that did not run is worse than a clean one (nothing is known) but better than a flagged result, so a retry never
    replaces a result that was proven clean with one that could not be checked.
    """
    if not report.get("checked"):
        return 0.25
    total = float(len((report.get("outside") or {}).get("problems", [])))
    for region in report["regions"]:
        if region["verdict"] != "check":
            continue
        residual, flicker = region.get("residual"), region.get("flicker")
        total += 1.0
        if residual is not None:
            total += max(0.0, residual - RESIDUAL_WARN)
        if flicker is not None:
            total += max(0.0, flicker - FLICKER_WARN) / FLICKER_WARN
    return total


def retry_quality(current: str, report: dict[str, Any]) -> str | None:
    """The quality to render again at, or None when a retry would not change anything or nothing was flagged."""
    if current == "high" or not report.get("checked"):
        return None
    return "high" if any(r["verdict"] == "check" for r in report["regions"]) else None


def warnings_from(report: dict[str, Any]) -> list[str]:
    """Plain sentences for the person. Empty when everything checked out."""
    if not report.get("checked"):
        return [f"The result couldn't be checked automatically: {report.get('reason', 'the check did not run')}."]
    notes = []
    for region in report["regions"]:
        if region["verdict"] == "check":
            notes.append(f"{region['name']}: {' and '.join(region['problems'])}. Look at it before you use the video.")
    for problem in (report.get("outside") or {}).get("problems", []):
        notes.append(f"{problem[0].upper()}{problem[1:]}. Look at it before you use the video.")
    return notes
