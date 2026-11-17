"""Follows marked regions through a video with sparse optical flow.

For each keyframe, corners are found inside the marked shape and followed from frame to frame with
pyramidal Lucas-Kanade. Each step is checked in reverse (a point must come back to where it started)
and fitted with RANSAC to a similarity transform, so points that slip onto the background or get
covered are ignored. The per-step transforms are accumulated into one transform per frame relative
to the keyframe (see shapes.py).

Tracking stops, and says where and why, when it can no longer follow the area: too few points
survive, the area leaves the frame, or the video ends. It never invents a position. The person then
corrects the mask on that frame and tracks again, which is how drift is handled.

What this does not do: it follows image detail inside the marked area. If the marked area is mostly
moving background rather than the logo, it follows the background. A tight shape around the logo works
best.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas.mask import Keyframe, MaskDocument, Region
from app.tracking.raster import rasterize
from app.tracking.shapes import IDENTITY, shape_center, shape_from_keyframe

logger = logging.getLogger(__name__)

NO_DETAIL = "There isn't enough detail in this area to follow. Try marking a little more of the logo's edge."
LOST_DETAIL = "The detail being followed disappeared or became unreliable here."
LEFT_FRAME = "The marked area moved out of the frame."
VIDEO_ENDED = "The preview video ended before this point."
UNSTABLE = "The movement became implausible here."


class TrackingCancelled(Exception):
    """The caller's progress callback asked to stop (for example, the project was deleted)."""


@dataclass(frozen=True)
class TrackerSettings:
    max_corners: int = 150
    corner_quality: float = 0.01
    min_distance: int = 4
    # Pixels added around (positive) or removed from (negative) the shape when looking for corners.
    search_margin: int = 0
    window: int = 15
    pyramid_levels: int = 3
    # A point followed forward and then back must land within this many pixels of where it began.
    round_trip_pixels: float = 1.2
    ransac_pixels: float = 0.8
    min_features: int = 6
    min_inliers: int = 4
    refill_below: float = 0.6
    low_inlier_share: float = 0.3
    patience: int = 5
    min_visible: float = 0.25
    max_step_scale: float = 1.25
    max_total_scale: float = 6.0
    # Drift correction: match the keyframe's own corners directly into the current frame.
    anchor_share: float = 0.35  # share of the keyframe's points that must be found again
    anchor_agreement: float = 0.2  # may differ from the chained estimate by this fraction of the shape's size


DEFAULT_SETTINGS = TrackerSettings()


@dataclass
class StepResult:
    transform: np.ndarray | None  # 3x3 matrix for this step, or None when lost
    reason: str | None = None


class SegmentTracker:
    """Follows one shape, from the frame it was drawn on, through the frames after it."""

    def __init__(
        self,
        anchor_gray: np.ndarray,
        anchor_mask: np.ndarray,
        *,
        rotate: bool,
        settings: TrackerSettings = DEFAULT_SETTINGS,
    ) -> None:
        self.settings = settings
        self.rotate = rotate
        self.height, self.width = anchor_gray.shape
        self.anchor_mask = anchor_mask
        self.anchor_area = int(np.count_nonzero(anchor_mask))
        self.total = np.eye(3)
        self.anchor_gray = anchor_gray
        self.previous = anchor_gray
        self.low_streak = 0
        self.points = self._detect(anchor_gray, anchor_mask, np.empty((0, 1, 2), np.float32))
        self.target = len(self.points)
        self.anchor_points = self.points.copy()
        ys, xs = np.nonzero(anchor_mask)
        self.anchor_size = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)) if len(xs) else 0.0
        self.failure: str | None = NO_DETAIL if self.target < settings.min_features else None

    # --- feature detection -------------------------------------------------------------------

    def _detect(self, gray: np.ndarray, mask: np.ndarray, existing: np.ndarray) -> np.ndarray:
        s = self.settings
        if not mask.any():
            return existing
        # Include a few pixels around the shape: a logo's edge is where the useful corners are.
        size = 2 * abs(s.search_margin) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        search = cv2.dilate(mask, kernel) if s.search_margin >= 0 else cv2.erode(mask, kernel)
        if not search.any():
            search = mask
        for x, y in existing.reshape(-1, 2):
            cv2.circle(search, (int(round(x)), int(round(y))), s.min_distance, 0, -1)
        wanted = s.max_corners - len(existing)
        if wanted <= 0:
            return existing
        found = cv2.goodFeaturesToTrack(
            gray, maxCorners=wanted, qualityLevel=s.corner_quality, minDistance=s.min_distance, mask=search, blockSize=5
        )
        if found is None:
            return existing
        return np.concatenate([existing, found.astype(np.float32)], axis=0)

    # --- one frame ---------------------------------------------------------------------------

    def _fit(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Similarity transform (or scale + translation when not rotating) from a to b, with RANSAC inliers."""
        s = self.settings
        matrix, inliers = cv2.estimateAffinePartial2D(
            a, b, method=cv2.RANSAC, ransacReprojThreshold=s.ransac_pixels, maxIters=500, confidence=0.99
        )
        if matrix is None or inliers is None:
            return None
        keep = inliers.ravel().astype(bool)
        if not self.rotate:
            qa, qb = a[keep], b[keep]
            ma, mb = qa.mean(axis=0), qb.mean(axis=0)
            ca, cb = qa - ma, qb - mb
            denominator = float((ca * ca).sum())
            scale = float((ca * cb).sum()) / denominator if denominator > 1e-6 else 1.0
            translation = mb - scale * ma
            matrix = np.array([[scale, 0.0, translation[0]], [0.0, scale, translation[1]]])
        return matrix, keep


    def _refine(self, gray: np.ndarray, chained: np.ndarray) -> np.ndarray:
        """Correct accumulated drift by matching the keyframe's points directly into `gray`.

        Frame-to-frame chaining adds a little error every step. Matching the keyframe's own corners
        gives an estimate that doesn't accumulate. It is only used when enough of those corners are
        found again and the result agrees with the chained estimate; otherwise (the logo's appearance
        changed, it is partly covered...) the chained estimate stands.
        """
        s = self.settings
        anchor = self.anchor_points
        if len(anchor) < s.min_features:
            return chained
        flat = anchor.reshape(-1, 2).astype(np.float64)
        homogeneous = np.hstack([flat, np.ones((len(flat), 1))])
        guess = (homogeneous @ chained[:2].T).astype(np.float32).reshape(-1, 1, 2)
        lk = {
            "winSize": (s.window, s.window),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            "flags": cv2.OPTFLOW_USE_INITIAL_FLOW,
        }
        found, ok_forward, _ = cv2.calcOpticalFlowPyrLK(self.anchor_gray, gray, anchor, guess.copy(), **lk)
        back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.anchor_gray, found, anchor.copy(), **lk)
        round_trip = np.linalg.norm((back - anchor).reshape(-1, 2), axis=1)
        good = (ok_forward.ravel() == 1) & (ok_back.ravel() == 1) & (round_trip < s.round_trip_pixels)
        if int(good.sum()) < max(s.min_inliers, s.anchor_share * len(anchor)):
            return chained
        fit = self._fit(flat[good].astype(np.float32), found.reshape(-1, 2)[good])
        if fit is None:
            return chained
        matrix, keep = fit
        if int(keep.sum()) < max(s.min_inliers, s.anchor_share * len(anchor)):
            return chained
        refined = np.vstack([matrix, [0.0, 0.0, 1.0]])
        apart = np.linalg.norm((homogeneous @ refined[:2].T) - (homogeneous @ chained[:2].T), axis=1).mean()
        if apart > max(3.0, s.anchor_agreement * self.anchor_size):
            return chained
        return refined

    def update(self, gray: np.ndarray) -> StepResult:
        s = self.settings
        if self.failure:
            return StepResult(None, self.failure)

        lk = {
            "winSize": (s.window, s.window),
            "maxLevel": s.pyramid_levels,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        before = self.points
        after, ok_forward, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, before, None, **lk)
        back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, after, None, **lk)
        round_trip = np.linalg.norm((back - before).reshape(-1, 2), axis=1)
        a_all, b_all = before.reshape(-1, 2), after.reshape(-1, 2)
        inside = (b_all[:, 0] >= 0) & (b_all[:, 0] < self.width) & (b_all[:, 1] >= 0) & (b_all[:, 1] < self.height)
        good = (ok_forward.ravel() == 1) & (ok_back.ravel() == 1) & (round_trip < s.round_trip_pixels) & inside
        if int(good.sum()) < s.min_inliers:
            return self._lose(LOST_DETAIL)

        fit = self._fit(a_all[good], b_all[good])
        if fit is None:
            return self._lose(LOST_DETAIL)
        matrix, keep = fit
        inliers = int(keep.sum())
        if inliers < s.min_inliers:
            return self._lose(LOST_DETAIL)

        step_scale = math.hypot(matrix[0, 0], matrix[1, 0])
        if not (1 / s.max_step_scale <= step_scale <= s.max_step_scale):
            return self._lose(UNSTABLE)

        step = np.vstack([matrix, [0.0, 0.0, 1.0]])
        total = self._refine(gray, step @ self.total)
        total_scale = math.hypot(total[0, 0], total[1, 0])
        if not (1 / s.max_total_scale <= total_scale <= s.max_total_scale):
            return self._lose(UNSTABLE)

        warped = cv2.warpAffine(self.anchor_mask, total[:2], (self.width, self.height), flags=cv2.INTER_NEAREST)
        if self.anchor_area and np.count_nonzero(warped) / self.anchor_area < s.min_visible:
            return self._lose(LEFT_FRAME)

        # Too many of the surviving points disagree with the fitted motion for several frames running.
        if inliers / max(1, int(good.sum())) < s.low_inlier_share:
            self.low_streak += 1
            if self.low_streak >= s.patience:
                return self._lose(LOST_DETAIL)
        else:
            self.low_streak = 0

        self.total = total
        self.previous = gray
        self.points = after[good][keep].reshape(-1, 1, 2).astype(np.float32)
        if len(self.points) < max(s.min_features * 2, s.refill_below * self.target):
            self.points = self._detect(gray, warped, self.points)
        return StepResult(step)

    def _lose(self, reason: str) -> StepResult:
        self.failure = reason
        return StepResult(None, reason)

    # --- result ------------------------------------------------------------------------------

    def params(self, centre: tuple[float, float]) -> tuple[float, float, float, float]:
        """The accumulated motion as [dx, dy, scale, angle] about `centre` (see shapes.py)."""
        a, b = self.total[0, 0], self.total[1, 0]
        scale = math.hypot(a, b)
        angle = math.atan2(b, a) if self.rotate else 0.0
        cx, cy = centre[0] * self.width, centre[1] * self.height
        moved = self.total @ np.array([cx, cy, 1.0])
        return (
            round(float((moved[0] - cx) / self.width), 6),
            round(float((moved[1] - cy) / self.width), 6),
            round(float(scale), 6),
            round(float(angle), 6),
        )


# --- reading frames ------------------------------------------------------------------------------


def read_gray_forward(path: Path, start: int, stop: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame index, grayscale frame) for start..stop inclusive, in order."""
    capture = cv2.VideoCapture(str(path))
    try:
        for _ in range(start):  # decode without converting: cheaper than read()
            if not capture.grab():
                return
        for index in range(start, stop + 1):
            ok, frame = capture.read()
            if not ok:
                return
            yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        capture.release()


def read_gray_backward(path: Path, start: int, *, frame_pixels: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield frames start, start-1, ... 0.

    Video can only be decoded forwards reliably, so frames are decoded in chunks and played back in
    reverse. The chunk size keeps memory around 150 MB whatever the resolution.
    """
    chunk = max(30, min(240, 150_000_000 // max(1, frame_pixels)))
    high = start
    while high >= 0:
        low = max(0, high - chunk + 1)
        frames = list(read_gray_forward(path, low, high))
        if len(frames) != high - low + 1:
            for item in reversed(frames):
                yield item
            return
        yield from reversed(frames)
        high = low - 1


# --- tracking a whole mask document --------------------------------------------------------------


@dataclass
class _Job:
    region: Region
    anchor: Keyframe
    step: int  # +1 forward, -1 backward
    until: int
    tracker: SegmentTracker | None = None
    transforms: list[tuple[float, float, float, float]] = field(default_factory=list)
    lost_at: int | None = None
    reason: str | None = None
    finished: bool = False

    @property
    def rotate(self) -> bool:
        return self.region.type != "rectangle"

    def active_at(self, frame: int) -> bool:
        return self.tracker is not None and not self.finished and 1 <= (frame - self.anchor.frame) * self.step <= abs(
            self.until - self.anchor.frame
        )


Progress = Callable[[int, int], bool]


def _plan(doc: MaskDocument, frame_count: int) -> tuple[list[_Job], list[_Job]]:
    forward: list[_Job] = []
    backward: list[_Job] = []
    last = frame_count - 1
    for region in doc.regions:
        keyframes = region.keyframes
        for i, keyframe in enumerate(keyframes):
            until = keyframes[i + 1].frame - 1 if i + 1 < len(keyframes) else last
            if until > keyframe.frame and not keyframe.hidden:  # nothing to follow where the region isn't shown
                forward.append(_Job(region, keyframe, +1, until))
        first = keyframes[0]
        if first.frame > 0 and not first.hidden:
            backward.append(_Job(region, first, -1, 0))
    return forward, backward


def _start(job: _Job, gray: np.ndarray, width: int, height: int, settings: TrackerSettings) -> None:
    mask = rasterize(shape_from_keyframe(job.anchor), width, height)
    job.tracker = SegmentTracker(gray, mask, rotate=job.rotate, settings=settings)
    if job.tracker.failure:
        job.finished = True
        job.lost_at = job.anchor.frame + job.step
        job.reason = job.tracker.failure


def _advance(job: _Job, frame: int, gray: np.ndarray, centre: tuple[float, float]) -> None:
    assert job.tracker is not None
    result = job.tracker.update(gray)
    if result.transform is None:
        job.finished = True
        job.lost_at = frame
        job.reason = result.reason
        return
    job.transforms.append(job.tracker.params(centre))
    if frame == job.until:
        job.finished = True


def _run_pass(
    jobs: list[_Job],
    frames: Iterator[tuple[int, np.ndarray]],
    *,
    width: int,
    height: int,
    settings: TrackerSettings,
    tick: Callable[[], None],
) -> None:
    centres = {id(job): shape_center(shape_from_keyframe(job.anchor)) for job in jobs}
    by_anchor: dict[int, list[_Job]] = {}
    for job in jobs:
        by_anchor.setdefault(job.anchor.frame, []).append(job)

    last_frame = -1
    for frame, gray in frames:
        last_frame = frame
        tick()
        for job in jobs:
            if job.active_at(frame):
                _advance(job, frame, gray, centres[id(job)])
        for job in by_anchor.get(frame, ()):
            _start(job, gray, width, height, settings)
        # Every job gets its tracker on its own anchor frame, so no tracker means the anchor is still ahead.
        if all(job.tracker is not None and job.finished for job in jobs):
            break

    # Anything still running when the video stops early was cut short, not tracked.
    for job in jobs:
        if not job.finished:
            job.finished = True
            job.lost_at = last_frame + job.step if job.tracker is not None else job.anchor.frame + job.step
            job.reason = VIDEO_ENDED


def track_document(
    doc: MaskDocument,
    video: Path,
    *,
    width: int,
    height: int,
    frame_count: int,
    settings: TrackerSettings = DEFAULT_SETTINGS,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Track every region of `doc` through `video` (the editor's preview) and return the stored document.

    `progress(done, total)` is called as frames are processed; returning False stops tracking.
    """
    forward, backward = _plan(doc, frame_count)
    forward_frames = (max(j.until for j in forward) + 1) if forward else 0
    backward_frames = (max(j.anchor.frame for j in backward) + 1) if backward else 0
    total = forward_frames + backward_frames
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress is not None and (done % 10 == 0 or done == total) and progress(done, total) is False:
            raise TrackingCancelled

    if forward:
        _run_pass(
            forward,
            read_gray_forward(video, 0, forward_frames - 1),
            width=width,
            height=height,
            settings=settings,
            tick=tick,
        )
    if backward:
        start = max(j.anchor.frame for j in backward)
        _run_pass(
            backward,
            read_gray_backward(video, start, frame_pixels=width * height),
            width=width,
            height=height,
            settings=settings,
            tick=tick,
        )

    regions: dict[str, Any] = {region.id: {"segments": []} for region in doc.regions}
    for job in [*forward, *backward]:
        segment: dict[str, Any] = {
            "anchor": job.anchor.frame,
            "step": job.step,
            "until": job.until,
            "transforms": [list(t) for t in job.transforms],
        }
        if job.lost_at is not None:
            segment["lost_at"] = job.lost_at
            segment["reason"] = job.reason
        regions[job.region.id]["segments"].append(segment)

    return {
        "version": 1,
        "width": width,
        "height": height,
        "aspect": height / width,
        "frame_count": frame_count,
        "regions": regions,
    }


__all__ = [
    "DEFAULT_SETTINGS",
    "IDENTITY",
    "SegmentTracker",
    "TrackerSettings",
    "TrackingCancelled",
    "track_document",
]
