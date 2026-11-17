"""Automatic detection, start to finish: a clip in, the overlays on every frame out, with a time record."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from app.detect.build import runs_of
from app.detect.candidates import ISOLATION, MARGIN, find_candidates
from app.detect.drift import carry_along, find_drifting, learn_template
from app.detect.evidence import (
    ACTIVITY_MIN,
    high_pass,
    near_activity,
    picture_activity,
    window_evidence,
)
from app.detect.follow import ACCEPT_ELSEWHERE, Matcher, follow
from app.detect.glyph import lettering_like
from app.detect.refine import fill_dim_frames, grow_extent, settle_static
from app.detect.survey import Survey, read_survey
from app.detect.types import Candidate, DetectionReport, Overlay
from app.video.source import SourceMedia

logger = logging.getLogger(__name__)

MAX_OVERLAYS = 8
CONFIDENT = 0.8  # median match needed before a mark is searched for elsewhere
STEADY = 0.9  # the 10th-percentile match over the frames it is on: an overlay does not drop out
COHERENT = 0.88  # see track_stats
STRONG = 18.0  # evidence (grey levels) that lets a slightly less steady match through
MIN_SECONDS = 0.4  # shorter than this on screen is not a watermark
CAPTION_WIDTH = 0.42  # wider than this fraction of the frame, or this much area, reads as a title or caption
CAPTION_AREA = 0.04
CENTRED_CAPTION_WIDTH = 0.2  # text this wide, in the middle of the picture, is a title or subtitle
CENTRE_BAND = 0.15  # how far from the middle (as a share of width) still counts as centred
MIDDLE_BAND = (0.15, 0.85)  # vertical band where a centred line of text is part of the video
WINDOW_SECONDS = 1.0
# A mark that can't be matched perfectly on every frame of real, compressed footage is still an overlay when the whole
# track agrees with itself, it is lettering or a logo, and it stays on screen for most of the clip (scene structure
# that stays put is blurry or one-directional, and rarely lasts that long). See _persistent_lettering.
PERSISTENT_SHARE = 0.5
PERSISTENT_COHERENCE = 0.85
PERSISTENT_STRENGTH = 12.0
PERSISTENT_MATCH = 0.65
PERSISTENT_SCORE = 12.0  # evidence of the strongest part of the track: scene structure that happens to stay put scores lower than a mark
PERSISTENT_WANDER = 4.0  # survey px: a painted overlay does not drift
SUBTITLE_SHARE = 0.6  # centred text on screen for less than this share of the clip is a subtitle, wherever it sits
Progress = Callable[[str, float], None]


def scene_cuts(grey: np.ndarray) -> list[int]:
    """Sample indices where a new shot starts: a jump in the picture far above its usual frame-to-frame change."""
    if grey.shape[0] < 3:
        return []
    small = np.stack([cv2.resize(f, (48, 48), interpolation=cv2.INTER_AREA) for f in grey])
    change = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    typical = float(np.median(change)) + 1.0
    return [i + 1 for i, c in enumerate(change) if c > max(25.0, 6.0 * typical)]


def windows(count: int, fps: float, stride: int, cuts: list[int]) -> list[tuple[int, int]]:
    """Whole clip, each shot, and overlapping ~1 s slices: a mark that is only there for part of the clip, or that sits
    somewhere new from some moment on, stands out in the window that holds it."""
    result: list[tuple[int, int]] = [(0, count)]
    edges = [0, *cuts, count]
    result += [(a, b) for a, b in zip(edges, edges[1:], strict=False) if b - a >= 4 and (a, b) != (0, count)]
    size = max(6, round(WINDOW_SECONDS * fps / stride))
    step = max(3, size // 2)
    slices = [(a, min(count, a + size)) for a in range(0, max(1, count - size // 2), step)]
    if len(slices) > 40:  # long clips: fewer, wider slices
        step = -(-count // 40)
        slices = [(a, min(count, a + 2 * step)) for a in range(0, count, step)]
    result += [w for w in slices if w[1] - w[0] >= 4]
    return list(dict.fromkeys(result))


def _rank(c: Candidate) -> tuple[bool, float]:
    return (c.basis == "motion", c.score * (min(c.activity, 20.0) if c.basis == "motion" else 1.0))


def _distinct(pool: list[Candidate]) -> list[Candidate]:
    """The same box found in many windows is one candidate: keep the strongest, best-evidenced reading of it."""
    pool = sorted(pool, key=_rank, reverse=True)
    kept: list[Candidate] = []
    for candidate in pool:
        if any(_overlap(tuple(map(float, candidate.box)), tuple(map(float, k.box))) > 0.6 for k in kept):  # type: ignore[arg-type]
            continue
        kept.append(candidate)
    return kept


def _claimed(tracks: list[Overlay], candidate: Candidate) -> bool:
    """Is this box already explained by an overlay that has been followed? (same place, same moment)"""
    mid = (candidate.window[0] + candidate.window[1]) // 2
    x, y, w, h = candidate.box
    for overlay in tracks:
        if not overlay.present[mid]:
            continue
        ox, oy, ow, oh = overlay.box_at(mid)
        ix, iy = max(0.0, min(x + w, ox + ow) - max(x, ox)), max(0.0, min(y + h, oy + oh) - max(y, oy))
        inter = ix * iy
        if inter / max(1.0, min(w * h, ow * oh)) > 0.5:
            return True
    return False


def _new_overlay(source: Candidate | Overlay, count: int, index: int) -> Overlay:
    kind = source.kind if isinstance(source, Overlay) else ("text" if source.texty else "logo")
    return Overlay(
        id=index, kind=kind, template=source.template, support=source.support, basis=source.basis,
        score=source.score, activity=source.activity, present=np.zeros(count, bool),
        x=np.zeros(count, np.float32), y=np.zeros(count, np.float32), s=np.ones(count, np.float32), match=np.zeros(count, np.float32),
    )  # fmt: skip


def _repeats(matcher: Matcher, parent: Overlay, tracks: list[Overlay], count: int) -> list[tuple[int, tuple[float, float, float]]]:
    """The same mark somewhere else (another size, another corner): look for the template on a spread of frames."""
    picks = np.unique(np.linspace(0, count - 1, min(count, 12)).astype(int))
    seeds: list[tuple[int, tuple[float, float, float]]] = []
    for i in picks:
        for score, x, y, s in matcher.everywhere(int(i), parent.template):
            if score < ACCEPT_ELSEWHERE + 0.05:
                continue
            h, w = parent.template.shape
            box = (x, y, w * s, h * s)
            if any(t.present[i] and _overlap(box, t.box_at(int(i))) > 0.3 for t in [*tracks, parent]):
                continue
            if any(abs(x - sx) < 10 and abs(y - sy) < 10 and abs(s - ss) < 0.2 for _, (sx, sy, ss) in seeds):
                continue
            seeds.append((int(i), (x, y, s)))
    return seeds


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix, iy = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])), max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, min(a[2] * a[3], b[2] * b[3]))


def detect(survey: Survey, *, progress: Progress | None = None, appearance: bool = True, drift: bool = True) -> DetectionReport:
    """Overlays on the surveyed clip. Nothing is assumed: every overlay returned was matched on the frames it claims."""
    timings: dict[str, float] = {}
    notes: list[str] = []
    note = progress or (lambda stage, fraction: None)
    clock = time.monotonic()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.monotonic()
        timings[name] = round(now - clock, 3)
        clock = now

    count = survey.count
    grey = survey.grey
    note("looking", 0.0)
    hp = high_pass(grey)
    activity = picture_activity(grey)
    typical = float(np.median(activity))
    cuts = scene_cuts(grey)
    lap("prepare")
    if count < 6:
        notes.append("The video is too short to tell an overlay from the picture.")
        return DetectionReport([], timings, notes, count, survey.stride, typical)
    if float(near_activity(activity).max()) < ACTIVITY_MIN:
        notes.append("The picture hardly changes, so a fixed overlay can't be proven from motion; only marks that look like marks are considered.")

    pool: list[Candidate] = []
    all_windows = windows(count, survey.fps, survey.stride, cuts)
    for k, window in enumerate(all_windows):
        pool += find_candidates(hp, grey, window, appearance)
        note("looking", 0.3 * (k + 1) / len(all_windows))
    pool = _distinct(pool)
    lap("candidates")

    detail = window_evidence(hp, grey)[3]
    matcher = Matcher(hp)
    tracks: list[Overlay] = []
    tried = 0
    for candidate in pool:
        if len(tracks) >= MAX_OVERLAYS or tried >= 2 * MAX_OVERLAYS:
            break
        if _claimed(tracks, candidate):
            continue
        tried += 1
        overlay = _new_overlay(candidate, count, len(tracks) + 1)
        a, b = candidate.window
        seed = (a + b) // 2
        x, y, w, h = candidate.box
        follow(matcher, overlay, seed, (float(x), float(y), 1.0), tracks)
        if overlay.frames_present >= 3 and sharpen_template(overlay, hp):  # second pass with the template made only of what stayed
            best = int(np.argmax(np.where(overlay.present, overlay.match, -1)))
            follow(matcher, overlay, best, (float(overlay.x[best]), float(overlay.y[best]), float(overlay.s[best])), tracks)
        if overlay.frames_present < max(3, round(MIN_SECONDS * survey.fps / survey.stride)):
            continue
        tracks.append(overlay)
        note("following", min(0.9, 0.3 + 0.6 * len(tracks) / MAX_OVERLAYS))
    lap("follow")

    height, width = survey.height, survey.width
    for overlay in tracks:
        overlay.coherence, overlay.strength = track_stats(overlay, hp)
        _judge(overlay, activity, detail, width, height, appearance)
    lap("judge")
    if drift:
        for overlay in find_drifting(grey, hp, survey.fps, survey.stride):
            if not learn_template(overlay, hp):  # drifting marks have their own slots: static ones may have filled MAX_OVERLAYS
                continue
            overlay.id = len(tracks) + 1
            carry_along(matcher, overlay)
            overlay.coherence, overlay.strength = track_stats(overlay, hp)
            _judge_drift(overlay, survey, activity, width, height)
            tracks.append(overlay)
    lap("drift")
    for overlay in tracks:
        if overlay.verdict == "remove":
            grow_extent(overlay, hp)
            if settle_static(overlay, survey.fps, survey.stride) and "text-like, on screen for most of the clip" in overlay.notes:
                fill_dim_frames(matcher, overlay)
    lap("refine")

    # A proven overlay may sit elsewhere too (another size, another corner): look for copies of it.
    for parent in [o for o in tracks if o.verdict == "remove" and o.confidence >= CONFIDENT]:
        for at, pose in _repeats(matcher, parent, tracks, count)[:3]:
            if len(tracks) >= MAX_OVERLAYS:
                break
            twin = _new_overlay(parent, count, len(tracks) + 1)
            twin.notes.append("same mark as another one, found elsewhere")
            follow(matcher, twin, at, pose, tracks)
            if twin.frames_present >= max(3, round(MIN_SECONDS * survey.fps / survey.stride)):
                _judge(twin, activity, detail, width, height, appearance)
                tracks.append(twin)
    lap("repeats")
    return DetectionReport(tracks, timings, notes, count, survey.stride, typical)


def _aligned(overlay: Overlay, hp: np.ndarray, limit: int = 60) -> np.ndarray | None:
    """The overlay's patch from up to `limit` of the frames it is on, each cut out at its own pose, at template size."""
    idx = np.flatnonzero(overlay.present)
    if len(idx) == 0:
        return None
    idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), limit)).astype(int)]
    h, w = overlay.template.shape
    patches = []
    for i in idx:
        s = float(overlay.s[i])
        pw, ph = max(2, round(w * s)), max(2, round(h * s))
        x, y = int(round(overlay.x[i])), int(round(overlay.y[i]))
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(hp.shape[2], x + pw), min(hp.shape[1], y + ph)
        if x1 - x0 < pw or y1 - y0 < ph:
            continue
        patch = hp[i, y0:y1, x0:x1].astype(np.float32)
        patches.append(cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA) if (pw, ph) != (w, h) else patch)
    return np.stack(patches) if len(patches) >= 3 else None


def sharpen_template(overlay: Overlay, hp: np.ndarray) -> bool:
    """Rebuild the template from everything the overlay was found on: the part that is the same on every frame is the
    overlay; whatever beside it changed is scene. Returns True when the template changed enough to follow it again."""
    stack = _aligned(overlay, hp)
    if stack is None:
        return False
    mean, mean_abs = stack.mean(axis=0), np.abs(stack).mean(axis=0)
    coherence = np.abs(mean) / (mean_abs + 1.0)
    keep = (coherence > 0.85) & (np.abs(mean) > 0.25 * float(np.abs(mean).max()))
    if keep.sum() < 24:
        return False
    overlay.template, overlay.support = mean, keep.astype(np.uint8)
    return True


def track_stats(overlay: Overlay, hp: np.ndarray) -> tuple[float, float]:
    """(coherence, strength) of the overlay's own pixels over every frame it is on, each frame cut out at its own pose.

    Coherence is |mean| / mean|.| of the detail picture: near 1 when the same structure is there, unchanged, on every
    frame, however the picture behind it moves. Scene structure that gets covered or relit scores lower over a whole
    clip. Strength is the mean detail (grey levels) over the structure.
    """
    stack = _aligned(overlay, hp)
    support = overlay.support.astype(bool)
    if stack is None or not support.any():
        return 0.0, 0.0
    mean, mean_abs = stack.mean(axis=0), np.abs(stack).mean(axis=0)
    coherence = np.abs(mean) / (mean_abs + 1.0)
    return float(np.median(coherence[support])), float(np.abs(mean)[support].mean())


def _ring_activity(overlay: Overlay, activity: np.ndarray, i: int) -> float:
    x, y, w, h = overlay.box_at(i)
    m = 8
    x0, y0 = max(0, int(x) - m), max(0, int(y) - m)
    x1, y1 = min(activity.shape[1], int(x + w) + m), min(activity.shape[0], int(y + h) + m)
    patch = activity[y0:y1, x0:x1].copy()
    if patch.size == 0:
        return 0.0
    inner = np.zeros_like(patch, bool)
    inner[max(0, int(y) - y0) : max(0, int(y + h) - y0), max(0, int(x) - x0) : max(0, int(x + w) - x0)] = True
    ring = patch[~inner]
    return float(np.percentile(ring, 80)) if ring.size else 0.0


def _judge(overlay: Overlay, activity: np.ndarray, detail: np.ndarray, width: int, height: int, appearance: bool = True) -> None:
    """Decide whether this is a watermark to remove, with the reason written down either way.

    Three questions, in this order: is it steady enough on screen to be an overlay at all (a scene's own structure
    gets covered and relit, so over a whole clip it matches worse); is it a title or caption rather than a mark; and
    does the picture around it move, which is what proves it isn't part of the scene. Where nothing moves around it
    the call rests on how marks look and where they sit, and the result says so.
    """
    idx = np.flatnonzero(overlay.present)
    mid = int(idx[len(idx) // 2])
    overlay.activity = _ring_activity(overlay, activity, mid)
    matches = overlay.match[idx]
    overlay.confidence = round(float(np.clip(np.median(matches), 0, 1)), 3)
    steady = float(np.percentile(matches, 10))
    runs = [(a, b) for a, b in runs_of(overlay) if b - a + 1 >= 3]
    centres = [(float(np.median(overlay.x[a : b + 1])), float(np.median(overlay.y[a : b + 1]))) for a, b in runs]
    far = max(24.0, 1.5 * max(overlay.template.shape))  # a different place, not a drift in size or a wobble
    apart = any(np.hypot(p[0] - q[0], p[1] - q[1]) > far for k, p in enumerate(centres) for q in centres[k + 1 :])
    big_enough = min(overlay.template.shape) >= 16  # a thin sliver matches edges everywhere
    if apart and big_enough and overlay.confidence >= 0.85 and all(float(np.median(overlay.match[a : b + 1])) >= 0.7 for a, b in runs):
        steady = max(steady, STEADY)  # it was found again, as itself, in more than one place: that is not scene structure
        overlay.notes.append("moves between positions")
    widths = np.array([overlay.template.shape[1] * overlay.s[i] for i in idx])
    heights = np.array([overlay.template.shape[0] * overlay.s[i] for i in idx])
    w_frac, area = float(np.median(widths) / width), float(np.median(widths * heights) / (width * height))
    overlay.verdict = "skipped"
    share = overlay.frames_present / max(1, len(overlay.present))
    persistent = _persistent_lettering(overlay, share)
    if persistent and "text-like, on screen for most of the clip" not in overlay.notes:
        overlay.notes.append("text-like, on screen for most of the clip")
    if not persistent and ((steady < STEADY and not (steady >= 0.8 and overlay.score >= STRONG)) or overlay.coherence < COHERENT):
        overlay.reason = "Something is there, but it isn't steady enough on screen to be sure it is an overlay and not part of the scene."
        return
    if w_frac >= CAPTION_WIDTH or area >= CAPTION_AREA:
        overlay.reason = "Large text or graphics across the picture: treated as the video's own title or caption, not a watermark."
        return
    cx = float(np.median([overlay.x[i] + widths[k] / 2 for k, i in enumerate(idx)]))
    cy = float(np.median([overlay.y[i] + heights[k] / 2 for k, i in enumerate(idx)]))
    if (
        "moves between positions" not in overlay.notes
        and w_frac >= CENTRED_CAPTION_WIDTH
        and abs(cx / width - 0.5) <= CENTRE_BAND
        and (MIDDLE_BAND[0] <= cy / height <= MIDDLE_BAND[1] or share < SUBTITLE_SHARE)
    ):
        overlay.reason = "Text in the middle of the picture: treated as the video's own title or caption, not a watermark."
        return
    if overlay.activity >= ACTIVITY_MIN or "moves between positions" in overlay.notes:
        overlay.verdict, overlay.basis = "remove", "motion"
        return
    if overlay.notes and overlay.confidence >= 0.9 and 0.85 <= float(np.median(overlay.s[idx])) <= 1.2:
        overlay.verdict, overlay.basis = "remove", "copy"
        overlay.reason = "Recognised as the same mark as another one that was proven to be an overlay."
        return
    if appearance and _looks_like_a_mark(overlay, detail, mid, width, height):
        overlay.verdict, overlay.basis = "remove", "appearance"
        overlay.reason = "Judged by its look and position only: the picture around it doesn't move, so there was no motion to prove it."
        overlay.notes.append("appearance only")
        return
    overlay.reason = (
        "The picture around it doesn't change, so nothing proves it is an overlay and not part of the scene."
        if not appearance
        else "The picture around it doesn't change and it doesn't look like a mark, so it can't be told apart from the scene."
    )


DRIFT_COHERENCE = 0.6  # aligned over the whole path, a drifting mark keeps its look; the picture it passes over does not
DRIFT_STRENGTH = 6.0


def _judge_drift(overlay: Overlay, survey: Survey, activity: np.ndarray, width: int, height: int) -> None:
    """A mark found moving over a still part of the picture. What proves it isn't the picture: the picture there doesn't
    move, and this does, along a smooth path, keeping the same look."""
    idx = np.flatnonzero(overlay.present)
    overlay.confidence = round(float(np.clip(np.median(overlay.match[idx]), 0, 1)), 3)
    widths = overlay.template.shape[1] * overlay.s[idx]
    heights = overlay.template.shape[0] * overlay.s[idx]
    w_frac, area = float(np.median(widths) / width), float(np.median(widths * heights) / (width * height))
    overlay.verdict = "skipped"
    if overlay.coherence < DRIFT_COHERENCE or overlay.strength < DRIFT_STRENGTH:
        overlay.reason = "Something moves across the picture, but it doesn't keep the same look from frame to frame, so it may be part of the scene."
        return
    if w_frac >= CAPTION_WIDTH or area >= CAPTION_AREA:
        overlay.reason = "Large text or graphics across the picture: treated as the video's own title or caption, not a watermark."
        return
    overlay.verdict = "remove"
    overlay.reason = "Moves across a still part of the picture on a smooth path, keeping the same look: a drifting mark."
    overlay.notes.append("moves across the picture")


def _persistent_lettering(overlay: Overlay, share: float) -> bool:
    """Strong whole-track agreement, lettering or logo strokes, on screen for most of the clip."""
    idx = np.flatnonzero(overlay.present)
    if len(idx) < 3:
        return False
    # An overlay is painted at one place on the screen. A scene object that merely resembles lettering drifts with the camera.
    wander = np.maximum(np.abs(overlay.x[idx] - np.median(overlay.x[idx])), np.abs(overlay.y[idx] - np.median(overlay.y[idx])))
    if float(np.percentile(wander, 95)) > PERSISTENT_WANDER:
        return False
    return bool(
        share >= PERSISTENT_SHARE
        and overlay.coherence >= PERSISTENT_COHERENCE
        and overlay.strength >= PERSISTENT_STRENGTH
        and overlay.score >= PERSISTENT_SCORE
        and float(np.median(overlay.match[idx])) >= PERSISTENT_MATCH
        and lettering_like(overlay.template)
    )


def _looks_like_a_mark(overlay: Overlay, detail: np.ndarray, i: int, width: int, height: int) -> bool:
    """Small, in a corner, and alone: plain picture all round it. Only corners: a label sitting along an edge (the years
    above a chart, say) is part of the picture far more often than a watermark is."""
    x, y, w, h = overlay.box_at(i)
    cx, cy = x + w / 2, y + h / 2
    if min(cx, width - cx) > MARGIN * width or min(cy, height - cy) > MARGIN * height:
        return False
    reach = int(0.6 * max(w, h)) + 4
    x0, y0, x1, y1 = max(0, int(x) - reach), max(0, int(y) - reach), min(width, int(x + w) + reach), min(height, int(y + h) + reach)
    inside = np.zeros((height, width), bool)
    inside[max(0, int(y)) : int(y + h), max(0, int(x)) : int(x + w)] = True
    around = np.zeros((height, width), bool)
    around[y0:y1, x0:x1] = True
    around &= ~inside
    if not inside.any() or not around.any():
        return False
    return float(detail[inside].mean()) >= ISOLATION * float(detail[around].mean() + 0.5)


def survey_and_detect(
    path: Path, media: SourceMedia, frame_count: int, *, progress: Progress | None = None, appearance: bool = True, drift: bool = True
) -> tuple[Survey, DetectionReport]:
    started = time.monotonic()
    survey = read_survey(path, media, frame_count, progress=(lambda done, total: progress("reading", done / max(1, total))) if progress else None)
    read_seconds = round(time.monotonic() - started, 3)
    report = detect(survey, progress=progress, appearance=appearance, drift=drift)
    report.timings = {"read_frames": read_seconds, **report.timings}
    return survey, report


def removable(report: DetectionReport) -> list[Overlay]:
    return [o for o in report.overlays if o.verdict == "remove"]
