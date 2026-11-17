"""Checking the finished video before anyone is told it is ready.

Structural checks (does the file decode, are all frames there, is the size, frame rate and length right, is the
audio there and in step) are exact. The artifact check looks for shimmer inside the removed area while the
rebuilt picture is being written: where a mark stays put between two frames, the rebuilt patch should change about
as much as the real picture next to it does, not much more. When too many frames break that, the render fails with
an explanation instead of being handed over.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.video.ffmpeg import ProcessingError, ffmpeg_command
from app.video.probe import VideoRejected, run_ffprobe

logger = logging.getLogger(__name__)

DECODE_TIMEOUT_SECONDS = 900
LENGTH_TOLERANCE_SECONDS = 0.1  # or two frames, whichever is more
AUDIO_DRIFT_TOLERANCE_SECONDS = 0.3  # how much the audio/picture length gap may differ from the original's

# Artifact check (see ArtifactMonitor). These are deliberately lenient: a render fails only when it is clearly bad.
PATCH_JUMP_MIN = 8.0  # grey levels: below this a change isn't visible shimmer
PATCH_JUMP_FACTOR = 2.0  # flagged when the patch changes this much more than the real picture beside it...
PATCH_JUMP_SLACK = 4.0  # ...allowing for this much noise in the real picture
ARTIFACT_WARN_FRACTION = 0.15
ARTIFACT_FAIL_FRACTION = 0.5
MIN_PATCH_PIXELS = 64
RING_PX = 12


class VerificationFailed(ProcessingError):
    """The finished video didn't pass a check. The message says what is wrong, in plain words."""


class ArtifactMonitor:
    """Measures frame-to-frame shimmer in the rebuilt area as frames are written."""

    def __init__(self) -> None:
        self._previous: tuple[np.ndarray, np.ndarray] | None = None
        self.evaluated = 0
        self.flagged = 0

    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        previous, self._previous = self._previous, (frame, mask)
        if previous is None:
            return
        before, before_mask = previous
        marked_now, marked_before = mask > 127, before_mask > 127
        both = marked_now & marked_before  # the same pixels were rebuilt in both frames
        if int(both.sum()) < MIN_PATCH_PIXELS:
            return
        union = marked_now | marked_before
        rows, cols = np.flatnonzero(union.any(axis=1)), np.flatnonzero(union.any(axis=0))
        height, width = mask.shape
        y0, y1 = max(0, rows[0] - RING_PX - 1), min(height, rows[-1] + RING_PX + 2)
        x0, x1 = max(0, cols[0] - RING_PX - 1), min(width, cols[-1] + RING_PX + 2)
        window = (slice(y0, y1), slice(x0, x1))
        union_u8 = union[window].astype(np.uint8)
        size = 2 * RING_PX + 1
        ring = (cv2.dilate(union_u8, np.ones((size, size), np.uint8)) > 0) & ~(union_u8 > 0)
        if int(ring.sum()) < MIN_PATCH_PIXELS:
            return
        change = np.abs(frame[window].astype(np.int16) - before[window].astype(np.int16)).mean(axis=2)
        patch_jump = float(change[both[window]].mean())
        ring_jump = float(change[ring].mean())
        self.evaluated += 1
        if patch_jump > PATCH_JUMP_MIN and patch_jump > PATCH_JUMP_FACTOR * (ring_jump + PATCH_JUMP_SLACK):
            self.flagged += 1

    @property
    def fraction(self) -> float:
        return self.flagged / self.evaluated if self.evaluated else 0.0


@dataclass(frozen=True)
class Expectation:
    width: int
    height: int
    frames: int
    fps: float
    expect_audio: bool
    source_audio_gap: float | None  # source audio length minus source video length; None when it had no audio


@dataclass
class VerificationReport:
    frames: int
    width: int
    height: int
    fps: float
    video_seconds: float
    audio_seconds: float | None
    artifact_fraction: float
    warnings: list[str] = field(default_factory=list)


def _count_frames(path: Path) -> int:
    command = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "json", "-i", f"file:{path}",
    ]  # fmt: skip
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=DECODE_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        raise VerificationFailed("Checking the restored video took too long.") from None
    if completed.returncode != 0 or completed.stderr.strip():
        logger.warning("restored video has decode errors: %s", completed.stderr.strip()[-500:])
        raise VerificationFailed("The restored video has damaged frames, so it was not delivered. Please try again.")
    try:
        return int(json.loads(completed.stdout)["streams"][0]["nb_read_frames"])
    except (KeyError, IndexError, ValueError, TypeError):
        raise VerificationFailed("The restored video couldn't be read back.") from None


def _audio_decodes(path: Path) -> None:
    try:
        completed = subprocess.run(
            ffmpeg_command(["-i", f"file:{path}", "-map", "0:a", "-f", "null", "-"]),
            capture_output=True, text=True, timeout=DECODE_TIMEOUT_SECONDS, check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        raise VerificationFailed("Checking the restored audio took too long.") from None
    if completed.returncode != 0 or completed.stderr.strip():
        logger.warning("restored audio has decode errors: %s", completed.stderr.strip()[-500:])
        raise VerificationFailed("The audio in the restored video is damaged, so it was not delivered. Please try again.")


def _seconds(*values: object) -> float | None:
    for value in values:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if number == number and number > 0:
            return number
    return None


def _rate(value: object) -> float:
    numerator, _, denominator = str(value).partition("/")
    try:
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return 0.0


def verify_output(path: Path, expected: Expectation, monitor: ArtifactMonitor | None = None) -> VerificationReport:
    try:
        probe = run_ffprobe(path)
    except VideoRejected:
        raise VerificationFailed("The restored video couldn't be read back, so it was not delivered.") from None
    streams = [s for s in probe.get("streams") or [] if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if video is None:
        raise VerificationFailed("The restored video has no picture.")

    if (video.get("width"), video.get("height")) != (expected.width, expected.height):
        raise VerificationFailed(
            f"The restored video is {video.get('width')}×{video.get('height')} but should be {expected.width}×{expected.height}."
        )
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    if abs(fps - expected.fps) > 0.01:
        raise VerificationFailed(f"The frame rate changed during processing ({fps:.2f} instead of {expected.fps:.2f}).")

    frames = _count_frames(path)
    if frames != expected.frames:
        raise VerificationFailed(f"The restored video is missing frames ({frames} instead of {expected.frames}).")

    video_seconds = _seconds(video.get("duration"), (probe.get("format") or {}).get("duration")) or frames / fps
    ideal = frames / fps
    if abs(video_seconds - ideal) > max(LENGTH_TOLERANCE_SECONDS, 2 / fps):
        raise VerificationFailed("The restored video's length doesn't match its frames.")

    audio_seconds: float | None = None
    warnings: list[str] = []
    if expected.expect_audio:
        if not audio:
            raise VerificationFailed("The original audio was lost, so the video was not delivered.")
        _audio_decodes(path)
        audio_seconds = _seconds(audio[0].get("duration"), (probe.get("format") or {}).get("duration"))
        if audio_seconds is not None and expected.source_audio_gap is not None:
            gap = audio_seconds - video_seconds
            if abs(gap - expected.source_audio_gap) > AUDIO_DRIFT_TOLERANCE_SECONDS:
                raise VerificationFailed("The audio no longer lines up with the picture, so the video was not delivered.")
    elif audio:
        raise VerificationFailed("The restored video has audio that shouldn't be there.")

    fraction = monitor.fraction if monitor else 0.0
    if monitor and fraction > ARTIFACT_FAIL_FRACTION:
        raise VerificationFailed(
            "The rebuilt area flickers too much between frames to be a good result. "
            "Try again with higher quality, or choose a different reconstruction setting."
        )
    if monitor and fraction > ARTIFACT_WARN_FRACTION:
        warnings.append("Some of the rebuilt area shimmers between frames. Check the preview before downloading.")

    return VerificationReport(
        frames=frames, width=expected.width, height=expected.height, fps=expected.fps,
        video_seconds=video_seconds, audio_seconds=audio_seconds, artifact_fraction=fraction, warnings=warnings,
    )  # fmt: skip
