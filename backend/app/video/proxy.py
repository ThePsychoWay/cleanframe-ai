"""Builds the editor's preview media: a small browser-playable proxy and a poster thumbnail.

The person marks overlays on the proxy, never on the full-resolution original, so the editor stays
fast. The proxy keeps the original's frame rate (constant, so frame N is always at N / fps seconds)
and its display orientation. It has no audio: the editor is for choosing regions, not for listening.
Frame numbers in saved masks refer to the source video; the proxy has the same frame count.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.video.probe import VideoRejected, demuxer_args, run_ffprobe

logger = logging.getLogger(__name__)

PROXY_MAX_HEIGHT = 720
PROXY_TIMEOUT_SECONDS = 180
THUMBNAIL_WIDTH = 640
THUMBNAIL_TIMEOUT_SECONDS = 30

_FAILED = "We couldn't prepare a preview of this video. Please try again, or upload another copy."


class MediaPreparationError(Exception):
    """Preview media couldn't be produced. The message is safe to show to the person."""

    def __init__(self, message: str = _FAILED) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class ProxyInfo:
    width: int
    height: int
    # The proxy is constant-frame-rate, so this is the project's timeline: the editor, tracking and the final
    # render all count frames the same way. For a constant-rate source it equals the source's own numbers; a
    # variable-rate source is resampled and can differ.
    fps: float
    frame_count: int


def _run_ffmpeg(args: list[str], timeout: int) -> None:
    # An argument list, never a shell. Only the plain file protocol is allowed, so a crafted
    # container can't make ffmpeg fetch URLs or read other local files through playlists.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-protocol_whitelist", "file", *args]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False, text=True)
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out", extra={"seconds": timeout})
        raise MediaPreparationError("Preparing the preview took too long. Try a shorter or smaller video.") from None
    if completed.returncode != 0:
        logger.warning("ffmpeg failed: %s", completed.stderr.strip()[-500:])
        raise MediaPreparationError


def build_proxy(source: Path, destination: Path, *, max_height: int = PROXY_MAX_HEIGHT) -> ProxyInfo:
    """Transcode to H.264 at most `max_height` tall with even dimensions. Never upscales."""
    scale = (
        f"scale=w='trunc(min(iw,iw*{max_height}/ih)/2)*2':h='trunc(min(ih,{max_height})/2)*2',"
        "setsar=1"
    )
    _run_ffmpeg(
        [
            *demuxer_args(source),
            "-i", f"file:{source}",
            "-map", "0:v:0",
            "-an",
            "-vf", scale,
            "-fps_mode", "cfr",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-f", "mp4",
            f"file:{destination}",
        ],
        PROXY_TIMEOUT_SECONDS,
    )
    return _read_proxy_info(destination)


def _read_proxy_info(path: Path) -> ProxyInfo:
    try:
        probe = run_ffprobe(path)
    except VideoRejected:
        raise MediaPreparationError from None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and stream.get("width") and stream.get("height"):
            fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
            frames = str(stream.get("nb_frames", ""))
            if fps is None or not frames.isdigit() or int(frames) <= 0:
                raise MediaPreparationError
            return ProxyInfo(width=int(stream["width"]), height=int(stream["height"]), fps=round(fps, 3), frame_count=int(frames))
    raise MediaPreparationError


def _rate(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        n, d = float(numerator), float(denominator)
    except ValueError:
        return None
    return n / d if d > 0 and n > 0 else None


def build_thumbnail(proxy: Path, destination: Path, *, duration: float) -> None:
    """One poster frame, taken from the proxy (small, already upright) about a second in."""
    seek = min(1.0, max(duration, 0.0) / 2)
    for at in (seek, 0.0):
        try:
            _run_ffmpeg(
                [
                    "-ss", f"{at:.3f}",
                    "-i", f"file:{proxy}",
                    "-frames:v", "1",
                    "-vf", f"scale='min(iw,{THUMBNAIL_WIDTH})':-2",
                    "-q:v", "3",
                    f"file:{destination}",
                ],
                THUMBNAIL_TIMEOUT_SECONDS,
            )
        except MediaPreparationError:
            if at == 0.0:
                raise
            continue
        if destination.exists() and destination.stat().st_size > 0:
            return
    raise MediaPreparationError
