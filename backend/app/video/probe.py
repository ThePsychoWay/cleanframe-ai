"""Reads and validates a video's real properties with ffprobe.

Nothing the browser told us (file name, MIME type, size) is trusted here: the container and
codec come from the file's own bytes. Every VideoRejected message is written for the person
who uploaded the file.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_CODECS = frozenset({"h264", "hevc", "vp8", "vp9", "av1", "prores", "mpeg4", "mjpeg"})
# Real container families ffprobe may report for each accepted extension.
CONTAINERS_BY_EXTENSION: dict[str, frozenset[str]] = {
    ".mp4": frozenset({"mov", "mp4"}),
    ".mov": frozenset({"mov", "mp4"}),
    ".webm": frozenset({"matroska", "webm"}),
    ".avi": frozenset({"avi"}),
}
SUPPORTED_FORMATS_TEXT = "an MP4, MOV, WebM or AVI file"
MAX_FPS = 120.0
FFPROBE_TIMEOUT_SECONDS = 30

_UNREADABLE = "We couldn't read this video. The file may be corrupted. Please upload another copy."


class VideoRejected(Exception):
    """The file can't be used. The message is safe to show to the person who uploaded it."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class VideoInfo:
    container: str
    codec: str
    audio_codec: str | None
    width: int  # as displayed, after applying any rotation metadata
    height: int
    fps: float
    duration: float
    bitrate: int | None
    frame_count: int


# What the first bytes of a real video look like. Only these two families are ever opened, and FFmpeg is told which
# one instead of being left to guess from the content: otherwise a file named .mp4 that is really a playlist, a
# concat script or an image sequence would be handed to demuxers that read other files or open network streams.
_ISO_BMFF_ATOMS = (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot")
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"


SNIFF_BYTES = 12


def sniff_head(head: bytes) -> str | None:
    """`mov` (MP4 and QuickTime), `matroska` (WebM) or `avi`, judged from a file's first bytes; None for anything else."""
    if head[:4] == _EBML_MAGIC:
        return "matroska"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    if len(head) >= 8 and head[4:8] in _ISO_BMFF_ATOMS:
        return "mov"
    return None


def sniff_demuxer(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            return sniff_head(handle.read(SNIFF_BYTES))
    except OSError:
        return None


def looks_like_video(head: bytes) -> bool:
    return sniff_head(head) is not None


UNREADABLE_MESSAGE = _UNREADABLE


def demuxer_args(path: Path) -> list[str]:
    """`-f <demuxer>` for FFmpeg/ffprobe input options. Raises VideoRejected for a file that isn't MP4, MOV, WebM or AVI."""
    demuxer = sniff_demuxer(path)
    if demuxer is None:
        raise VideoRejected(_UNREADABLE)
    return ["-f", demuxer]


def run_ffprobe(path: Path, timeout: int = FFPROBE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run ffprobe on a local file and return its JSON. Arguments are a list; no shell is involved."""
    command = [
        "ffprobe",
        "-v", "error",
        # Only the plain file protocol: a crafted file can't make ffprobe fetch URLs or other files.
        "-protocol_whitelist", "file",
        "-probesize", "50M",
        "-analyzeduration", "10M",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        *demuxer_args(path),
        "-i", f"file:{path}",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False, text=True)
    except subprocess.TimeoutExpired:
        raise VideoRejected("Reading this video took too long. The file may be damaged. Please upload another copy.") from None

    if completed.returncode != 0:
        raise VideoRejected(_UNREADABLE)
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise VideoRejected(_UNREADABLE) from None
    if not isinstance(data, dict):
        raise VideoRejected(_UNREADABLE)
    return data


def _fraction(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        n, d = float(numerator), float(denominator)
    except ValueError:
        return None
    return n / d if d != 0 else None


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in (float("inf"), float("-inf")) else None


def rotation_degrees(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, dict) and "rotation" in side_data:
            rotation = _float(side_data["rotation"])
            if rotation is not None:
                return int(round(rotation)) % 360
    rotate_tag = _float((stream.get("tags") or {}).get("rotate"))
    return int(round(rotate_tag)) % 360 if rotate_tag is not None else 0


def _safe_name(value: object, limit: int = 32) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(value).lower())[:limit]


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def parse_video_info(probe: dict[str, Any], extension: str, *, max_duration: float, max_pixels: int) -> VideoInfo:
    ext = extension.lower()
    allowed_containers = CONTAINERS_BY_EXTENSION.get(ext)
    if allowed_containers is None:
        raise VideoRejected(f"Unsupported video format. Please upload {SUPPORTED_FORMATS_TEXT}.")

    fmt = probe.get("format") or {}
    streams = [s for s in probe.get("streams") or [] if isinstance(s, dict)]

    names = {n.strip() for n in str(fmt.get("format_name", "")).split(",") if n.strip()}
    matching = names & allowed_containers
    if not matching:
        raise VideoRejected(
            f"This file's contents don't match its {ext} extension. Please upload a real MP4, MOV, WebM or AVI video."
        )
    container = sorted(matching)[0]

    video = next(
        (s for s in streams if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic") != 1),
        None,
    )
    if video is None:
        raise VideoRejected("No video track was found in this file. Please upload an MP4, MOV, WebM or AVI video.")

    codec = _safe_name(video.get("codec_name", "unknown")) or "unknown"
    if codec not in ALLOWED_CODECS:
        raise VideoRejected(
            f"This video uses the {codec} codec, which can't be read. Please export it as H.264 or H.265 (HEVC) and upload that."
        )

    width, height = video.get("width"), video.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise VideoRejected(_UNREADABLE)
    if rotation_degrees(video) in (90, 270):
        width, height = height, width
    if width * height > max_pixels:
        raise VideoRejected(
            f"This video is {width}×{height}, which is larger than the maximum this server is set up for "
            f"({max_pixels / 1e6:.1f} megapixels, 4K is 8.3). Please export a smaller version."
        )

    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    if fps is None or fps <= 0:
        raise VideoRejected("This video's frame rate couldn't be read. Please upload another copy.")
    if fps > MAX_FPS:
        raise VideoRejected(f"This video runs at {fps:.0f} fps; the maximum supported is {MAX_FPS:.0f} fps.")

    duration = _float(video.get("duration")) or _float(fmt.get("duration"))
    if duration is None or duration <= 0:
        raise VideoRejected("This video's length couldn't be read. Please upload another copy.")
    if duration > max_duration:
        raise VideoRejected(
            f"This video is {format_duration(duration)} long. The current limit is {format_duration(max_duration)}. "
            "Please trim it and upload again."
        )

    nb_frames = str(video.get("nb_frames", ""))
    frame_count = int(nb_frames) if nb_frames.isdigit() and int(nb_frames) > 0 else max(1, round(duration * fps))

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    audio_codec = (_safe_name(audio.get("codec_name", "")) or None) if audio else None

    bit_rate = _float(fmt.get("bit_rate"))
    return VideoInfo(
        container=container,
        codec=codec,
        audio_codec=audio_codec,
        width=width,
        height=height,
        fps=round(fps, 3),
        duration=round(duration, 3),
        bitrate=int(bit_rate) if bit_rate is not None else None,
        frame_count=frame_count,
    )
