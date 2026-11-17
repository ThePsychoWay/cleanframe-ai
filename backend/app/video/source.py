"""What the render pipeline needs to know about the original video beyond the basic analysis: how its colours are
defined, what audio it carries, and whether it can be processed faithfully at all."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.video.ffmpeg import ProcessingError
from app.video.probe import VideoRejected, rotation_degrees, run_ffprobe

HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})  # PQ and HLG
# colour tag -> (what to call it when tagging output, name the scale filter understands)
MATRICES: dict[str, str] = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470", "bt2020nc": "bt2020"}
_PRIMARIES_FOR = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470bg", "bt2020nc": "bt2020"}
_TRANSFER_FOR = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "smpte170m", "bt2020nc": "bt709"}
_UNKNOWN = frozenset({"", "unknown", "unspecified", "reserved", "gbr", "none"})


@dataclass(frozen=True)
class ColorSpec:
    """How the picture's colours are defined. It is read from the source and written to the output, so players
    interpret the result exactly as they interpreted the original."""

    matrix: str  # key of MATRICES
    primaries: str
    transfer: str
    range: str  # "tv" (limited) or "pc" (full)

    @property
    def scale_matrix(self) -> str:
        return MATRICES[self.matrix]


@dataclass(frozen=True)
class AudioStream:
    codec: str
    channels: int | None
    start_time: float | None
    duration: float | None


@dataclass(frozen=True)
class SourceMedia:
    width: int  # as displayed (rotation applied by FFmpeg when decoding)
    height: int
    frame_rate: str  # exact rational, e.g. "30000/1001"
    fps: float
    pix_fmt: str
    bit_depth: int
    color: ColorSpec
    color_was_tagged: bool
    video_start: float
    duration: float | None
    audio: tuple[AudioStream, ...]

    @property
    def even_size(self) -> tuple[int, int]:
        return self.width - self.width % 2, self.height - self.height % 2


def _tag(stream: dict[str, Any], key: str) -> str | None:
    value = str(stream.get(key) or "").lower()
    return None if value in _UNKNOWN else value


def _float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def resolve_color(stream: dict[str, Any], width: int, height: int) -> tuple[ColorSpec, bool]:
    """The source's colour definition, with the usual player convention filling any gap: anything above 576 lines or
    1024 pixels wide is BT.709, smaller is BT.601 (what mpv and browsers assume). The flag says whether the file
    actually carried a matrix tag."""
    transfer_tag = _tag(stream, "color_transfer")
    if transfer_tag in HDR_TRANSFERS:
        raise ProcessingError(
            "This video uses HDR (high dynamic range), which can't be restored faithfully yet. "
            "Export an SDR version and upload that."
        )
    matrix_tag = _tag(stream, "color_space")
    tagged = matrix_tag in MATRICES
    matrix = matrix_tag if tagged else ("bt709" if (height > 576 or width > 1024) else "smpte170m")
    assert matrix is not None
    primaries = _tag(stream, "color_primaries")
    transfer = transfer_tag
    return (
        ColorSpec(
            matrix=matrix,
            primaries=primaries if primaries in {"bt709", "smpte170m", "bt470bg", "bt2020"} else _PRIMARIES_FOR[matrix],
            transfer=transfer if transfer in {"bt709", "smpte170m", "bt2020-10", "iec61966-2-1"} else _TRANSFER_FOR[matrix],
            range="pc" if _tag(stream, "color_range") == "pc" or str(stream.get("pix_fmt", "")).startswith("yuvj") else "tv",
        ),
        tagged,
    )


def _bit_depth(stream: dict[str, Any]) -> int:
    depth = _float(stream.get("bits_per_raw_sample"))
    if depth:
        return int(depth)
    pix_fmt = str(stream.get("pix_fmt", ""))
    for bits in (16, 14, 12, 10, 9):
        if f"{bits}le" in pix_fmt or f"{bits}be" in pix_fmt:
            return bits
    return 8


def inspect_source(path: Path) -> SourceMedia:
    try:
        probe = run_ffprobe(path)
    except VideoRejected as rejected:
        raise ProcessingError(rejected.message) from None
    streams = [s for s in probe.get("streams") or [] if isinstance(s, dict)]
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic") != 1),
        None,
    )
    if video is None:
        raise ProcessingError("No video track was found in this file.")
    width, height = int(video["width"]), int(video["height"])
    if rotation_degrees(video) in (90, 270):
        width, height = height, width
    frame_rate = str(video.get("r_frame_rate") or video.get("avg_frame_rate") or "")
    numerator, _, denominator = frame_rate.partition("/")
    try:
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        raise ProcessingError("This video's frame rate couldn't be read.") from None
    if fps <= 0:
        raise ProcessingError("This video's frame rate couldn't be read.")
    color, tagged = resolve_color(video, width, height)
    audio = tuple(
        AudioStream(
            codec=str(s.get("codec_name", "")).lower(),
            channels=int(s["channels"]) if isinstance(s.get("channels"), int) else None,
            start_time=_float(s.get("start_time")),
            duration=_float(s.get("duration")),
        )
        for s in streams
        if s.get("codec_type") == "audio"
    )
    return SourceMedia(
        width=width,
        height=height,
        frame_rate=frame_rate,
        fps=fps,
        pix_fmt=str(video.get("pix_fmt", "")),
        bit_depth=_bit_depth(video),
        color=color,
        color_was_tagged=tagged,
        video_start=_float(video.get("start_time")) or 0.0,
        duration=_float(video.get("duration")) or _float((probe.get("format") or {}).get("duration")),
        audio=audio,
    )
