"""The container formats the app can write, in one place: which codecs each holds, how FFmpeg is told to write it,
which audio can be copied into it untouched, and what to convert to when it can't.

Reading is separate (see probe.py): a file's real container is judged from its first bytes, never from its name.
"""

from __future__ import annotations

CONTAINERS = ("mp4", "mov", "webm", "mkv")

# codecs a container can hold, first = the default
CONTAINER_VIDEO_CODECS: dict[str, tuple[str, ...]] = {
    "mp4": ("h264", "hevc", "av1"),
    "mov": ("h264", "hevc"),
    "webm": ("vp9", "av1"),
    "mkv": ("h264", "hevc", "vp9", "av1"),
}

CONTENT_TYPES = {"mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm", "mkv": "video/x-matroska"}

# audio codecs each container carries unchanged (as ffprobe names them)
COPYABLE_AUDIO: dict[str, frozenset[str]] = {
    "mp4": frozenset({"aac", "mp3", "ac3", "eac3", "opus", "flac", "alac"}),
    "mov": frozenset({"aac", "mp3", "ac3", "eac3", "alac"}),
    "webm": frozenset({"opus", "vorbis"}),
    "mkv": frozenset({"aac", "mp3", "ac3", "eac3", "opus", "vorbis", "flac", "alac"}),
}

# what audio is converted to once when the original codec can't go in: (FFmpeg encoder, bitrate, name shown to the person)
REENCODE_AUDIO: dict[str, tuple[str, str, str]] = {
    "mp4": ("aac", "192k", "aac"),
    "mov": ("aac", "192k", "aac"),
    "webm": ("libopus", "160k", "opus"),
    "mkv": ("libopus", "160k", "opus"),
}


def container_args(container: str) -> list[str]:
    """FFmpeg output options that select the container (and, for MP4/MOV, put the index at the front for streaming)."""
    if container == "mp4":
        return ["-movflags", "+faststart", "-f", "mp4"]
    if container == "mov":
        return ["-movflags", "+faststart", "-f", "mov"]
    if container == "webm":
        return ["-f", "webm"]
    if container == "mkv":
        return ["-f", "matroska"]
    raise ValueError(f"container must be one of {CONTAINERS}")


def file_extension(container: str) -> str:
    """The file extension for a container; anything unknown falls back to mp4."""
    return container if container in CONTAINERS else "mp4"
