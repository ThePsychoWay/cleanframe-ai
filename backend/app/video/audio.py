"""Putting the original audio back.

The picture is rebuilt separately and has no sound. Here the original soundtrack is combined with it. It is copied
bit for bit (no re-encoding, no quality loss) whenever the target container can hold it; otherwise it is converted
once (AAC for MP4 and MOV, Opus for WebM and MKV), and the result says so. The soundtrack keeps its timing relative to the
picture, including any start offset the original had: both inputs keep their own timestamps and are
joined without shifting either (checked against audio that starts late and audio that starts early).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.video.containers import COPYABLE_AUDIO, REENCODE_AUDIO, container_args
from app.video.ffmpeg import run_ffmpeg
from app.video.probe import demuxer_args
from app.video.source import SourceMedia

MUX_TIMEOUT_SECONDS = 600
COPYABLE = COPYABLE_AUDIO


class AudioAction(StrEnum):
    NONE = "none"  # the original had no audio
    REMOVED = "removed"  # the person chose to remove it
    COPIED = "copied"  # kept exactly as it was
    CONVERTED = "converted"  # kept, converted once because the output format can't hold the original codec


@dataclass(frozen=True)
class AudioResult:
    action: AudioAction
    codec: str | None  # codec of the audio in the output


def plan_audio(media: SourceMedia, container: str, keep_audio: bool) -> tuple[AudioAction, str | None]:
    if not media.audio:
        return AudioAction.NONE, None
    if not keep_audio:
        return AudioAction.REMOVED, None
    if all(stream.codec in COPYABLE[container] for stream in media.audio):
        return AudioAction.COPIED, media.audio[0].codec
    return AudioAction.CONVERTED, REENCODE_AUDIO[container][2]


def mux_audio(video_only: Path, source: Path, destination: Path, media: SourceMedia, *, container: str, keep_audio: bool) -> AudioResult:
    action, codec = plan_audio(media, container, keep_audio)
    if action in (AudioAction.NONE, AudioAction.REMOVED):
        # Nothing to add: the picture file is the result.
        run_ffmpeg(["-i", f"file:{video_only}", "-map", "0:v:0", "-c", "copy", *_container_args(container), f"file:{destination}"], timeout=MUX_TIMEOUT_SECONDS)
        return AudioResult(action, None)

    if action is AudioAction.COPIED:
        audio_args = ["-c:a", "copy"]
    else:
        encoder, bitrate, _ = REENCODE_AUDIO[container]
        audio_args = ["-c:a", encoder, "-b:a", bitrate]
    run_ffmpeg(
        [
            "-i", f"file:{video_only}",
            *demuxer_args(source),
            "-i", f"file:{source}",
            "-map", "0:v:0", "-map", "1:a",
            "-c:v", "copy", *audio_args,
            "-map_metadata", "-1",  # container tags (title, location, dates) of the original aren't carried over
            *_container_args(container),
            f"file:{destination}",
        ],
        timeout=MUX_TIMEOUT_SECONDS,
        failure="The original audio couldn't be added to the restored video.",
    )  # fmt: skip
    return AudioResult(action, codec)


def _container_args(container: str) -> list[str]:
    return container_args(container)


__all__ = ["AudioAction", "AudioResult", "mux_audio", "plan_audio"]
