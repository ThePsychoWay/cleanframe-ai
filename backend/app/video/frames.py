"""Decoding the original into frames, and encoding frames into a video.

Both sides go through FFmpeg with raw BGR frames on a pipe, so the pipeline only ever holds a frame or two in
memory, whatever the video's length.

Decoding follows exactly the rules the editor's preview was made with (constant frame rate, orientation applied), so
frame N here is frame N in the editor and in every saved mask. Colour is converted with the source's own matrix and
range, and the output is encoded with the same matrix and range and tagged with them, so colours stay put.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np

from app.config import get_settings
from app.video.encoders import CODECS, CONTAINER_CODECS, Encoder, choose_encoder
from app.video.ffmpeg import FFmpegProcess, ProcessingError
from app.video.probe import demuxer_args
from app.video.source import ColorSpec, SourceMedia

logger = logging.getLogger(__name__)

QUALITIES = ("fast", "balanced", "high")
CONTAINERS = ("mp4", "webm")
OUTPUT_HEIGHTS = (None, 2160, 1440, 1080, 720)  # None keeps the original size; the others are caps, never enlargements


class FrameReader:
    """Yields the original's frames as HxWx3 uint8 BGR arrays, in order."""

    def __init__(self, path: Path, media: SourceMedia) -> None:
        self.width, self.height = media.even_size  # yuv420p needs even sizes, so an odd edge is trimmed by one pixel
        self._path = path
        self._media = media
        self._ffmpeg: FFmpegProcess | None = None
        self.had_decode_errors = False  # FFmpeg reported damage but carried on (it exits 0 for a truncated file)

    def _filters(self) -> str:
        media = self._media
        parts = []
        if (self.width, self.height) != (media.width, media.height):
            parts.append(f"crop={self.width}:{self.height}:0:0")
        parts.append(f"scale=in_color_matrix={media.color.scale_matrix}:in_range={media.color.range}:out_range=pc")
        parts.append("format=bgr24")
        return ",".join(parts)

    def __enter__(self) -> FrameReader:
        self._ffmpeg = FFmpegProcess(
            [
                *demuxer_args(self._path),
                "-i", f"file:{self._path}",
                "-map", "0:v:0", "-an", "-sn", "-dn",
                "-fps_mode", "cfr",  # the same rule the preview used, so frame N matches
                "-vf", self._filters(),
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "pipe:1",
            ],
            stdout=True,
        )  # fmt: skip
        return self

    def __exit__(self, *exc: object) -> None:
        if self._ffmpeg is not None:
            self._ffmpeg.close()
            self._ffmpeg = None

    def frames(self) -> Iterator[np.ndarray]:
        assert self._ffmpeg is not None, "use FrameReader as a context manager"
        size = self.width * self.height * 3
        stream = self._ffmpeg.stdout
        while True:
            frame = np.empty((self.height, self.width, 3), np.uint8)
            view = memoryview(frame).cast("B")
            got = 0
            while got < size:
                count = stream.readinto(view[got:])
                if not count:
                    break
                got += count
            if got == 0:
                break
            if got < size:
                logger.warning("decoder stopped mid-frame: %s", self._ffmpeg.error_tail())
                raise ProcessingError("The original video couldn't be read to the end. It may be damaged.")
            yield frame
        code = self._ffmpeg.process.wait()
        tail = self._ffmpeg.error_tail()
        if tail:
            self.had_decode_errors = True
            logger.warning("decoder reported problems: %s", tail)
        if code != 0:
            logger.warning("decoder failed (%s)", code)
            raise ProcessingError("The original video couldn't be read to the end. It may be damaged.")


@dataclass(frozen=True)
class EncodeSettings:
    container: str = "mp4"
    quality: str = "balanced"
    output_height: int | None = None  # a cap on the shorter side (the resolution class): never enlarges
    codec: str | None = None  # None = H.264 in MP4, VP9 in WebM
    encoder: str | None = None  # auto | cpu | nvenc; None = the VIDEO_ENCODER setting

    def __post_init__(self) -> None:
        if self.container not in CONTAINERS:
            raise ValueError(f"container must be one of {CONTAINERS}")
        if self.quality not in QUALITIES:
            raise ValueError(f"quality must be one of {QUALITIES}")
        if self.output_height not in OUTPUT_HEIGHTS:
            raise ValueError(f"output_height must be one of {OUTPUT_HEIGHTS}")
        if self.codec is not None and (self.codec not in CODECS or self.codec not in CONTAINER_CODECS[self.container]):
            raise ValueError(f"codec {self.codec!r} can't be used in {self.container}")

    def choose(self) -> Encoder:
        return choose_encoder(self.container, self.codec, self.quality, self.encoder or get_settings().video_encoder)


def output_size(width: int, height: int, cap: int | None) -> tuple[int, int]:
    """The encoded size: the original (even), or scaled down so its *shorter* side is `cap` pixels, with the same
    aspect. The shorter side is what names a resolution (1920x1080 and 1080x1920 are both "1080p"). Never enlarges."""
    width, height = width - width % 2, height - height % 2
    short = min(width, height)
    if cap is None or short <= cap:
        return width, height
    factor = cap / short
    return max(2, round(width * factor / 2) * 2), max(2, round(height * factor / 2) * 2)


class FrameWriter:
    """Encodes frames (no audio) to `destination`."""

    def __init__(
        self,
        destination: Path,
        *,
        width: int,
        height: int,
        frame_rate: str,
        color: ColorSpec,
        settings: EncodeSettings,
    ) -> None:
        self.width, self.height = width, height
        self.out_width, self.out_height = output_size(width, height, settings.output_height)
        self._destination = destination
        self._frame_bytes = width * height * 3
        self._written = 0

        scale = (
            f"scale={self.out_width}:{self.out_height}:flags=lanczos+accurate_rnd+full_chroma_int:"
            f"in_range=pc:out_color_matrix={color.scale_matrix}:out_range={color.range},format=yuv420p"
        )
        tags = ["-colorspace", color.matrix, "-color_primaries", color.primaries, "-color_trc", color.transfer, "-color_range", color.range]
        self.encoder = settings.choose()
        codec = list(self.encoder.args)
        container = ["-f", "webm"] if settings.container == "webm" else ["-movflags", "+faststart", "-f", "mp4"]
        self._ffmpeg = FFmpegProcess(
            [
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", frame_rate,
                "-protocol_whitelist", "file,pipe", "-i", "pipe:0",  # stdin is ours; still no network protocols
                "-vf", scale, "-fps_mode", "passthrough", "-an",
                *codec, *tags, *container,
                f"file:{destination}",
            ],
            stdin=True,
        )  # fmt: skip

    @property
    def frames_written(self) -> int:
        return self._written

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ProcessingError("A reconstructed frame had the wrong size.")
        try:
            self._ffmpeg.stdin.write(np.ascontiguousarray(frame).data)
        except (BrokenPipeError, OSError):
            logger.warning("encoder stopped: %s", self._ffmpeg.error_tail())
            raise ProcessingError("The video couldn't be encoded.") from None
        self._written += 1

    def finish(self) -> None:
        """Flush and wait for the encoder. Raises if it reported a problem."""
        try:
            self._ffmpeg.stdin.close()
        except OSError:
            pass
        code = self._ffmpeg.process.wait()
        if code != 0:
            logger.warning("encoder failed (%s): %s", code, self._ffmpeg.error_tail())
            raise ProcessingError("The video couldn't be encoded.")

    def abort(self) -> None:
        self._ffmpeg.close()

    def close(self) -> None:
        self._ffmpeg.close()

    def __enter__(self) -> FrameWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()
