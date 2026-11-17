"""One cheap decode of the whole timeline at low resolution: the only pixels the detector looks at."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.video.ffmpeg import FFmpegProcess, ProcessingError
from app.video.probe import demuxer_args
from app.video.source import SourceMedia

logger = logging.getLogger(__name__)

SURVEY_WIDTH = 384  # wide enough to read a logo's shape, small enough that a minute of video fits in RAM
MAX_SAMPLES = 1200  # 40 s at 30 fps is looked at frame by frame; longer videos are sampled evenly


@dataclass(frozen=True)
class Survey:
    grey: np.ndarray  # N x H x W float32, 0..255
    colour: np.ndarray  # N x H x W x 3 uint8 (BGR); kept for contrast and colour checks
    frames: np.ndarray  # source frame number of each sample, int
    scale: float  # survey pixels per source pixel
    fps: float
    source_frames: int

    @property
    def count(self) -> int:
        return int(self.grey.shape[0])

    @property
    def height(self) -> int:
        return int(self.grey.shape[1])

    @property
    def width(self) -> int:
        return int(self.grey.shape[2])

    @property
    def stride(self) -> int:
        return int(self.frames[1] - self.frames[0]) if self.count > 1 else 1


def survey_size(media: SourceMedia) -> tuple[int, int]:
    width = min(SURVEY_WIDTH, media.width)
    height = max(2, round(media.height * width / media.width))
    return width - width % 2, height - height % 2


def _read_exactly(stream: Any, size: int) -> bytes:
    """A pipe may hand back less than asked for; a whole frame, or nothing at the end of the stream."""
    chunks: list[bytes] = []
    got = 0
    while got < size:
        chunk = stream.read(size - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    if 0 < got < size:
        raise ProcessingError("The video couldn't be read to the end. It may be damaged.")
    return b"".join(chunks)


def read_survey(
    path: Path, media: SourceMedia, frame_count: int, *, progress: Callable[[int, int], None] | None = None
) -> Survey:
    """Every frame (or every n-th when there are more than MAX_SAMPLES) at survey size, in the editor's frame numbering."""
    width, height = survey_size(media)
    even_w, even_h = media.even_size
    stride = max(1, -(-frame_count // MAX_SAMPLES))
    filters = []
    if (even_w, even_h) != (media.width, media.height):
        filters.append(f"crop={even_w}:{even_h}:0:0")
    filters += [
        f"scale={width}:{height}:flags=area:in_color_matrix={media.color.scale_matrix}:in_range={media.color.range}:out_range=pc",
        "format=bgr24",
    ]
    ffmpeg = FFmpegProcess(
        [
            *demuxer_args(path), "-i", f"file:{path}", "-map", "0:v:0", "-an", "-sn", "-dn",
            "-fps_mode", "cfr", "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ],
        stdout=True,
    )  # fmt: skip
    size = width * height * 3
    kept: list[np.ndarray] = []
    numbers: list[int] = []
    index = 0
    try:
        while True:
            buffer = _read_exactly(ffmpeg.stdout, size)
            if not buffer:
                break
            if index % stride == 0:
                kept.append(np.frombuffer(buffer, np.uint8).reshape(height, width, 3).copy())
                numbers.append(index)
            index += 1
            if progress and index % 60 == 0:
                progress(index, frame_count)
    finally:
        ffmpeg.close()
    if not kept:
        raise ProcessingError("The video has no frames to look at.")
    colour = np.stack(kept)
    grey = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in kept]).astype(np.float32)
    return Survey(
        grey=grey, colour=colour, frames=np.asarray(numbers, dtype=np.int64), scale=width / media.width, fps=media.fps, source_frames=index
    )
