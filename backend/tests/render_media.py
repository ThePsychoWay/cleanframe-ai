"""Small real media files for the render tests. Everything is made with FFmpeg, so the tests exercise the same
decoders, encoders and muxers the app uses."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg(*args: str, timeout: int = 180) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, timeout=timeout)


def probe(path: Path, *args: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", *args, str(path)], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    return json.loads(out)


def stream_of(path: Path, kind: str) -> dict:
    data = probe(path, "-show_streams", "-show_format")
    return next(s for s in data["streams"] if s["codec_type"] == kind)


def first_packet_time(path: Path, kind: str) -> float:
    """Presentation time of the first packet of a stream: what a player uses to line sound up with picture."""
    data = probe(path, "-select_streams", f"{kind[0]}:0", "-show_entries", "packet=pts_time", "-read_intervals", "%+#1")
    return float(data["packets"][0]["pts_time"])


def pcm(path: Path) -> bytes:
    """Decoded audio of the first audio stream, as raw samples."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-"], capture_output=True, check=True, timeout=120
    ).stdout


def make_source(
    path: Path,
    *,
    size: str = "192x108",
    frames: int = 30,
    rate: str = "30",
    audio: str | None = None,  # aac | mp3 | libopus | pcm_s16le | libvorbis
    video: str = "libx264",
    pix_fmt: str = "yuv420p",
    extra_video: tuple[str, ...] = (),
    audio_delay: float = 0.0,
) -> Path:
    """A moving test pattern with an optional tone. `audio_delay` shifts the tone later (>0) or earlier (<0)."""
    fps = eval_rate(rate)
    seconds = frames / fps
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds:.4f}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds:.4f}"]
        if audio_delay > 0:
            args += ["-af", f"adelay={int(audio_delay * 1000)}:all=1"]
    args += ["-c:v", video, "-pix_fmt", pix_fmt, *extra_video]
    if audio:
        args += ["-c:a", audio]
    ffmpeg(*args, str(path))
    return path


def eval_rate(rate: str) -> float:
    numerator, _, denominator = rate.partition("/")
    return float(numerator) / float(denominator or 1)


def add_audio_start(source: Path, destination: Path, seconds: float) -> Path:
    """Copy `source`, shifting its audio's start by `seconds` without touching the picture (negative = earlier)."""
    ffmpeg("-i", str(source), "-itsoffset", f"{seconds}", "-i", str(source), "-map", "0:v", "-map", "1:a", "-c", "copy", str(destination))
    return destination


def child_processes() -> list[int]:
    """PIDs of this process's children that are still alive (zombies excluded)."""
    mine = os.getpid()
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        state, parent = fields[0], int(fields[1])
        if parent == mine and state != "Z":
            found.append(int(entry.name))
    return found


def solid(width: int, height: int, colour: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty((height, width, 3), np.uint8)
    frame[:] = colour
    return frame
