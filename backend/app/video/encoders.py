"""Which video encoder runs, and with what settings.

Every encode is quality-based (constant quality, not a fixed bitrate), so a busy 4K picture gets the bits it needs
and a flat one doesn't waste them. NVIDIA's NVENC is used only when it is both asked for (VIDEO_ENCODER=auto or nvenc)
and proven to work on this machine by encoding a few test frames: having `h264_nvenc` listed in FFmpeg only means it
was compiled in, not that a GPU and driver are there. When the proof fails, the CPU encoder runs and the result says so.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass

from app.video.containers import CONTAINER_VIDEO_CODECS
from app.video.ffmpeg import ffmpeg_command

logger = logging.getLogger(__name__)

CODECS = ("h264", "hevc", "vp9", "av1")
QUALITIES = ("fast", "balanced", "high")
CONTAINER_CODECS = CONTAINER_VIDEO_CODECS
DEFAULT_CODEC = {container: codecs[0] for container, codecs in CONTAINER_VIDEO_CODECS.items()}
LABELS = {"h264": "H.264", "hevc": "H.265 (HEVC)", "vp9": "VP9", "av1": "AV1"}

# quality -> (preset, constant-quality value); lower value = better picture and a bigger file.
_X264 = {"fast": ("veryfast", 23), "balanced": ("fast", 20), "high": ("medium", 17)}
_X265 = {"fast": ("veryfast", 27), "balanced": ("fast", 24), "high": ("medium", 21)}
_VP9 = {"fast": (34, 5), "balanced": (31, 3), "high": (28, 1)}  # (crf, cpu-used)
_SVTAV1 = {"fast": (35, 10), "balanced": (30, 8), "high": (26, 6)}  # (crf, preset): a lower crf is a better picture
_LIBAOM = {"fast": (34, 7), "balanced": (30, 6), "high": (26, 4)}  # (crf, cpu-used)
_NVENC_PRESET = {"fast": "p3", "balanced": "p5", "high": "p7"}
_NVENC_CQ = {"h264": {"fast": 25, "balanced": 21, "high": 18}, "hevc": {"fast": 28, "balanced": 24, "high": 21}}

_PROBE_TIMEOUT_SECONDS = 30
_nvenc_cache: dict[str, bool] = {}
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class Encoder:
    codec: str  # h264 | hevc | vp9 | av1
    name: str  # FFmpeg's encoder name
    hardware: bool
    args: tuple[str, ...]

    @property
    def description(self) -> str:
        return f"{LABELS[self.codec]} ({'NVIDIA NVENC' if self.hardware else 'CPU'})"


def nvenc_works(codec: str) -> bool:
    """True only if FFmpeg can really encode `codec` with NVENC here: a short test encode is run once and remembered."""
    if codec not in ("h264", "hevc"):
        return False
    with _cache_lock:
        if codec in _nvenc_cache:
            return _nvenc_cache[codec]
    name = f"{codec}_nvenc"
    command = ffmpeg_command(
        ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=0.3", "-pix_fmt", "yuv420p", "-c:v", name, "-f", "null", "-"]
    )
    try:
        works = subprocess.run(command, capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        works = False
    logger.info("NVENC check", extra={"encoder": name, "works": works})
    with _cache_lock:
        _nvenc_cache[codec] = works
    return works


def forget_hardware_checks() -> None:
    with _cache_lock:
        _nvenc_cache.clear()


def choose_encoder(container: str, codec: str | None, quality: str, preference: str = "auto") -> Encoder:
    """The encoder for this container, codec and quality. `preference`: auto (NVENC if it works), cpu, or nvenc (still
    falls back to the CPU if NVENC doesn't work: the output is the same kind of file either way)."""
    if container not in CONTAINER_CODECS:
        raise ValueError(f"container must be one of {tuple(CONTAINER_CODECS)}")
    codec = codec or DEFAULT_CODEC[container]
    if codec not in CONTAINER_CODECS[container]:
        raise ValueError(f"{LABELS.get(codec, codec)} can't be stored in {container.upper()}.")
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {QUALITIES}")

    if codec == "av1":
        listed = _listed_encoders()
        if "libsvtav1" in listed:
            crf, preset = _SVTAV1[quality]
            return Encoder("av1", "libsvtav1", False, ("-c:v", "libsvtav1", "-crf", str(crf), "-preset", str(preset)))
        if "libaom-av1" in listed:
            crf, cpu = _LIBAOM[quality]
            return Encoder("av1", "libaom-av1", False, ("-c:v", "libaom-av1", "-b:v", "0", "-crf", str(crf), "-cpu-used", str(cpu), "-row-mt", "1"))
        raise ValueError("This server's FFmpeg has no AV1 encoder.")

    if codec == "vp9":
        crf, cpu = _VP9[quality]
        return Encoder("vp9", "libvpx-vp9", False, ("-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf), "-row-mt", "1", "-deadline", "good", "-cpu-used", str(cpu)))

    if preference in ("auto", "nvenc") and nvenc_works(codec):
        args = ["-c:v", f"{codec}_nvenc", "-preset", _NVENC_PRESET[quality], "-rc", "vbr", "-cq", str(_NVENC_CQ[codec][quality]), "-b:v", "0", "-spatial-aq", "1"]
        if codec == "hevc":
            args += ["-tag:v", "hvc1"]
        return Encoder(codec, f"{codec}_nvenc", True, tuple(args))

    if codec == "hevc":
        preset, crf = _X265[quality]
        return Encoder("hevc", "libx265", False, ("-c:v", "libx265", "-preset", preset, "-crf", str(crf), "-tag:v", "hvc1", "-x265-params", "log-level=error"))
    preset, crf = _X264[quality]
    return Encoder("h264", "libx264", False, ("-c:v", "libx264", "-preset", preset, "-crf", str(crf)))


_listed: set[str] | None = None


def _listed_encoders() -> set[str]:
    global _listed
    with _cache_lock:
        if _listed is not None:
            return _listed
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=_PROBE_TIMEOUT_SECONDS, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    names = {parts[1] for parts in (line.split() for line in out.splitlines()) if len(parts) > 2 and parts[0].startswith("V")}
    with _cache_lock:
        _listed = names
    return names


def codec_available(codec: str) -> bool:
    """Whether FFmpeg on this machine has an encoder for the codec (a CPU one, or NVENC that passed its test)."""
    listed = _listed_encoders()
    if codec == "av1":
        return "libsvtav1" in listed or "libaom-av1" in listed
    cpu = {"h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9"}[codec]
    return cpu in listed or nvenc_works(codec)
