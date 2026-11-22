"""Rendering the restored video: decode -> reconstruct -> encode -> add the original audio -> verify.

    original --FFmpeg--> frames --InpaintingEngine (+ per-frame masks)--> frames --FFmpeg--> picture
    picture + original audio --FFmpeg (stream copy)--> restored video --> verification

Frames stream through one at a time, so memory use doesn't grow with the video's length. The caller supplies the
mask document and (optionally) tracking results, the engine, a scratch directory and a progress callback; storage,
job state and queues belong to the worker that calls this.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.ai.base import InpaintingEngine, InpaintingError, InpaintingOptions
from app.schemas.mask import MaskDocument
from app.video.audio import AudioAction, mux_audio
from app.video.ffmpeg import ProcessingCancelled, ProcessingError, safe_unlink
from app.video.frames import EncodeSettings, FrameReader, FrameWriter
from app.video.masks import MaskSequence
from app.video.source import SourceMedia, inspect_source
from app.video.verify import (
    ArtifactMonitor,
    Expectation,
    VerificationReport,
    verify_output,
)

logger = logging.getLogger(__name__)

# stage names reported to the progress callback
STAGE_MEASURE = "measuring"
STAGE_RECONSTRUCT = "reconstructing"
STAGE_AUDIO = "audio"
STAGE_VERIFY = "verifying"

# progress(stage, done, total) -> keep going? Returning False cancels the render.
Progress = Callable[[str, int, int], bool | None]

SURVEY_OVERSAMPLE = 3  # candidate frames per frame kept
SURVEY_MARGIN = 24  # picture around the mark that is compared when looking for repeated frames
SURVEY_MIN_DIFFERENCE = 2.0  # mean level difference below which two frames count as the same picture


@dataclass(frozen=True)
class RenderSettings:
    quality: str = "balanced"  # fast | balanced | high: encoder effort and the engine's quality level
    container: str = "mp4"  # mp4 | mov | webm | mkv
    output_height: int | None = None  # None = original size; 2160, 1440, 1080 or 720 = scale down so the shorter side is at most that
    codec: str | None = None  # h264 | hevc (MP4), vp9 (WebM); None = H.264 in MP4, VP9 in WebM
    keep_audio: bool = True
    context_padding_px: int = 64
    temporal_smoothing: bool = True

    def encode(self) -> EncodeSettings:
        return EncodeSettings(container=self.container, quality=self.quality, output_height=self.output_height, codec=self.codec)

    def inpainting(self) -> InpaintingOptions:
        return InpaintingOptions(
            quality=self.quality, context_padding_px=self.context_padding_px, temporal_smoothing=self.temporal_smoothing
        )


@dataclass
class RenderResult:
    path: Path
    frames: int
    width: int
    height: int
    fps: float
    container: str
    audio: AudioAction
    audio_codec: str | None
    engine: str  # AI_PROVIDER value
    development_only: bool  # true: the picture was rebuilt by the development engine, not an AI model
    tracking_used: bool
    verification: VerificationReport
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    codec: str = "h264"
    encoder: str = ""  # FFmpeg's name for the encoder that ran
    hardware_encoder: bool = False
    source_width: int = 0  # the original as decoded (even-sized), for "original vs output" displays
    source_height: int = 0


def _notes(media: SourceMedia, size: tuple[int, int], settings: RenderSettings) -> list[str]:
    notes: list[str] = []
    if size != (media.width, media.height) and settings.output_height is None:
        notes.append("The original has an odd width or height, so one pixel row or column was trimmed.")
    if media.bit_depth > 8:
        notes.append(f"The original is {media.bit_depth}-bit; the restored video is 8-bit.")
    return notes


def _survey(
    source: Path,
    media: SourceMedia,
    document: MaskDocument,
    tracking: dict[str, Any] | None,
    estimate: int,
    wanted: int,
    notify: Progress,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Frames spread evenly over the part of the clip where something is marked, for an engine that measures first.

    The masks are known without decoding, so only frames with a mark are candidates. The video is decoded once, front
    to back, keeping the chosen frames, and the pass stops after the last of them."""
    width, height = media.even_size
    masks = MaskSequence(document, tracking, width=width, height=height, frame_count=estimate)
    marked = [index for index in range(estimate) if masks.mask(index).any()]
    if not marked:
        return []
    # A frame where the mark differs from the first marked frame is always kept, however late it comes, so an engine
    # that needs a mark that stays put finds out now and not halfway through the render.
    first_mask = masks.mask(marked[0])
    changed = next((index for index in marked if not np.array_equal(masks.mask(index), first_mask)), None)
    # More candidates than frames wanted: ones that show the same picture under the mark as a frame already kept are
    # skipped (a still card repeated for seconds would otherwise outvote the picture that moves).
    spread = min(len(marked), wanted * SURVEY_OVERSAMPLE)
    candidates = sorted({marked[int(i)] for i in np.linspace(0, len(marked) - 1, spread)} | ({changed} if changed is not None else set()))
    wanted_at = set(candidates)
    kept: list[tuple[int, np.ndarray, np.ndarray]] = []
    seen: list[np.ndarray] = []
    with FrameReader(source, media) as reader:
        for index, frame in enumerate(reader.frames()):
            if index in wanted_at:
                mask = masks.mask(index)
                rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
                window = (
                    slice(max(int(rows[0]) - SURVEY_MARGIN, 0), int(rows[-1]) + SURVEY_MARGIN),
                    slice(max(int(cols[0]) - SURVEY_MARGIN, 0), int(cols[-1]) + SURVEY_MARGIN),
                )
                crop = frame[window].astype(np.int16)
                distinct = len(kept) < wanted and all(c.shape != crop.shape or np.abs(c - crop).mean() >= SURVEY_MIN_DIFFERENCE for c in seen)
                if index == changed or distinct:
                    seen.append(crop)
                    kept.append((index, frame, mask))
                if len(kept) >= wanted and (changed is None or index >= changed):
                    break
            if index % 30 == 0 and notify(STAGE_MEASURE, index, max(estimate, 1)) is False:
                raise ProcessingCancelled
    return kept


def render_restored_video(
    source: Path,
    destination: Path,
    *,
    document: MaskDocument,
    tracking: dict[str, Any] | None,
    engine: InpaintingEngine,
    work_dir: Path,
    settings: RenderSettings | None = None,
    expected_frames: int | None = None,
    progress: Progress | None = None,
) -> RenderResult:
    """Render `source` with the marked areas removed into `destination`.

    `expected_frames` is the project's frame count (the editor's timeline). If the decoded video doesn't have that
    many frames (±1), the masks would be on the wrong frames, so the render stops instead of guessing.
    Raises ProcessingError (message safe to show), InpaintingError, or ProcessingCancelled."""
    settings = settings or RenderSettings()
    started = time.monotonic()
    engine.require_available()  # fail in a second, not after decoding half the video
    engine.prepare()
    media = inspect_source(source)
    if not document.regions:
        raise ProcessingError("There is nothing marked for removal.")

    video_only = work_dir / f"picture.{settings.container}"
    muxed = work_dir / f"restored.{settings.container}"
    for leftover in (video_only, muxed):
        safe_unlink(leftover)

    estimate = expected_frames or max(1, round((media.duration or 0.0) * media.fps))
    monitor = ArtifactMonitor()
    notify = progress or (lambda stage, done, total: True)

    wanted = engine.survey_size(media.even_size[0] * media.even_size[1] * 3)
    if wanted > 0:
        if notify(STAGE_MEASURE, 0, max(estimate, 1)) is False:
            raise ProcessingCancelled
        engine.survey(_survey(source, media, document, tracking, estimate, wanted, notify))

    with FrameReader(source, media) as reader:
        masks = MaskSequence(document, tracking, width=reader.width, height=reader.height, frame_count=estimate)
        writer = FrameWriter(
            video_only,
            width=reader.width,
            height=reader.height,
            frame_rate=media.frame_rate,
            color=media.color,
            settings=settings.encode(),
        )
        try:
            frames_in = {"count": 0}

            def source_frames() -> Iterator[np.ndarray]:
                for frame in reader.frames():
                    frames_in["count"] += 1
                    yield frame

            def mask_frames() -> Iterator[np.ndarray]:
                index = 0
                while index < frames_in["count"]:  # one per frame read, and none after the last
                    yield masks.mask(index)
                    index += 1

            written = 0
            for restored in engine.process(source_frames(), mask_frames(), settings.inpainting()):
                writer.write(restored)
                monitor.update(restored, masks.mask(written))
                written += 1
                if notify(STAGE_RECONSTRUCT, written, max(estimate, written)) is False:
                    raise ProcessingCancelled
            writer.finish()
        except BaseException:
            writer.abort()
            safe_unlink(video_only)
            raise

    if written == 0:
        raise ProcessingError("The video has no frames.")
    if written != frames_in["count"]:
        raise ProcessingError("The reconstruction returned a different number of frames than it was given.")
    if reader.had_decode_errors and written < 0.98 * estimate:
        raise ProcessingError("The original video couldn't be read to the end. It may be damaged.")
    if expected_frames is not None and abs(written - expected_frames) > 1:
        raise ProcessingError(
            f"The video has {written} frames but the editor's timeline has {expected_frames}, so the marks can't be lined up with it."
        )

    if notify(STAGE_AUDIO, 0, 1) is False:
        safe_unlink(video_only)
        raise ProcessingCancelled
    audio = mux_audio(video_only, source, muxed, media, container=settings.container, keep_audio=settings.keep_audio)
    safe_unlink(video_only)

    if notify(STAGE_VERIFY, 0, 1) is False:
        safe_unlink(muxed)
        raise ProcessingCancelled
    out_width, out_height = writer.out_width, writer.out_height
    source_gap = None
    if media.audio and media.audio[0].duration and media.duration:
        source_gap = media.audio[0].duration - media.duration
    report = verify_output(
        muxed,
        Expectation(
            width=out_width, height=out_height, frames=written, fps=media.fps,
            expect_audio=audio.action in (AudioAction.COPIED, AudioAction.CONVERTED),
            source_audio_gap=source_gap if audio.action is not AudioAction.CONVERTED else None,
        ),
        monitor,
    )  # fmt: skip
    destination.parent.mkdir(parents=True, exist_ok=True)
    muxed.replace(destination)

    warnings = _notes(media, (reader.width, reader.height), settings) + report.warnings
    if writer.encoder.hardware:
        warnings.append("The video was encoded on the GPU (NVIDIA NVENC), which is faster than the CPU encoder at a slightly larger file for the same quality.")
    if reader.had_decode_errors:
        warnings.append("The original had minor decoding errors. Check the preview for glitches.")
    warnings.extend(engine.notes())
    if engine.is_development_only:
        warnings.append("This was rebuilt by the development engine, which blurs the area. It is not an AI restoration.")
    result = RenderResult(
        path=destination, frames=written, width=out_width, height=out_height, fps=media.fps, container=settings.container,
        audio=audio.action, audio_codec=audio.codec, engine=engine.provider, development_only=engine.is_development_only,
        tracking_used=masks.uses_tracking, verification=report, warnings=warnings, seconds=time.monotonic() - started,
        codec=writer.encoder.codec, encoder=writer.encoder.name, hardware_encoder=writer.encoder.hardware,
        source_width=reader.width, source_height=reader.height,
    )  # fmt: skip
    logger.info(
        "render finished",
        extra={
            "frames": written,
            "seconds": round(result.seconds, 1),
            "fps": round(written / result.seconds, 2) if result.seconds > 0 else None,
            "audio": audio.action.value,
            "engine": engine.provider,
        },
    )
    return result


__all__ = ["InpaintingError", "ProcessingCancelled", "ProcessingError", "RenderResult", "RenderSettings", "render_restored_video"]
