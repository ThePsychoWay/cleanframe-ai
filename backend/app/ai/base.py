"""Inpainting abstraction.

Every model or provider sits behind InpaintingEngine, so one can be swapped for another without
touching the video pipeline. The contract is deliberately small:

    process(frames, masks, options) -> frames

* A frame is an HxWx3 uint8 BGR array. A mask is an HxW uint8 array, 255 = remove (anything above 127).
* Output has exactly one frame per input frame, in order, with the same size.
* A frame with an empty mask is returned unchanged. Pixels far from the mask are never modified.
  "Far" is `edge_dilate_px + feather_px` (see InpaintingOptions): the engine may reconstruct a thin
  halo around the mask so that the soft edge of a logo does not survive.
* Engines are streaming: they pull frames lazily and yield as soon as a frame is final, so a long video
  never has to fit in memory.
* An engine either does the work or raises. Nothing silently substitutes a different engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# A frame is an HxWx3 uint8 BGR array; a mask is an HxW uint8 array (255 = remove).
Frame = np.ndarray
Mask = np.ndarray

MASK_THRESHOLD = 127  # mask values above this count as "remove"
QUALITIES = ("fast", "balanced", "high")


class InpaintingError(Exception):
    """Processing failed. The message is safe to show to the person using the app."""


class ModelMemoryExhausted(InpaintingError):
    """Even the smallest piece didn't fit in memory. The message names the video's size and what ran out, and is shown as it is."""


class EngineUnavailableError(InpaintingError):
    """The configured engine can't run right now. The message says what is missing."""


@dataclass(frozen=True)
class InpaintingOptions:
    quality: str = "balanced"  # fast | balanced | high
    # Context around the masked area that is given to the model, in pixels.
    context_padding_px: int = 64
    # Grow the area that is reconstructed by this much, to catch the soft edge of a logo.
    edge_dilate_px: int = 2
    # Blend the reconstruction into its surroundings over this many pixels (outside the grown mask).
    feather_px: int = 3
    temporal_smoothing: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quality not in QUALITIES:
            raise ValueError(f"quality must be one of {QUALITIES}, got {self.quality!r}")
        if self.context_padding_px < 0 or self.edge_dilate_px < 0 or self.feather_px < 0:
            raise ValueError("padding, dilation and feathering can't be negative")

    @property
    def reach_px(self) -> int:
        """How far from the mask a pixel can change."""
        return self.edge_dilate_px + self.feather_px


@dataclass(frozen=True)
class DeviceInfo:
    kind: str  # "cuda" | "cpu" | "remote"
    name: str
    note: str  # what the person should expect, in one sentence


@dataclass(frozen=True)
class EngineInfo:
    provider: str  # the AI_PROVIDER value
    name: str
    available: bool
    development_only: bool
    sends_frames_to_third_party: bool
    device: DeviceInfo | None
    reason: str | None  # why it isn't available, or None
    expectation: str  # what processing will be like


class InpaintingEngine(ABC):
    """Reconstructs masked regions of a frame sequence."""

    provider: str = "base"  # the AI_PROVIDER value that selects this engine
    name: str = "base"
    # Dev-only engines must set this so the UI never presents their output as AI restoration.
    is_development_only: bool = False
    # True when image data leaves this server. The UI says so before processing.
    sends_frames_to_third_party: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """True if the engine can run now (weights present, API key set, device usable)."""

    def unavailable_reason(self) -> str | None:
        """Why is_available() is False, in words the person can act on. None when available."""
        return None if self.is_available() else "This inpainting engine isn't set up."

    def device(self) -> DeviceInfo | None:
        return None

    def expectation(self) -> str:
        return ""

    def prepare(self) -> None:
        """Load (or download and verify) whatever the engine needs, so a missing or damaged model fails before any
        video is decoded. Raises EngineUnavailableError. Does nothing by default."""
        return None

    def survey_size(self, frame_nbytes: int) -> int:
        """How many frames, spread over the whole clip, the engine wants to look at before it starts (0: none).
        The pipeline reads them in a first pass and hands them to survey()."""
        return 0

    def survey(self, sample: list[tuple[int, Frame, Mask]]) -> None:
        """Receive the surveyed (frame index, frame, mask) triples, in order, before process() is called."""

    def notes(self) -> list[str]:
        """What the person should know about the work just done, in plain words (for example how much of the
        marked area could be rebuilt from other frames). Read after process() has finished; empty by default."""
        return []

    def info(self) -> EngineInfo:
        available = self.is_available()
        return EngineInfo(
            provider=self.provider,
            name=self.name,
            available=available,
            development_only=self.is_development_only,
            sends_frames_to_third_party=self.sends_frames_to_third_party,
            device=self.device(),
            reason=None if available else self.unavailable_reason(),
            expectation=self.expectation(),
        )

    @abstractmethod
    def process(
        self,
        frames: Iterable[Frame],
        masks: Iterable[Mask],
        options: InpaintingOptions,
    ) -> Iterator[Frame]:
        """Yield reconstructed frames in input order, one per input frame."""

    def require_available(self) -> None:
        if not self.is_available():
            raise EngineUnavailableError(self.unavailable_reason() or "This inpainting engine isn't available.")


def check_frame(frame: Frame, mask: Mask, index: int) -> None:
    """Reject malformed input with a message that points at the frame, instead of failing deep in OpenCV."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise InpaintingError(f"Frame {index} must be an HxWx3 uint8 BGR image.")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise InpaintingError(f"Mask {index} must be an HxW uint8 image.")
    if mask.shape != frame.shape[:2]:
        raise InpaintingError(
            f"Mask {index} is {mask.shape[1]}x{mask.shape[0]} but its frame is {frame.shape[1]}x{frame.shape[0]}."
        )
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise InpaintingError(f"Frame {index} is too small to process.")


def paired(frames: Iterable[Frame], masks: Iterable[Mask]) -> Iterator[tuple[int, Frame, Mask]]:
    """Zip frames with masks, failing loudly if one runs out before the other."""
    frame_iter, mask_iter = iter(frames), iter(masks)
    index = 0
    while True:
        frame = next(frame_iter, None)
        mask = next(mask_iter, None)
        if frame is None and mask is None:
            return
        if frame is None or mask is None:
            more, fewer = ("frames", "masks") if mask is None else ("masks", "frames")
            raise InpaintingError(f"There are more {more} than {fewer}: they ran out at {index}.")
        check_frame(frame, mask, index)
        yield index, frame, mask
        index += 1
