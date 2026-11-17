"""Automatic removal: the exact method when the mark allows it, otherwise the next best, and always said out loud.

For each video, in this order:

1. **Overlay inversion** (`overlay.py`): a fixed, semi-transparent, single-colour logo over a picture that changes is
   measured from the clip and taken off exactly. Real pixels come back, on a still camera too.
2. **Reconstruction from other frames** (`temporal.py`): wherever the camera or the mark moves, the hidden pixels are
   copied from the frames that show them. Whatever no frame shows goes to LaMa if one is configured, otherwise the
   render stops.

The choice is made once, from the first frames, before anything is written. Nothing is hidden: the render's notes
say which method was used, and when the exact one was passed over, why.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from itertools import chain, islice, tee

from app.ai.base import (
    DeviceInfo,
    Frame,
    InpaintingEngine,
    InpaintingError,
    InpaintingOptions,
    Mask,
    paired,
)
from app.ai.image import ImageInpaintingEngine
from app.ai.overlay import DEFAULT_MEMORY_MB, OverlayInversionEngine, measure
from app.ai.temporal import TemporalReconstructionEngine

logger = logging.getLogger(__name__)


class AutoRemovalEngine(InpaintingEngine):
    provider = "auto"
    name = "Automatic removal"
    is_development_only = False
    sends_frames_to_third_party = False

    def __init__(
        self,
        fallback: ImageInpaintingEngine | None = None,
        fallback_label: str = "",
        memory_mb: int = DEFAULT_MEMORY_MB,
        search_memory_mb: int | None = None,
    ) -> None:
        self._overlay = OverlayInversionEngine(memory_mb=memory_mb)
        self._reconstruct = TemporalReconstructionEngine(
            fallback=fallback, fallback_label=fallback_label, **({"memory_mb": search_memory_mb} if search_memory_mb else {})
        )
        self._chosen: InpaintingEngine | None = None
        self._reason = ""
        self._surveyed: list[tuple[int, Frame, Mask]] | None = None

    def is_available(self) -> bool:
        return self._reconstruct.is_available()

    def unavailable_reason(self) -> str | None:
        return self._reconstruct.unavailable_reason()

    def device(self) -> DeviceInfo | None:
        return self._reconstruct.device()

    def expectation(self) -> str:
        return (
            "A fixed semi-transparent logo is measured and taken off exactly. Otherwise the marked area is rebuilt from "
            "other frames of your video wherever the camera or the mark moves. " + self._reconstruct.expectation().split(". ", 1)[-1]
        )

    def prepare(self) -> None:
        self._reconstruct.prepare()

    def survey_size(self, frame_nbytes: int) -> int:
        return OverlayInversionEngine.sample_frames_for(frame_nbytes)

    def survey(self, sample: list[tuple[int, Frame, Mask]]) -> None:
        self._surveyed = sample

    def notes(self) -> list[str]:
        chosen = self._chosen
        if chosen is None:
            return []
        lead = [f"The exact overlay method couldn't be used ({self._reason}) so the area was rebuilt from other frames instead."] if chosen is self._reconstruct and self._reason else []
        return lead + chosen.notes()

    def process(self, frames: Iterable[Frame], masks: Iterable[Mask], options: InpaintingOptions) -> Iterator[Frame]:
        self.require_available()
        self._chosen, self._reason = None, ""
        pairs = paired(frames, masks)
        first = next(pairs, None)
        if first is None:
            return
        surveyed, self._surveyed = self._surveyed, None  # held only until the overlay has been measured
        if surveyed:
            sample, taken = surveyed, [first]  # spread over the whole clip; the stream below still starts at frame 0
        else:
            sample = taken = [first, *islice(pairs, self._overlay.sample_size(first[1]) - 1)]
        try:
            model = measure(sample, options, self._overlay.memory_mb * 1024 * 1024)
        except InpaintingError as reason:
            self._reason = str(reason).rstrip(".")
            logger.info("overlay removal not used", extra={"reason": self._reason})
            self._chosen = self._reconstruct
            for_frames, for_masks = tee(chain(taken, pairs))  # the engine reads frames and masks side by side
            yield from self._reconstruct.process((f for _, f, _ in for_frames), (m for _, _, m in for_masks), options)
            return
        self._chosen = self._overlay
        del sample
        yield from self._overlay.run(taken, pairs, options, model)

    def close(self) -> None:
        self._reconstruct.close()

