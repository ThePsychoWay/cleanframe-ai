"""Engines that reconstruct one image at a time (a local model, a remote API, the development engine).

Subclasses implement `inpaint_image` for a single crop. This base does everything else the same way for all
of them: cropping to the marked area plus context, colour matching, blending back, and smoothing the result
over time. Frames with nothing marked are returned untouched.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable, Iterator

import numpy as np

from app.ai.base import (
    Frame,
    InpaintingEngine,
    InpaintingError,
    InpaintingOptions,
    Mask,
    paired,
)
from app.ai.consistency import TemporalStabilizer, match_colour
from app.ai.roi import blend_into, plan_regions, prepare_mask

logger = logging.getLogger(__name__)


def check_result(result: object, expected_shape: tuple[int, ...]) -> Frame:
    if not isinstance(result, np.ndarray) or result.dtype != np.uint8 or result.shape != expected_shape:
        raise InpaintingError("The inpainting engine returned an image that doesn't match its input.")
    return result


class ImageInpaintingEngine(InpaintingEngine):
    # Some models need crop sizes to be a multiple of this (8, 16...). The crop is grown to fit.
    input_multiple: int = 1

    @abstractmethod
    def inpaint_image(self, crop: Frame, mask: Mask, options: InpaintingOptions) -> Frame:
        """Return `crop` with the pixels where `mask` is 255 reconstructed. Same size, uint8 BGR."""

    def process(
        self,
        frames: Iterable[Frame],
        masks: Iterable[Mask],
        options: InpaintingOptions,
    ) -> Iterator[Frame]:
        self.require_available()
        stabilizer = TemporalStabilizer() if options.temporal_smoothing else None
        try:
            for index, frame, mask in paired(frames, masks):
                prepared = prepare_mask(mask, options)
                if prepared is None:
                    if stabilizer is not None:
                        stabilizer.skip(frame)
                    yield frame.copy()
                    continue
                output = frame.copy()
                try:
                    for roi in plan_regions(prepared, options, multiple=self.input_multiple):
                        crop = np.ascontiguousarray(frame[roi.slices])
                        crop_mask = np.ascontiguousarray(prepared.grown[roi.slices])
                        filled = check_result(self.inpaint_image(crop, crop_mask, options), crop.shape)
                        blend_into(output, match_colour(filled, crop, crop_mask), prepared.alpha, roi)
                except InpaintingError:
                    raise
                except Exception as error:
                    logger.exception("inpainting failed on frame %d (%s)", index, self.provider)
                    raise InpaintingError(f"Reconstruction failed on frame {index + 1}.") from error
                if stabilizer is not None:
                    output = stabilizer.stabilize(frame, prepared, output)
                yield output
        finally:
            self.close()

    def close(self) -> None:
        """Release anything held between frames (connections, GPU buffers). Called when processing ends."""
