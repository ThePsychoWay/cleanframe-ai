"""Development engine: classic OpenCV inpainting.

This is NOT AI restoration. It smears the surrounding colours into the marked area, which is enough to exercise
the whole pipeline (masks, tracking, cropping, blending, encoding) without downloading a model, and it is real
processing, not a simulation. It is refused in production (see registry.py), and the API reports
`development_only` so the interface never presents its output as AI restoration.
"""

from __future__ import annotations

import cv2

from app.ai.base import DeviceInfo, Frame, InpaintingOptions, Mask
from app.ai.image import ImageInpaintingEngine

_SETTINGS = {  # quality -> (algorithm, radius)
    "fast": (cv2.INPAINT_TELEA, 3),
    "balanced": (cv2.INPAINT_TELEA, 5),
    "high": (cv2.INPAINT_NS, 7),
}


class DevInpaintingEngine(ImageInpaintingEngine):
    provider = "dev"
    name = "OpenCV inpainting (development only, not AI)"
    is_development_only = True

    def is_available(self) -> bool:
        return True

    def device(self) -> DeviceInfo:
        return DeviceInfo(kind="cpu", name="CPU", note="Development engine on the CPU.")

    def expectation(self) -> str:
        return "Development engine: quick, but it blurs the area instead of rebuilding it. Not for final results."

    def inpaint_image(self, crop: Frame, mask: Mask, options: InpaintingOptions) -> Frame:
        algorithm, radius = _SETTINGS[options.quality]
        return cv2.inpaint(crop, mask, radius, algorithm)
