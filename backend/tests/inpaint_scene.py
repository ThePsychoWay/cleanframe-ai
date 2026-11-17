"""Scenes with known ground truth for the inpainting tests: a textured background (optionally panning or
cutting to a different texture) with a solid watermark at fixed positions in the frame. The clean frames are the
answer a perfect engine would give."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


@dataclass
class Scene:
    clean: list[np.ndarray]  # what the video looked like without the watermark
    marked: list[np.ndarray]  # what is given to the engine
    masks: list[np.ndarray]  # 255 over the watermark
    boxes: list[Box]

    @property
    def region(self) -> np.ndarray:
        return self.masks[0] > 0


def texture(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    layers = []
    for scale, weight in ((6, 0.5), (16, 0.35), (48, 0.15)):
        small = rng.random((height // scale + 2, width // scale + 2, 3)).astype(np.float32)
        layers.append(cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC) * weight)
    return (np.clip(sum(layers), 0, 1) * 150 + 40).astype(np.uint8)


def make_scene(
    *,
    width: int = 320,
    height: int = 180,
    frames: int = 12,
    boxes: list[Box] | None = None,
    pan_per_frame: int = 0,
    cut_at: int | None = None,
    mask_margin: int = 2,
    seed: int = 3,
) -> Scene:
    boxes = boxes if boxes is not None else [(220, 16, 300, 48)]
    pad = abs(pan_per_frame) * frames + 8
    first = texture(width + 2 * pad, height + 2 * pad, seed)
    second = texture(width + 2 * pad, height + 2 * pad, seed + 100)

    clean, marked, masks = [], [], []
    for t in range(frames):
        source = second if cut_at is not None and t >= cut_at else first
        dx = pad + pan_per_frame * t
        picture = np.ascontiguousarray(source[pad : pad + height, dx : dx + width])
        shown = picture.copy()
        mask = np.zeros((height, width), np.uint8)
        for x0, y0, x1, y1 in boxes:
            shown[y0:y1, x0:x1] = (30, 30, 200)
            cv2.rectangle(shown, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)
            mask[max(0, y0 - mask_margin) : y1 + mask_margin, max(0, x0 - mask_margin) : x1 + mask_margin] = 255
        clean.append(picture)
        marked.append(shown)
        masks.append(mask)
    return Scene(clean, marked, masks, boxes)


def mean_error(frames: list[np.ndarray], truth: list[np.ndarray], region: np.ndarray) -> float:
    return float(np.mean([np.abs(f[region].astype(np.float32) - t[region].astype(np.float32)).mean() for f, t in zip(frames, truth, strict=True)]))
