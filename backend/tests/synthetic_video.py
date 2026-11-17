"""Synthetic test videos with known ground truth for the tracker.

A textured background (optionally panning) with a logo sprite on top that follows a path of
(centre x, centre y, scale, angle). The logo's true outline on every frame is returned so tracked
masks can be scored by overlap. Videos are encoded with H.264 like the editor's real preview, so the
tracker sees compression noise too.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

Pose = tuple[float, float, float, float]  # centre x (px), centre y (px), scale, angle (radians)


@dataclass
class SyntheticVideo:
    path: Path
    width: int
    height: int
    frame_count: int
    outlines: list[np.ndarray | None]  # logo corners (4x2, px) per frame; None when the logo isn't shown
    sprite_size: tuple[int, int]

    @property
    def aspect(self) -> float:
        return self.height / self.width

    def outline_mask(self, frame: int) -> np.ndarray:
        mask = np.zeros((self.height, self.width), np.uint8)
        outline = self.outlines[frame]
        if outline is not None:
            cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 255)
        return mask

    def normalised_outline(self, frame: int) -> list[list[float]]:
        outline = self.outlines[frame]
        assert outline is not None
        return [[float(x / self.width), float(y / self.height)] for x, y in outline]


def _texture(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    layers = []
    for scale, weight in ((4, 0.5), (12, 0.35), (40, 0.15)):
        small = rng.random((height // scale + 2, width // scale + 2, 3)).astype(np.float32)
        layers.append(cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC) * weight)
    image = np.clip(sum(layers), 0, 1)
    return (image * 150 + 40).astype(np.uint8)  # mid-tones, so a white logo stands out


def _sprite(width: int, height: int) -> np.ndarray:
    sprite = np.full((height, width, 3), (30, 30, 200), np.uint8)
    cv2.rectangle(sprite, (0, 0), (width - 1, height - 1), (255, 255, 255), 3)
    cv2.putText(sprite, "LG", (width // 8, height * 3 // 4), cv2.FONT_HERSHEY_DUPLEX, height / 28, (255, 255, 255), 2)
    cv2.circle(sprite, (width * 3 // 4, height // 3), height // 6, (0, 220, 255), -1)
    return sprite


def make_video(
    path: Path,
    *,
    width: int = 480,
    height: int = 270,
    frames: int = 90,
    pose: Callable[[int], Pose],
    background_shift: Callable[[int], tuple[float, float]] = lambda f: (0.0, 0.0),
    visible: Callable[[int], bool] = lambda f: True,
    sprite_size: tuple[int, int] = (96, 56),
    opacity: float = 1.0,
    seed: int = 7,
    fps: int = 30,
) -> SyntheticVideo:
    pad = 80
    background = _texture(width + 2 * pad, height + 2 * pad, seed)
    sprite = _sprite(*sprite_size)
    sw, sh = sprite_size
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    outlines: list[np.ndarray | None] = []

    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    for f in range(frames):
        ox, oy = background_shift(f)
        shift = np.float32([[1, 0, -(pad + ox)], [0, 1, -(pad + oy)]])
        frame = cv2.warpAffine(background, shift, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if visible(f):
            cx, cy, scale, angle = pose(f)
            cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
            matrix = np.float32(
                [[cos, -sin, cx - (cos * sw / 2 - sin * sh / 2)], [sin, cos, cy - (sin * sw / 2 + cos * sh / 2)]]
            )
            layer = cv2.warpAffine(sprite, matrix, (width, height), flags=cv2.INTER_LINEAR)
            alpha = cv2.warpAffine(np.full((sh, sw), 255, np.uint8), matrix, (width, height), flags=cv2.INTER_LINEAR)
            weight = (alpha.astype(np.float32) / 255.0 * opacity)[..., None]
            frame = (frame * (1 - weight) + layer * weight).astype(np.uint8)
            outlines.append((np.hstack([corners, np.ones((4, 1), np.float32)]) @ matrix.T).astype(np.float64))
        else:
            outlines.append(None)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0, "ffmpeg failed to encode the synthetic video"
    return SyntheticVideo(path, width, height, frames, outlines, sprite_size)
