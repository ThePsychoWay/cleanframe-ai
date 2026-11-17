#!/usr/bin/env python3
"""Make test clips with overlays whose exact position is known, so detection can be scored against the truth.

    python scripts/make_overlay_clips.py SOURCE.mp4 OUT_DIR

SOURCE is any real footage with movement. Each clip is 5 s (10 s for the scene-change one) at 30 fps; every overlay is
drawn into the pixels with the opacity, size and motion named in the file name. `truth.json` lists, per clip and per
frame, the boxes (x, y, w, h in pixels) that cover each overlay: the ground truth for scripts/bench_detect.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FPS = 30


def decode(source: Path, width: int, height: int, seconds: float, start: float = 0.0) -> list[np.ndarray]:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(source), "-t", str(seconds), "-vf", f"fps={FPS},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    size = width * height * 3
    return [np.frombuffer(raw[i : i + size], np.uint8).reshape(height, width, 3).copy() for i in range(0, len(raw) - size + 1, size)]


def sprite(kind: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(BGR, alpha 0..1) of a logo or a line of text, drawn with anti-aliasing."""
    if kind == "logo":
        d = int(round(84 * scale))
        img = np.zeros((d, d, 3), np.uint8)
        a8 = np.zeros((d, d), np.uint8)
        cv2.circle(a8, (d // 2, d // 2), d // 2 - 2, 255, -1, cv2.LINE_AA)
        a = a8.astype(np.float32) / 255
        img[:] = (40, 120, 255)
        cv2.putText(img, "CF", (int(d * 0.18), int(d * 0.64)), cv2.FONT_HERSHEY_DUPLEX, d / 50, (255, 255, 255), max(2, d // 24), cv2.LINE_AA)
        return img, a
    text = "@channel_name" if kind == "text" else "STUDIO"
    font, thick = cv2.FONT_HERSHEY_DUPLEX, max(1, round(2 * scale))
    (tw, th), base = cv2.getTextSize(text, font, 0.9 * scale, thick)
    img = np.zeros((th + base + 8, tw + 8, 3), np.uint8)
    a8 = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(a8, text, (4, th + 2), font, 0.9 * scale, 255, thick, cv2.LINE_AA)
    a = a8.astype(np.float32) / 255
    img[:] = (255, 255, 255)
    return img, a


def put(frame: np.ndarray, spr: tuple[np.ndarray, np.ndarray], x: float, y: float, opacity: float) -> tuple[int, int, int, int] | None:
    img, a = spr
    h, w = a.shape
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    sa = (a[y0 - y : y1 - y, x0 - x : x1 - x] * opacity)[..., None]
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (region * (1 - sa) + img[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32) * sa).astype(np.uint8)
    return (x, y, w, h)


def write(path: Path, frames: list[np.ndarray]) -> None:
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def corners(w: int, h: int, sw: int, sh: int, m: int = 24) -> list[tuple[float, float]]:
    return [(m, m), (w - sw - m, m), (w - sw - m, h - sh - m), (m, h - sh - m)]


def scenario(name: str, base: list[np.ndarray]) -> tuple[list[np.ndarray], list[list[tuple[int, int, int, int]]]]:
    n, (h, w) = len(base), base[0].shape[:2]
    out = [f.copy() for f in base]
    truth: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n)]
    logo, text, studio = sprite("logo", h / 540), sprite("text", h / 540), sprite("studio", h / 540)
    lh, lw = logo[1].shape
    th, tw = text[1].shape

    def add(i: int, spr, x: float, y: float, op: float) -> None:
        box = put(out[i], spr, x, y, op)
        if box:
            truth[i].append(box)

    for i in range(n):
        t = i / FPS
        if name == "static_logo":
            add(i, logo, w - lw - 24, h - lh - 24, 0.95)
        elif name == "transparent_logo":
            add(i, logo, 24, 24, 0.35)
        elif name == "text":
            add(i, text, (w - tw) / 2, h - th - 20, 0.85)
        elif name == "multiple":
            add(i, logo, w - lw - 24, 24, 0.9)
            add(i, text, 24, h - th - 20, 0.8)
            add(i, studio, w - studio[1].shape[1] - 24, h - studio[1].shape[0] - 20, 0.6)
        elif name == "moving_jump":
            c = corners(w, h, lw, lh)
            add(i, logo, *c[int(t // 1.0) % 4], 0.9)
        elif name == "moving_smooth":
            x = 24 + (w - lw - 48) * abs(((t * 0.35) % 2) - 1)
            y = 24 + (h - lh - 48) * abs(((t * 0.22 + 0.3) % 2) - 1)
            add(i, logo, x, y, 0.9)
        elif name == "appear_disappear":
            if 1.0 <= t < 3.0 or t >= 4.0:
                add(i, logo, w - lw - 24, 24, 0.9)
        elif name == "size_change":
            s = 0.8 + 0.5 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 5))
            sp = sprite("logo", h / 540 * s)
            add(i, sp, w - sp[1].shape[1] - 24, h - sp[1].shape[0] - 24, 0.9)
        elif name == "scene_changes":
            add(i, logo, w - lw - 24, 24, 0.8)
            add(i, text, 24, h - th - 20, 0.8)
        elif name == "no_overlay":
            pass
    return out, truth


def main() -> None:
    source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = 960, 454
    truth_all: dict[str, dict] = {}
    # The source is portrait with a still tweet above and the moving footage below: use the moving part, widened.
    crop = "crop=720:340:0:785"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-t", "5.1", "-vf", f"fps={FPS},{crop},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], check=True, capture_output=True).stdout
    size = width * height * 3
    base = [np.frombuffer(raw[i : i + size], np.uint8).reshape(height, width, 3).copy() for i in range(0, len(raw) - size + 1, size)][:150]
    for name in ["static_logo", "transparent_logo", "text", "multiple", "moving_jump", "moving_smooth", "appear_disappear", "size_change", "no_overlay"]:
        frames, truth = scenario(name, base)
        write(out_dir / f"{name}.mp4", frames)
        truth_all[name] = {"width": width, "height": height, "fps": FPS, "boxes": truth}
    # Changing scenes: four shots made from the same footage, mirrored/flipped/shifted/reversed so each looks different.
    shots = [base[:75], [f[:, ::-1] for f in base[40:115]], [f[::-1, :] for f in base[10:85]][::-1], [np.roll(f, 120, axis=1) for f in base[60:135]]]
    scenes = [f for shot in shots for f in shot][:300]
    frames, truth = scenario("scene_changes", scenes)
    write(out_dir / "scene_changes.mp4", frames)
    truth_all["scene_changes"] = {"width": width, "height": height, "fps": FPS, "boxes": truth}
    # 1080p copy of the static logo case, for resolution scaling.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-t", "5.1", "-vf", f"fps={FPS},{crop},scale=2288:1080", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], check=True, capture_output=True).stdout
    size = 2288 * 1080 * 3
    big = [np.frombuffer(raw[i : i + size], np.uint8).reshape(1080, 2288, 3).copy() for i in range(0, len(raw) - size + 1, size)][:150]
    frames, truth = scenario("static_logo", big)
    write(out_dir / "static_logo_1080p.mp4", frames)
    truth_all["static_logo_1080p"] = {"width": 2288, "height": 1080, "fps": FPS, "boxes": truth}
    (out_dir / "truth.json").write_text(json.dumps(truth_all))
    print("wrote", len(truth_all), "clips to", out_dir)


if __name__ == "__main__":
    main()
