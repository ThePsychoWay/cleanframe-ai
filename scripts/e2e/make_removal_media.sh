#!/bin/bash
# Makes the clips e2e_removal.js uploads, from the same synthetic scene the engine tests use:
#   overlay.mp4  a 40%-opaque white logo over footage that changes every frame (what the exact method can remove)
#   overlay_truth.mp4  the same footage without the logo
#   fade.mp4 / fade_truth.mp4  the same, but the logo fades in over 12 frames and out again (72 frames)
#   fixed.mp4    an opaque logo that never moves over a camera that never moves (nothing in the video can rebuild it)
source "$(dirname "$0")/env.sh"
cd "$CF_ROOT/backend"
"$CF_VENV/bin/python" - <<PY
import subprocess, numpy as np
from tests.test_overlay_engine import make_clip, fade_gains, W, H
from tests.test_temporal_engine import make_scene, noisy, watermark, WINDOW_H, WINDOW_W

def write(frames, path, w, h):
    subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}","-r","30","-i","-",
                    "-f","lavfi","-i","sine=frequency=440:duration=3","-c:v","libx264","-crf","12","-pix_fmt","yuv420p","-c:a","aac","-shortest",path],
                   input=b"".join(f.tobytes() for f in frames), check=True)
M = "$CF_E2E_DIR/media"
truth, shown, _ = make_clip(count=60, opacity=0.4)
write(shown, f"{M}/overlay.mp4", W, H); write(truth, f"{M}/overlay_truth.mp4", W, H)
fade_truth, fade_shown, _ = make_clip(count=72, opacity=0.4, gains=fade_gains(72, fade=12))
write(fade_shown, f"{M}/fade.mp4", W, H); write(fade_truth, f"{M}/fade_truth.mp4", W, H)
scene = make_scene(); window = np.ascontiguousarray(scene[30:30+WINDOW_H, 100:100+WINDOW_W]); rng = np.random.default_rng(9)
write([watermark(noisy(window, rng)) for _ in range(60)], f"{M}/fixed.mp4", WINDOW_W, WINDOW_H)
PY
ls -l "$CF_E2E_DIR/media"
