"""Encodes 24 real 4K frames with H.264, HEVC and VP9 through FrameWriter and prints what ffprobe finds. Run from backend/ with PYTHONPATH=. in a folder holding clean_2160p.mp4 (see docs/RESOLUTION.md)."""
import json, subprocess, sys, time
from pathlib import Path
import cv2
from app.video.frames import EncodeSettings, FrameWriter
from app.video.source import ColorSpec

src = Path("clean_2160p.mp4")
cap = cv2.VideoCapture(str(src))
frames = []
for _ in range(24):
    ok, f = cap.read()
    assert ok
    frames.append(f)
color = ColorSpec(matrix="bt709", primaries="bt709", transfer="bt709", range="tv")
for container, codec, cap_h in [("mp4", "h264", None), ("mp4", "hevc", None), ("webm", "vp9", None), ("mp4", "h264", 1080), ("mp4", "hevc", 1440)]:
    out = Path(f"./codec_{codec}_{cap_h or 2160}.{container}")
    t = time.monotonic()
    w = FrameWriter(out, width=3840, height=2160, frame_rate="30/1", color=color, settings=EncodeSettings(container=container, quality="balanced", output_height=cap_h, codec=codec))
    for f in frames:
        w.write(f)
    w.finish()
    took = time.monotonic() - t
    p = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,codec_tag_string,width,height,r_frame_rate,nb_read_frames,pix_fmt", "-of", "json", str(out)], capture_output=True, text=True).stdout)["streams"][0]
    print(f"{container}/{codec} cap={cap_h}: {p['codec_name']} tag={p['codec_tag_string']} {p['width']}x{p['height']} {p['r_frame_rate']} frames={p['nb_read_frames']} {p['pix_fmt']} encoder={w.encoder.name} hw={w.encoder.hardware} {took:.1f}s {out.stat().st_size/1e6:.1f} MB")
