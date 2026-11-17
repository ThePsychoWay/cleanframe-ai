"""PSNR of a restored video against the original footage, inside the area that was marked (and over the whole frame).

Ground truth for a quality check: draw a logo onto clean footage, remove it, then compare with the footage as it was.
    quality_compare.py original.mp4 restored.mp4 --mask '[[x,y,w,h],...]'     (fractions of the frame, like bench_pipeline.py)
Also reports flicker: how much the restored area changes between consecutive frames compared with the original's change.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if error == 0 else 10 * float(np.log10(255.0**2 / error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original")
    parser.add_argument("restored")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    boxes = json.loads(args.mask)
    a, b = cv2.VideoCapture(args.original), cv2.VideoCapture(args.restored)
    inside, whole, drift, drift_ref = [], [], [], []
    previous_r = previous_o = None
    while True:
        ok1, original = a.read()
        ok2, restored = b.read()
        if not (ok1 and ok2):
            break
        h, w = original.shape[:2]
        hole = np.zeros((h, w), bool)
        for x, y, bw, bh in boxes:
            hole[int(y * h) : int((y + bh) * h), int(x * w) : int((x + bw) * w)] = True
        inside.append(psnr(restored[hole], original[hole]))
        whole.append(psnr(restored, original))
        if previous_r is not None:
            drift.append(float(np.abs(restored[hole].astype(np.float32) - previous_r).mean()))
            drift_ref.append(float(np.abs(original[hole].astype(np.float32) - previous_o).mean()))
        previous_r, previous_o = restored[hole].astype(np.float32), original[hole].astype(np.float32)
    result = {
        "frames": len(inside),
        "psnr_marked_area_mean": round(float(np.mean(inside)), 2),
        "psnr_marked_area_min": round(float(np.min(inside)), 2),
        "psnr_whole_frame_mean": round(float(np.mean(whole)), 2),
        "frame_to_frame_change_restored": round(float(np.mean(drift)), 3),
        "frame_to_frame_change_original": round(float(np.mean(drift_ref)), 3),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
