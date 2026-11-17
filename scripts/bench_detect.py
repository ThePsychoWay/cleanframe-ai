#!/usr/bin/env python3
"""Score automatic detection against known truth.

    python scripts/bench_detect.py CLIPS_DIR [clip names...] [--json out.json]

For each clip made by make_overlay_clips.py it runs the real detector and mask builder, rasterises the masks at the
clip's own resolution and compares them with the true overlay boxes, frame by frame:

  recall      share of true overlay pixels covered by the mask
  precision   share of mask pixels that are overlay (a mask much bigger than the overlay lowers it)
  hit frames  frames with an overlay where at least 70 % of its box is covered (a round logo fills only 78 % of its box)
  false area  mean mask area outside the (slightly grown) truth, as % of the frame
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.detect.build import document_for  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.detect.service import removable, survey_and_detect  # noqa: E402
from app.video.masks import MaskSequence  # noqa: E402
from app.video.source import inspect_source  # noqa: E402


def frame_count(path: Path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)])
    return int(out.split()[0])


def score(path: Path, truth: dict | None) -> dict:
    media = inspect_source(path)
    n = frame_count(path)
    started = time.monotonic()
    survey, report = survey_and_detect(path, media, n, appearance=get_settings().auto_appearance)
    keep = removable(report)
    doc = document_for(keep, survey)
    seconds = time.monotonic() - started
    row: dict = {
        "clip": path.stem, "frames": n, "size": f"{media.width}x{media.height}", "detect_seconds": round(seconds, 2),
        "timings": report.timings, "overlays_found": len(report.overlays), "removed": len(keep),
        "skipped": [{"why": o.reason, "conf": o.confidence} for o in report.overlays if o.verdict != "remove"],
        "notes": report.notes,
    }  # fmt: skip
    if not doc.regions:
        seq = None
    else:
        seq = MaskSequence(doc, None, width=media.width, height=media.height, frame_count=n)
    if truth is None:
        row["note"] = "no truth"
        return row
    tp = gt_px = pred_px = false_px = 0
    hit = with_overlay = 0
    kernel = np.ones((9, 9), np.uint8)
    import cv2

    for i in range(n):
        gt = np.zeros((media.height, media.width), np.uint8)
        for x, y, w, h in truth["boxes"][i] if i < len(truth["boxes"]) else []:
            gt[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)] = 1
        pred = (seq.mask(i) > 0).astype(np.uint8) if seq else np.zeros_like(gt)
        grown = cv2.dilate(gt, kernel * 1)
        inter = int((pred & gt).sum())
        tp, gt_px, pred_px = tp + inter, gt_px + int(gt.sum()), pred_px + int(pred.sum())
        false_px += int((pred & (1 - grown)).sum())
        if gt.any():
            with_overlay += 1
            hit += int(inter / gt.sum() >= 0.7)
    row.update(
        recall=round(tp / gt_px, 3) if gt_px else None, precision=round(tp / pred_px, 3) if pred_px else None,
        hit_frames=f"{hit}/{with_overlay}", false_area_pct=round(100 * false_px / (n * media.width * media.height), 3),
    )  # fmt: skip
    return row


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = None
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        args = [a for a in args if a != str(out)]
    folder = Path(args[0])
    truth = json.loads((folder / "truth.json").read_text())
    names = args[1:] or sorted(p.stem for p in folder.glob("*.mp4"))
    rows = []
    for name in names:
        row = score(folder / f"{name}.mp4", truth.get(name))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "timings"}))
        print("   ", row["timings"])
    if out:
        out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
