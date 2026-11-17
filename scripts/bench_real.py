#!/usr/bin/env python3
"""Score automatic detection on a real clip against masks made by hand.

    python scripts/bench_real.py VIDEO TRUTH_MASKS.json [--json out.json]

TRUTH_MASKS.json is a mask document (the same format the editor saves). The detector and mask builder run exactly as in
production; the masks they produce are compared with the hand-made ones, frame by frame:

  recall      share of hand-marked pixels the automatic mask covers
  precision   share of the automatic mask that lies on hand-marked pixels (the hand masks are generous, so this is a floor)
  false area  mean automatic mask area farther than 12 px from any hand-marked pixel, as % of the frame
  hit frames  frames with a hand mark where at least 70 % of it is covered
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import get_settings  # noqa: E402
from app.detect.build import document_for  # noqa: E402
from app.detect.service import removable, survey_and_detect  # noqa: E402
from app.schemas.mask import MaskDocument  # noqa: E402
from app.video.masks import MaskSequence  # noqa: E402
from app.video.source import inspect_source  # noqa: E402


def frame_count(path: Path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)])
    return int(out.split()[0])


def main() -> None:
    video, truth_path = Path(sys.argv[1]), Path(sys.argv[2])
    media = inspect_source(video)
    n = frame_count(video)
    document = MaskDocument.model_validate(json.loads(truth_path.read_text()))
    truth = MaskSequence(document, None, width=media.width, height=media.height, frame_count=n)
    parts = {
        r.id: MaskSequence(MaskDocument(version=1, regions=[r]), None, width=media.width, height=media.height, frame_count=n)
        for r in document.regions
    }
    part_tp = dict.fromkeys(parts, 0)
    part_px = dict.fromkeys(parts, 0)
    part_hit = dict.fromkeys(parts, 0)
    part_frames = dict.fromkeys(parts, 0)
    started = time.monotonic()
    survey, report = survey_and_detect(video, media, n, appearance=get_settings().auto_appearance)
    keep = removable(report)
    doc = document_for(keep, survey)
    seconds = time.monotonic() - started
    seq = MaskSequence(doc, None, width=media.width, height=media.height, frame_count=n) if doc.regions else None
    tp = gt_px = pred_px = far_px = hit = marked = 0
    near = np.ones((25, 25), np.uint8)
    for i in range(n):
        gt = (truth.mask(i) > 0).astype(np.uint8)
        pred = (seq.mask(i) > 0).astype(np.uint8) if seq else np.zeros_like(gt)
        inter = int((gt & pred).sum())
        tp, gt_px, pred_px = tp + inter, gt_px + int(gt.sum()), pred_px + int(pred.sum())
        far_px += int((pred & (1 - cv2.dilate(gt, near))).sum())
        for rid, part in parts.items():
            g = (part.mask(i) > 0).astype(np.uint8)
            if g.any():
                got = int((g & pred).sum())
                part_tp[rid] += got
                part_px[rid] += int(g.sum())
                part_frames[rid] += 1
                part_hit[rid] += int(got / g.sum() >= 0.7)
        if gt.any():
            marked += 1
            hit += int(inter / gt.sum() >= 0.7)
    row = {
        "clip": video.stem, "frames": n, "seconds": round(seconds, 1), "removed": len(keep), "found": len(report.overlays),
        "recall": round(tp / gt_px, 3) if gt_px else None, "precision": round(tp / pred_px, 3) if pred_px else None,
        "false_area_pct": round(100 * far_px / (n * media.width * media.height), 3), "hit_frames": f"{hit}/{marked}",
        "per_mark": {rid: {"recall": round(part_tp[rid] / part_px[rid], 3) if part_px[rid] else None, "hit": f"{part_hit[rid]}/{part_frames[rid]}"} for rid in parts},
        "reasons": [f"{o.verdict}:{o.basis}:{o.reason[:50]}" for o in report.overlays if o.verdict == "remove"],
    }  # fmt: skip
    print(json.dumps(row))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
