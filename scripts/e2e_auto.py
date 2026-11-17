"""End-to-end check of the AUTOMATIC flow against a RUNNING server: upload one video, touch nothing else.

Reports measured wall-clock times, never estimates:
    upload            bytes sent -> upload/complete accepted
    analysis+detect   complete -> project has its `automation` record (preview, probing and detection together)
    render            job queued -> job completed (with the stages the job itself reports)
    download          signed link -> file on disk
and verifies the downloaded file with ffprobe (size, frames, fps, audio kept).

    python scripts/e2e_auto.py --server http://localhost:8000 --video clip.mp4 --out /tmp/e2e [--clean twin.mp4]

With --clean (a copy of the clip without the overlay) the PSNR over the area the detector masked is reported next to
the PSNR of the unprocessed original over the same area, so "better than doing nothing" is a number, not a claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests


def probe(path: Path) -> dict[str, Any]:
    raw = subprocess.run(  # noqa: S603
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],  # noqa: S607
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(raw)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    num, den = video["r_frame_rate"].split("/")
    return {
        "width": int(video["width"]), "height": int(video["height"]), "fps": float(num) / float(den),
        "frames": int(video["nb_read_frames"]), "audio": None if audio is None else audio["codec_name"],
    }  # fmt: skip


def local(url: str) -> str:
    """Signed links point at the server's own origin; go through the --server base so a proxy/LAN address still works."""
    parts = urlsplit(url)
    return f"{parts.path}?{parts.query}" if parts.query else parts.path


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if err == 0 else 10 * float(np.log10(255.0**2 / err))


def masked_psnr(clean: Path, other: Path, boxes: list[tuple[float, float, float, float]]) -> float:
    a, b = cv2.VideoCapture(str(clean)), cv2.VideoCapture(str(other))
    values = []
    while True:
        ok1, x = a.read()
        ok2, y = b.read()
        if not (ok1 and ok2):
            break
        h, w = x.shape[:2]
        hole = np.zeros((h, w), bool)
        for bx, by, bw, bh in boxes:
            hole[int(by * h) : int((by + bh) * h) + 1, int(bx * w) : int((bx + bw) * w) + 1] = True
        values.append(psnr(x[hole], y[hole]))
    return round(float(np.mean(values)), 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--marks", type=Path, help="mask document (JSON) to save and render when detection finds nothing")
    parser.add_argument("--no-detect", action="store_true", help="server runs with AUTO_PROCESS=false: no detection, go straight to --marks")
    parser.add_argument("--timeout", type=int, default=3 * 3600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    label = args.label or args.video.stem
    api = args.server.rstrip("/") + "/api"
    s = requests.Session()
    report: dict[str, Any] = {"label": label, "source": probe(args.video)}
    print(f"== {label}: {report['source']}")

    s.post(f"{api}/auth/register", json={"email": f"auto-{uuid.uuid4().hex[:8]}@example.com", "name": "Auto", "password": "Correct-horse-9battery"}).raise_for_status()
    data = args.video.read_bytes()
    t0 = time.monotonic()
    created = s.post(f"{api}/projects", json={"filename": args.video.name, "size_bytes": len(data)})
    created.raise_for_status()
    body = created.json()
    pid, upload = body["project"]["id"], body["upload"]
    parts = []
    for n in range(1, upload["total_parts"] + 1):
        link = s.post(f"{api}/projects/{pid}/upload/urls", json={"part_numbers": [n]}).json()["urls"][0]
        put = s.put(args.server.rstrip("/") + local(link["url"]), data=data[(n - 1) * upload["part_size"] : n * upload["part_size"]])
        put.raise_for_status()
        parts.append({"part_number": n, "etag": put.headers["ETag"]})
    done = s.post(f"{api}/projects/{pid}/upload/complete", json={"parts": parts})
    assert done.status_code == 202, done.text
    t_uploaded = time.monotonic()
    report["upload_seconds"] = round(t_uploaded - t0, 2)

    project: dict[str, Any] = {}
    t_detected = None
    t_done = None
    last = None
    stages: list[tuple[float, str]] = []
    while time.monotonic() - t_uploaded < args.timeout:
        project = s.get(f"{api}/projects/{pid}").json()
        now = time.monotonic() - t_uploaded
        if project.get("automation") and t_detected is None:
            t_detected = now
        key = (project["status"], (project.get("automation") or {}).get("state"))
        if key != last:
            print(f"  [{now:7.1f}s] {key}")
            stages.append((round(now, 1), f"{key[0]}/{key[1]}"))
            last = key
        if args.no_detect and project["status"] == "MASKING" and t_detected is None:
            t_detected = now
        if project["status"] in {"COMPLETED", "FAILED", "MASKING"} and t_detected is not None:
            t_done = now
            break
        time.sleep(0.5)
    if project["status"] == "MASKING" and args.marks:
        print("  detection found nothing; saving the supplied marks and rendering them")
        t_marks = time.monotonic()
        saved = s.put(f"{api}/projects/{pid}/mask", json=json.loads(args.marks.read_text()))
        saved.raise_for_status()
        started = s.post(f"{api}/projects/{pid}/render", json={})
        started.raise_for_status()
        while time.monotonic() - t_marks < args.timeout:
            project = s.get(f"{api}/projects/{pid}").json()
            if project["status"] in {"COMPLETED", "FAILED"}:
                t_done = time.monotonic() - t_uploaded
                break
            time.sleep(1)
        report["manual_render_seconds"] = round(time.monotonic() - t_marks, 1)
    automation = project.get("automation") or {}
    report.update(
        status=project["status"], error=project.get("error"), automation=automation,
        seconds_to_detection_record=None if t_detected is None else round(t_detected, 1),
        seconds_upload_complete_to_finished=None if t_done is None else round(t_done, 1), timeline=stages,
    )
    if project["status"] != "COMPLETED":
        print(f"  not completed: {project['status']} {automation.get('state')} - {automation.get('state_reason') or project.get('error')}")
        return finish(report, args, label, 1 if project["status"] == "FAILED" else 0)

    job = s.get(f"{api}/projects/{pid}/render").json()["job"]
    report["job"] = {k: job.get(k) for k in ("status", "stages", "frames_total", "result", "warnings", "engine")}
    t1 = time.monotonic()
    export = s.post(f"{api}/projects/{pid}/export").json()
    out = args.out / f"{label}_restored.mp4"
    with s.get(args.server.rstrip("/") + local(export["url"]), stream=True) as response:
        response.raise_for_status()
        with out.open("wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
    report["download_seconds"] = round(time.monotonic() - t1, 2)
    report["output"] = probe(out)
    src, got = report["source"], report["output"]
    report["checks"] = {
        "same_size": (src["width"], src["height"]) == (got["width"], got["height"]),
        "same_frames": src["frames"] == got["frames"], "audio_kept": (src["audio"] is None) == (got["audio"] is None),
    }
    if args.clean:
        boxes = []
        doc = s.get(f"{api}/projects/{pid}/mask").json().get("mask") or {}
        for region in doc.get("regions", []):
            for k in region["keyframes"]:
                if "rect" in k:
                    r = k["rect"]; boxes.append((r["x"], r["y"], r["w"], r["h"]))
                elif "points" in k:
                    xs, ys = [p["x"] for p in k["points"]], [p["y"] for p in k["points"]]; boxes.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
        if boxes:
            report["psnr_over_masked_area_db"] = {"original_vs_clean": masked_psnr(args.clean, args.video, boxes), "result_vs_clean": masked_psnr(args.clean, out, boxes)}
    return finish(report, args, label, 0 if all(report["checks"].values()) else 1)


def finish(report: dict[str, Any], args: argparse.Namespace, label: str, code: int) -> int:
    (args.out / f"{label}.json").write_text(json.dumps(report, indent=2, default=str))
    summary = {k: report.get(k) for k in ("upload_seconds", "seconds_to_detection_record", "seconds_upload_complete_to_finished", "download_seconds", "checks", "psnr_over_masked_area_db")}
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
