"""End-to-end resolution check against a RUNNING server (local mode is fine).

For one marked clip it drives the real product flow over HTTP -- the same calls the browser makes:
    register -> create project -> upload in parts (interrupted and resumed) -> analyse -> save mask -> render
    -> preview -> download -> additional export (optional) -> download
and then verifies every file with ffprobe, never with the labels the app reports:
    size, frame rate, frame count, duration, audio codec / rate / channels, audio-video start offset,
    and (when a clean twin is given) PSNR inside the marked area and outside it.

    python scripts/e2e_resolution.py --server http://localhost:8000 --marked marked_1080p.mp4 --clean clean_1080p.mp4 \
        --label 1080p --out /tmp/bench/res/out [--export 720] [--server-pid 1234]

Exit code 1 if any check fails. The JSON written to <out>/<label>.json holds every number printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import requests

MASK = {"x": 0.69, "y": 0.05, "w": 0.26, "h": 0.15}  # a little larger than the drawn plate (0.70, 0.06, 0.24, 0.13)
PLATE = (0.70, 0.06, 0.24, 0.13)

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


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
        "width": int(video["width"]), "height": int(video["height"]), "codec": video["codec_name"],
        "pix_fmt": video.get("pix_fmt"), "fps": float(num) / float(den), "frames": int(video["nb_read_frames"]),
        "video_start": float(video.get("start_time", 0)), "video_duration": float(video.get("duration") or data["format"]["duration"]),
        "color_range": video.get("color_range"), "color_space": video.get("color_space"),
        "bitrate": int(data["format"].get("bit_rate", 0)), "size": int(data["format"]["size"]),
        "audio": None if audio is None else {
            "codec": audio["codec_name"], "rate": int(audio["sample_rate"]), "channels": audio["channels"],
            "start": float(audio.get("start_time", 0)), "duration": float(audio.get("duration") or 0),
            "bitrate": int(audio.get("bit_rate") or 0),
        },
    }


class Sampler(threading.Thread):
    """Peak resident memory (MB) and mean CPU of the server process and everything it started (ffmpeg, workers)."""

    def __init__(self, pid: int | None) -> None:
        super().__init__(daemon=True)
        self.pid, self.peak_mb, self.cpu, self.stop_flag = pid, 0.0, [], threading.Event()

    def run(self) -> None:
        if self.pid is None:
            return
        root = psutil.Process(self.pid)
        while not self.stop_flag.is_set():
            try:
                procs = [root, *root.children(recursive=True)]
                self.peak_mb = max(self.peak_mb, sum(p.memory_info().rss for p in procs) / 2**20)
                self.cpu.append(sum(p.cpu_percent(interval=None) for p in procs))
            except psutil.Error:
                pass
            self.stop_flag.wait(0.5)

    def finish(self) -> dict[str, float]:
        self.stop_flag.set()
        busy = [c for c in self.cpu if c > 0]
        return {"peak_rss_mb": round(self.peak_mb), "mean_cpu_percent": round(sum(busy) / len(busy)) if busy else 0}


def download(url: str, destination: Path) -> int:
    """Streamed to disk in 1 MiB pieces, like a browser download; never held in memory."""
    total = 0
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as out:
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)
                total += len(chunk)
    return total


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if err == 0 else 10 * float(np.log10(255.0**2 / err))


def compare_quality(clean: Path, restored: Path) -> dict[str, float]:
    a, b = cv2.VideoCapture(str(clean)), cv2.VideoCapture(str(restored))
    inside, outside, marked_in = [], [], []
    while True:
        ok1, truth = a.read()
        ok2, got = b.read()
        if not (ok1 and ok2):
            break
        h, w = truth.shape[:2]
        x0, y0 = int(PLATE[0] * w), int(PLATE[1] * h)
        x1, y1 = int((PLATE[0] + PLATE[2]) * w), int((PLATE[1] + PLATE[3]) * h)
        hole = np.zeros((h, w), bool)
        hole[y0:y1, x0:x1] = True
        inside.append(psnr(truth[hole], got[hole]))
        outside.append(psnr(truth[~hole], got[~hole]))
    return {"psnr_inside_plate_db": round(float(np.mean(inside)), 2), "psnr_outside_plate_db": round(float(np.mean(outside)), 2), "frames_compared": len(inside)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--marked", required=True, type=Path)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--export", type=int, action="append", default=[], help="extra export short side (720, 1080...), repeatable")
    parser.add_argument("--quality", default="balanced")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--timeout", type=int, default=3 * 3600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    api = args.server.rstrip("/") + "/api"
    report: dict[str, Any] = {"label": args.label, "quality_setting": args.quality}

    print(f"== {args.label}: {args.marked.name}")
    source = probe(args.marked)
    report["source"] = source
    print(f"  source {source['width']}x{source['height']} {source['fps']:.3f} fps {source['frames']} frames {source['codec']} + {source['audio']}")

    session = requests.Session()
    email = f"e2e-{uuid.uuid4().hex[:8]}@example.com"
    registered = session.post(f"{api}/auth/register", json={"email": email, "name": "E2E", "password": "Correct-horse-9battery"})
    registered.raise_for_status()

    # ---- upload, interrupted halfway and resumed
    data = args.marked.read_bytes()
    t0 = time.monotonic()
    created = session.post(f"{api}/projects", json={"filename": args.marked.name, "size_bytes": len(data)})
    created.raise_for_status()
    body = created.json()
    pid, upload = body["project"]["id"], body["upload"]
    part_size, total = upload["part_size"], upload["total_parts"]

    def send(numbers: list[int]) -> list[dict[str, Any]]:
        urls = session.post(f"{api}/projects/{pid}/upload/urls", json={"part_numbers": numbers})
        urls.raise_for_status()
        sent = []
        for item in urls.json()["urls"]:
            n = item["part_number"]
            put = requests.put(item["url"], data=data[(n - 1) * part_size : n * part_size], timeout=120)
            put.raise_for_status()
            sent.append({"part_number": n, "etag": put.headers["ETag"]})
        return sent

    first_half = list(range(1, total // 2 + 1))
    if first_half:
        send(first_half)
    # "connection lost": a new session of the same person asks to upload the same file again and is handed the same project
    again = session.post(f"{api}/projects", json={"filename": args.marked.name, "size_bytes": len(data)}).json()
    state = session.get(f"{api}/projects/{pid}/upload").json()
    have = {p["part_number"] for p in state["uploaded_parts"]}
    check(again["project"]["id"] == pid, f"interrupted upload resumes the same project ({total} parts of {part_size // 2**20} MiB)")
    check(have == set(first_half), f"server remembers the {len(first_half)} parts already sent")
    todo = [n for n in range(1, total + 1) if n not in have]
    send(todo)
    all_parts = session.get(f"{api}/projects/{pid}/upload").json()["uploaded_parts"]
    parts = [{"part_number": p["part_number"], "etag": p["etag"]} for p in all_parts]
    done = session.post(f"{api}/projects/{pid}/upload/complete", json={"parts": parts})
    check(done.status_code == 202, "upload completed")
    report["upload_seconds"] = round(time.monotonic() - t0, 1)
    del data

    # ---- analysis
    t0 = time.monotonic()
    project: dict[str, Any] = {}
    while time.monotonic() - t0 < 1800:
        project = session.get(f"{api}/projects/{pid}").json()
        if project["status"] in {"MASKING", "FAILED"}:
            break
        time.sleep(1)
    report["analysis_seconds"] = round(time.monotonic() - t0, 1)
    check(project["status"] == "MASKING", f"analysis finished ({project['status']}{': ' + str(project['error']) if project['error'] else ''})")
    if project["status"] != "MASKING":
        return finish(report, args)
    report["detected"] = {k: project[k] for k in ("container", "codec", "audio_codec", "width", "height", "fps", "duration", "bitrate", "frame_count", "size_bytes")}
    check((project["width"], project["height"]) == (source["width"], source["height"]), f"detected size {project['width']}x{project['height']} equals ffprobe's")
    check(abs(project["fps"] - source["fps"]) < 0.01, f"detected fps {project['fps']:.3f}")
    check(project["frame_count"] == source["frames"], f"detected frame count {project['frame_count']}")

    # ---- the editor's preview copy (browser memory protection) -- the original must not be touched
    media = session.get(f"{api}/projects/{pid}/media").json()
    report["editor_proxy"] = {"width": media["width"], "height": media["height"]}

    # ---- mask + render
    saved = session.put(
        f"{api}/projects/{pid}/mask",
        json={"version": 1, "regions": [{"id": "region_1", "name": "Logo", "type": "rectangle", "keyframes": [{"frame": 0, "rect": MASK}]}]},
    )
    check(saved.status_code == 200, "mask saved")
    sampler = Sampler(args.server_pid)
    sampler.start()
    started = session.post(f"{api}/projects/{pid}/render", json={"quality": args.quality})
    check(started.status_code == 202, f"render started ({started.status_code} {started.text[:200] if started.status_code != 202 else ''})")
    t0 = time.monotonic()
    job: dict[str, Any] = {}
    last_stage = ""
    while time.monotonic() - t0 < args.timeout:
        job = session.get(f"{api}/projects/{pid}/render").json()["job"] or {}
        if job.get("stage") != last_stage:
            last_stage = job.get("stage", "")
            print(f"    [{time.monotonic() - t0:6.1f}s] stage {last_stage} {job.get('frames_done')}/{job.get('frames_total')}")
        if job.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(2)
    report["render_seconds"] = round(time.monotonic() - t0, 1)
    report["server"] = sampler.finish()
    check(job.get("status") == "completed", f"render {job.get('status')}{': ' + str(job.get('error')) if job.get('error') else ''}")
    report["job"] = {k: job.get(k) for k in ("status", "error", "warnings", "result", "engine")}
    if job.get("status") != "completed":
        return finish(report, args)
    result = job["result"]
    print(f"  render {report['render_seconds']}s, peak server RSS {report['server']['peak_rss_mb']} MB, mean CPU {report['server']['mean_cpu_percent']} %")
    print(f"  engine notes: {job.get('warnings')}")

    # ---- download the default (original-size) result
    link = session.post(f"{api}/projects/{pid}/export").json()
    out_file = args.out / f"{args.label}_restored.mp4"
    nbytes = download(link["url"], out_file)
    check(nbytes == link["size_bytes"], f"downloaded {nbytes / 2**20:.1f} MiB, equals the size the server reported")
    out = probe(out_file)
    report["output"] = out
    check((out["width"], out["height"]) == (source["width"], source["height"]), f"OUTPUT {out['width']}x{out['height']} == source {source['width']}x{source['height']} (ffprobe)")
    check(abs(out["fps"] - source["fps"]) < 0.01, f"fps {out['fps']:.3f} == {source['fps']:.3f}")
    check(out["frames"] == source["frames"], f"frame count {out['frames']} == {source['frames']}")
    check(abs(out["video_duration"] - source["video_duration"]) < 0.05, f"video duration {out['video_duration']:.3f}s vs {source['video_duration']:.3f}s")
    check(out["pix_fmt"] == source["pix_fmt"], f"pixel format {out['pix_fmt']} == {source['pix_fmt']}")
    if source["audio"]:
        a, b = source["audio"], out["audio"]
        check(b is not None and (b["codec"], b["rate"], b["channels"]) == (a["codec"], a["rate"], a["channels"]), f"audio stream kept: {b}")
        check(b is not None and abs(b["duration"] - a["duration"]) < 0.05, "audio duration unchanged")
        check(b is not None and abs((b["start"] - out["video_start"]) - (a["start"] - source["video_start"])) < 0.05, "audio/video start offset unchanged (sync)")
        check(b is not None and abs(b["bitrate"] - a["bitrate"]) <= max(2000, 0.02 * a["bitrate"]), f"audio bitrate {b['bitrate'] if b else None} vs {a['bitrate']} (stream copy: no recompression)")
    check(result["width"] == source["width"] and result["height"] == source["height"], "result record matches the file")
    print(f"  output: {out['codec']} {out['bitrate'] / 1e6:.1f} Mb/s, {out['size'] / 2**20:.1f} MiB (source {source['size'] / 2**20:.1f} MiB)")

    if args.clean:
        report["quality"] = compare_quality(args.clean, out_file)
        print(f"  quality vs ground truth: {report['quality']}")
        check(report["quality"]["psnr_outside_plate_db"] > 30, "the unmarked picture is faithful (PSNR outside the logo > 30 dB, one H.264 generation)")

    # ---- options + additional exports
    options = session.get(f"{api}/projects/{pid}/exports/options").json()
    sizes = [s["height"] for s in options["sizes"]]
    report["export_options"] = {"sizes": sizes, "formats": [f["label"] for f in options["formats"]], "output": options["output_label"]}
    short = min(source["width"], source["height"])
    check(all(s < short for s in sizes), f"only smaller sizes offered ({sizes}) for a {short}p source")
    check(2160 not in sizes or short > 2160, "no fake 4K offered")
    refused = session.post(f"{api}/projects/{pid}/exports", json={"height": 2160, "container": "mp4"}) if short < 2160 else None
    if refused is not None:
        check(refused.status_code == 422, f"asking for an enlargement is refused ({refused.status_code})")

    report["exports"] = []
    for height in args.export:
        t0 = time.monotonic()
        made = session.post(f"{api}/projects/{pid}/exports", json={"height": height, "container": "mp4", "quality": "balanced"})
        check(made.status_code in {200, 201, 202}, f"export {height}p requested ({made.status_code} {made.text[:160] if made.status_code >= 300 else ''})")
        if made.status_code >= 300:
            continue
        export_id = made.json()["id"] if "id" in made.json() else made.json()["job"]["id"]
        entry: dict[str, Any] = {}
        while time.monotonic() - t0 < 3600:
            listing = session.get(f"{api}/projects/{pid}/exports").json()
            entry = next(e for e in listing["exports"] if e["id"] == export_id)
            if entry["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(1)
        check(entry["status"] == "completed", f"export {height}p {entry['status']} in {time.monotonic() - t0:.1f}s {entry.get('error') or ''}")
        if entry["status"] != "completed":
            continue
        got = session.post(f"{api}/projects/{pid}/exports/{export_id}/link").json()
        target = args.out / f"{args.label}_export_{height}p.mp4"
        download(got["url"], target)
        e = probe(target)
        expect_w = round(source["width"] * height / short / 2) * 2
        expect_h = round(source["height"] * height / short / 2) * 2
        check((e["width"], e["height"]) == (expect_w, expect_h), f"EXPORT {e['width']}x{e['height']} (expected {expect_w}x{expect_h}, aspect kept) (ffprobe)")
        check(abs(e["fps"] - source["fps"]) < 0.01 and e["frames"] == source["frames"], f"export fps {e['fps']:.3f}, {e['frames']} frames")
        if source["audio"]:
            check(e["audio"] is not None and e["audio"]["codec"] == source["audio"]["codec"] and abs(e["audio"]["duration"] - source["audio"]["duration"]) < 0.05, f"export audio kept: {e['audio']}")
            check(e["audio"] is not None and abs((e["audio"]["start"] - e["video_start"]) - (source["audio"]["start"] - source["video_start"])) < 0.05, "export A/V start offset unchanged")
        report["exports"].append({"height": height, "seconds": round(time.monotonic() - t0, 1), "file": e})
        if args.clean:
            # Same framing as the truth scaled the same way: compare the export with the clean twin scaled to its size.
            scaled = args.out / f"{args.label}_clean_{height}p.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(args.clean), "-vf", f"scale={e['width']}:{e['height']}:flags=lanczos", "-c:v", "libx264", "-crf", "12", "-an", str(scaled)], check=True)  # noqa: S603,S607
            report["exports"][-1]["quality"] = compare_quality(scaled, target)
            print(f"  export {height}p quality vs scaled ground truth: {report['exports'][-1]['quality']}")
    return finish(report, args)


def finish(report: dict[str, Any], args: argparse.Namespace) -> int:
    report["failures"] = failures
    (args.out / f"{args.label}.json").write_text(json.dumps(report, indent=1))
    print(f"== {args.label}: {'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
