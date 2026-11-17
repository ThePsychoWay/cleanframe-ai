#!/usr/bin/env python3
"""Load test against the running local stack (scripts/e2e/start.sh). Run it with the backend virtualenv's Python:

    RATE_LIMIT_ENABLED=false AI_PROVIDER=auto WORKER_CONCURRENCY=1 scripts/e2e/start.sh
    scripts/load/make_load_media.sh
    $CF_VENV/bin/python scripts/load/load_test.py --users 6

Three measurements, each printed as a small table:
  1. API reads: many signed-in users repeatedly asking for their projects, the engine status and their account.
  2. Sign-in: concurrent logins (password hashing is the expensive part).
  3. The whole pipeline for N people at once: upload, analysis, marking, render, download. The stack has one worker
     process pool; the report shows how jobs queue and how long they take, plus the worker's peak memory.
Nothing here is a real credential; the accounts are throwaway ones in the local database.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import httpx

API = os.environ.get("API", "http://localhost:8000")
MEDIA = Path(os.environ.get("CF_E2E_DIR", "/tmp/cleanframe-e2e")) / "media"
PASSWORD = "correct horse battery"
RUN = str(int(time.time()))


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def table(title: str, rows: list[tuple]) -> None:
    print(f"\n== {title}")
    for row in rows:
        print("  " + "  ".join(str(cell) for cell in row))


def client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=60.0)


def signup(index: int) -> tuple[httpx.Client, float]:
    c = client()
    started = time.perf_counter()
    r = c.post("/api/auth/register", json={"email": f"load{RUN}-{index}@example.com", "name": f"Load {index}", "password": PASSWORD})
    assert r.status_code == 201, (r.status_code, r.text[:200])
    return c, time.perf_counter() - started


def in_threads(count: int, work) -> list:
    results, errors = [None] * count, []

    def run(i: int) -> None:
        try:
            results[i] = work(i)
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(f"{i}: {error!r}")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    if errors:
        print("  errors:", errors[:3])
    return results


# -- 1. API reads --------------------------------------------------------------------------------------------


def api_reads(users: int, seconds: float) -> list[httpx.Client]:
    made = in_threads(users, signup)
    clients = [c for c, _ in made]
    table("sign-up (one request each, all at once)", [("users", users), ("p50 s", f"{statistics.median(t for _, t in made):.2f}"), ("max s", f"{max(t for _, t in made):.2f}")])
    paths = ["/api/projects", "/api/engine", "/api/auth/me"]
    samples: dict[str, list[float]] = {p: [] for p in paths}
    codes: dict[int, int] = {}
    lock = threading.Lock()
    stop = time.monotonic() + seconds

    def hammer(i: int) -> None:
        c, n = clients[i], 0
        while time.monotonic() < stop:
            path = paths[n % len(paths)]
            started = time.perf_counter()
            r = c.get(path)
            took = time.perf_counter() - started
            with lock:
                samples[path].append(took)
                codes[r.status_code] = codes.get(r.status_code, 0) + 1
            n += 1

    began = time.monotonic()
    in_threads(users, hammer)
    wall = time.monotonic() - began
    total = sum(len(v) for v in samples.values())
    rows = [("endpoint", "requests", "p50 ms", "p95 ms", "p99 ms")]
    for path, values in samples.items():
        rows.append((path, len(values), f"{pct(values, .5) * 1000:.0f}", f"{pct(values, .95) * 1000:.0f}", f"{pct(values, .99) * 1000:.0f}"))
    rows.append((f"TOTAL {total / wall:.0f} req/s with {users} users at once", "status codes:", codes))
    table("API reads", rows)
    return clients


# -- 2. sign-in -----------------------------------------------------------------------------------------------


def logins(users: int) -> None:
    def work(i: int) -> float:
        c = client()
        started = time.perf_counter()
        r = c.post("/api/auth/login", json={"email": f"load{RUN}-{i}@example.com", "password": PASSWORD})
        assert r.status_code == 200, r.status_code
        return time.perf_counter() - started

    took = in_threads(users, work)
    took = [t for t in took if t is not None]
    table("sign-in (all at once)", [("users", users), ("p50 s", f"{statistics.median(took):.2f}"), ("max s", f"{max(took):.2f}")])


# -- 3. the whole pipeline ---------------------------------------------------------------------------------


def worker_memory(stop: threading.Event, peak: list[int]) -> None:
    while not stop.is_set():
        out = subprocess.run(["ps", "-eo", "rss,args"], capture_output=True, text=True, check=False).stdout.splitlines()
        total = sum(int(line.split(None, 1)[0]) for line in out if "celery" in line and "worker" in line and "grep" not in line)
        peak[0] = max(peak[0], total)
        stop.wait(0.5)


def pipeline(users: int, clip: str, box: tuple[float, float, float, float]) -> None:
    path = MEDIA / clip
    data = path.read_bytes()
    stop, peak = threading.Event(), [0]
    threading.Thread(target=worker_memory, args=(stop, peak), daemon=True).start()
    t0 = time.monotonic()

    def person(i: int) -> dict:
        c, _ = signup(10_000 + i)
        mark: dict[str, float] = {}
        created = c.post("/api/projects", json={"filename": clip, "size_bytes": len(data)})
        assert created.status_code == 201, created.text[:200]
        body = created.json()
        pid, upload = body["project"]["id"], body["upload"]
        size, parts = upload["part_size"], []
        urls = c.post(f"/api/projects/{pid}/upload/urls", json={"part_numbers": list(range(1, upload["total_parts"] + 1))}).json()["urls"]
        for u in urls:
            put = httpx.put(u["url"], content=data[(u["part_number"] - 1) * size : u["part_number"] * size], timeout=120)
            parts.append({"part_number": u["part_number"], "etag": put.headers["ETag"]})
        assert c.post(f"/api/projects/{pid}/upload/complete", json={"parts": parts}).status_code == 202
        mark["uploaded"] = time.monotonic() - t0
        while (s := c.get(f"/api/projects/{pid}").json())["status"] not in ("MASKING", "FAILED"):
            time.sleep(0.3)
        assert s["status"] == "MASKING", s.get("error")
        mark["analysed"] = time.monotonic() - t0
        x, y, w, h = box
        region = {"version": 1, "regions": [{"id": "r", "name": "Logo", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": x, "y": y, "w": w, "h": h}}]}]}
        assert c.put(f"/api/projects/{pid}/mask", json=region).status_code == 200
        started = c.post(f"/api/projects/{pid}/render", json={"quality": "balanced", "container": "mp4", "keep_audio": True})
        assert started.status_code == 202, started.text[:200]
        mark["render_requested"] = time.monotonic() - t0
        while (job := c.get(f"/api/projects/{pid}/render").json()["job"])["status"] in ("queued", "running"):
            if "render_started" not in mark and job["status"] == "running":
                mark["render_started"] = time.monotonic() - t0
            time.sleep(0.3)
        assert job["status"] == "completed", job.get("error")
        mark["render_done"] = time.monotonic() - t0
        link = c.post(f"/api/projects/{pid}/export").json()["url"]
        assert len(httpx.get(link, timeout=120).content) > 1000
        mark["downloaded"] = time.monotonic() - t0
        mark["frames"] = job["result"]["frames"]
        return mark

    marks = [m for m in in_threads(users, person) if m]
    stop.set()
    wall = time.monotonic() - t0
    rows = [("person", "uploaded s", "analysed s", "render asked s", "render began s", "render done s", "render ran s", "downloaded s")]
    for i, m in enumerate(sorted(marks, key=lambda m: m["render_done"])):
        rows.append((i + 1, f"{m['uploaded']:.1f}", f"{m['analysed']:.1f}", f"{m['render_requested']:.1f}", f"{m.get('render_started', float('nan')):.1f}", f"{m['render_done']:.1f}", f"{m['render_done'] - m.get('render_started', m['render_requested']):.1f}", f"{m['downloaded']:.1f}"))
    renders = [m["render_done"] - m.get("render_started", m["render_requested"]) for m in marks]
    rows.append((f"{len(marks)} of {users} finished in {wall:.0f} s: {len(marks) / wall * 60:.1f} videos/min; a render takes {statistics.median(renders):.1f} s of worker time (median); worker memory peak {peak[0] / 1024:.0f} MB",))
    table(f"whole pipeline, {users} people at once, clip {clip} ({marks[0]['frames'] if marks else '?'} frames)", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=6, help="people running the whole pipeline at once")
    parser.add_argument("--read-users", type=int, default=20)
    parser.add_argument("--read-seconds", type=float, default=15)
    parser.add_argument("--clip", default="load.mp4")
    parser.add_argument("--box", default="0.075,0.225,0.806,0.55", help="x,y,w,h of the logo as fractions of the frame")
    parser.add_argument("--skip", default="", help="comma list of: reads, logins, pipeline")
    args = parser.parse_args()
    skip = set(args.skip.split(","))
    if "reads" not in skip:
        api_reads(args.read_users, args.read_seconds)
    if "logins" not in skip:
        logins(args.read_users)
    if "pipeline" not in skip:
        pipeline(args.users, args.clip, tuple(float(v) for v in args.box.split(",")))


if __name__ == "__main__":
    main()
