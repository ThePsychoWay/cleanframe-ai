"""Times every stage of the real render pipeline on a real clip. Nothing is simulated: it calls render_restored_video
with the configured engine and wraps the stage functions with timers, and samples CPU, memory and (if present) the GPU.

    python scripts/bench_pipeline.py CLIP.mp4 --mask '[[x,y,w,h],...]' [--quality balanced] [--label NAME] [--out result.json]

Environment: the usual app settings (AI_PROVIDER, AI_MODEL_PATH, LOCAL_MODE ...). Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ.get("BENCH_BACKEND") or str(Path(__file__).resolve().parents[1] / "backend"))
import psutil  # noqa: E402

from app.ai import auto as auto_mod  # noqa: E402
from app.ai import lama as lama_mod  # noqa: E402
from app.ai import temporal as temporal_mod  # noqa: E402
from app.ai.registry import get_engine  # noqa: E402
from app.schemas.mask import MaskDocument  # noqa: E402
from app.video import frames as frames_mod  # noqa: E402
from app.video import pipeline as pipe  # noqa: E402

TIMES: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)
DURATIONS: dict[str, list[float]] = defaultdict(list)  # every call, to see warm-up and spread


def timed(owner, name: str, label: str) -> None:
    original = getattr(owner, name)

    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return original(*a, **k)
        finally:
            dt = time.perf_counter() - t
            TIMES[label] += dt
            COUNTS[label] += 1
            DURATIONS[label].append(dt)

    setattr(owner, name, wrapper)


def timed_generator(owner, name: str, label: str) -> None:
    """Time spent producing each item (the generator body), not the consumer's time between items."""
    original = getattr(owner, name)

    def wrapper(*a, **k):
        gen = original(*a, **k)
        while True:
            t = time.perf_counter()
            try:
                item = next(gen)
            except StopIteration:
                TIMES[label] += time.perf_counter() - t
                return
            TIMES[label] += time.perf_counter() - t
            COUNTS[label] += 1
            yield item

    setattr(owner, name, wrapper)


class Sampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.cpu: list[float] = []
        self.rss_peak = 0
        self.gpu: list[tuple[float, int]] = []
        me = psutil.Process()
        self.me = me
        self.nvidia = shutil.which("nvidia-smi")

    def run(self) -> None:
        self.me.cpu_percent(None)
        procs: dict[int, psutil.Process] = {}
        while not self.stop.wait(0.25):
            try:
                family = [self.me, *self.me.children(recursive=True)]
                total_cpu, rss = 0.0, 0
                for p in family:
                    procs.setdefault(p.pid, p)
                    q = procs[p.pid]
                    total_cpu += q.cpu_percent(None)
                    rss += q.memory_info().rss
                self.cpu.append(total_cpu)
                self.rss_peak = max(self.rss_peak, rss)
            except psutil.Error:
                pass
            if self.nvidia:
                try:
                    out = subprocess.run([self.nvidia, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout.split(",")
                    self.gpu.append((float(out[0]), int(out[1])))
                except Exception:
                    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--mask", required=True, help="JSON list of [x,y,w,h] in 0..1")
    ap.add_argument("--quality", default="balanced")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--keep", default="", help="copy the output video here")
    args = ap.parse_args()

    regions = [
        {"id": f"r{i}", "name": f"Mark {i}", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": x, "y": y, "w": w, "h": h}}]}
        for i, (x, y, w, h) in enumerate(json.loads(args.mask), 1)
    ]
    document = MaskDocument.model_validate({"version": 1, "regions": regions})

    timed_generator(frames_mod.FrameReader, "frames", "decode")
    timed(frames_mod.FrameWriter, "write", "encode_write")
    timed(frames_mod.FrameWriter, "finish", "encode_flush")
    timed(pipe, "mux_audio", "audio_mux")
    timed(pipe, "verify_output", "verify")
    timed(pipe, "inspect_source", "probe")
    timed(lama_mod.LamaOnnxRuntime, "run", "lama_inference")
    timed(lama_mod.LamaOnnxRuntime, "load", "lama_load")
    timed(auto_mod, "measure", "overlay_measure")
    timed(temporal_mod.TemporalReconstructionEngine, "_rebuild", "temporal_rebuild")
    timed(temporal_mod.TemporalReconstructionEngine, "_enter", "temporal_enter")
    timed(temporal_mod, "_estimate_step", "camera_motion")
    timed(pipe, "_survey", "survey_decode")

    stage_marks: dict[str, float] = {}

    def progress(stage: str, done: int, total: int) -> bool:
        stage_marks.setdefault(stage, time.perf_counter())
        return True

    t_engine = time.perf_counter()
    engine = get_engine()
    TIMES["engine_setup"] = time.perf_counter() - t_engine
    work = Path(tempfile.mkdtemp(prefix="cf-bench-"))
    out = work / "out.mp4"
    sampler = Sampler()
    sampler.start()
    started = time.perf_counter()
    status, error, result = "completed", None, None
    try:
        result = pipe.render_restored_video(
            Path(args.clip), out, document=document, tracking=None, engine=engine, work_dir=work,
            settings=pipe.RenderSettings(quality=args.quality), progress=progress,
        )
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - started
    sampler.stop.set()
    sampler.join()

    frames = result.frames if result else COUNTS["encode_write"]
    clip_seconds = frames / result.fps if result else 0
    report = {
        "label": args.label or Path(args.clip).stem, "status": status, "error": error, "quality": args.quality,
        "engine": os.environ.get("AI_PROVIDER", "?"), "frames": frames, "clip_seconds": round(clip_seconds, 2),
        "total_seconds": round(total, 2), "avg_fps": round(frames / total, 2) if total else 0,
        "times": {k: round(v, 2) for k, v in sorted(TIMES.items())}, "counts": dict(COUNTS),
        "lama_first_calls_s": [round(x, 2) for x in DURATIONS["lama_inference"][:5]],
        "lama_median_s": round(sorted(DURATIONS["lama_inference"])[len(DURATIONS["lama_inference"]) // 2], 3) if DURATIONS["lama_inference"] else None,
        "stage_starts_s": {k: round(v - started, 2) for k, v in stage_marks.items()},
        "cpu_percent_mean": round(sum(sampler.cpu) / max(len(sampler.cpu), 1), 0), "cpu_percent_peak": round(max(sampler.cpu, default=0), 0),
        "rss_peak_mb": round(sampler.rss_peak / 1e6),
        "gpu": {"available": bool(sampler.nvidia), "util_mean": round(sum(g for g, _ in sampler.gpu) / len(sampler.gpu), 1) if sampler.gpu else None,
                "vram_peak_mb": max((m for _, m in sampler.gpu), default=None)},
        "notes": result.warnings if result else [], "verification_ok": (result.verification.ok if result and hasattr(result.verification, "ok") else None),
        "device": str(getattr(engine.device(), "kind", "?")) if hasattr(engine, "device") else "?",
        "cpus": os.cpu_count(),
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    if args.keep and out.exists():
        shutil.copy(out, args.keep)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
