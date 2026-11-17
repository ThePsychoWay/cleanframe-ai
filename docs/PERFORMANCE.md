# Performance report

Everything here was measured with `scripts/bench_pipeline.py`, which runs the real render pipeline (the same
`render_restored_video` the app uses) on real footage and times each stage. Nothing is estimated or simulated.

**Test machine:** 2 CPU cores, no GPU, Python 3.13, PyTorch CPU. A GPU machine was not available, so every GPU item
below is marked *not measured*.

## 1. Where the time goes (baseline, before any change)

Real 720p/480p footage with two marks (a small corner logo and a wide bottom caption), balanced quality.

| Stage | 480p, 5.2 s (155 fr) | 720p, 5.2 s (155 fr) | CPU | RAM | GPU / VRAM |
|---|---|---|---|---|---|
| Probe + survey decode | 0.8 s | 1.8 s | low | – | n/a |
| Overlay measurement (tried first) | 4.1 s | 9.9 s | 1 core | – | n/a |
| Camera-motion estimate | 1.7 s | 2.4 s | 1 core | – | n/a |
| Footage search (warp/median) | ~17 s | ~29 s | 1 core | – | n/a |
| **LaMa generative fill** | **107 s** | **156 s** | both cores (≈180 %) | peak 1.5 / 2.6 GB (whole process) | n/a |
| Decode + encode + audio mux + verify | ~1.5 s | ~2 s | low | – | n/a |
| **Total** | **132.6 s** | **200.8 s** | mean 182 % | | not measured |

**Bottleneck #1 – LaMa inference, 80 % of the time** (about 0.7 s per frame on CPU).
**Bottleneck #2 – a fill was generated for every frame**, even when the picture around the mark had not changed.
**Bottleneck #3 – per-frame footage search** that could not succeed (neighbour frame hides exactly the same pixels).
FFmpeg decode/encode/mux/verify together are ~1 % – they are not a bottleneck on this machine.

## 2. What was changed

| # | Change | Why faster | Quality trade-off | Measured |
|---|---|---|---|---|
| 1 | **Reuse a generated fill** while the picture around the mark is unchanged (`temporal.py`) | LaMa is not run again for a hole it has already filled from identical surroundings | At *high* quality only a byte-identical surrounding is reused; *balanced* allows mean grey difference ≤ 1.5 and ≤ 2 % of pixels off by > 24; *fast* ≤ 3.0. A reused fill is the same patch every frame, so it cannot flicker | Fixed logo: **128.1 s → 34.8 s (3.7×)**, LaMa runs 155 → 28; quality 17.35 → 17.39 dB, flicker 0.423 → 0.419 |
| 2 | **Skip the footage search for a neighbour with the same mask at the same position** | That neighbour hides exactly the missing pixels, so it can never supply any (proven by the mask maths; a test asserts identical output with the skip off) | none (exact) | Not measurable on its own (a few seconds of the ~30 s non-LaMa time at most) |
| 3 | **Model loaded once per process, warmed up, one inference at a time** (`registry.py`, `lama.py`, `main.py`) | No reload per video; first frame no longer pays one-time set-up; avoids two jobs fighting over cores/GPU memory | none. `AI_KEEP_MODEL_LOADED=false` restores the old behaviour (costs ~0.2 GB RAM while idle) | Model load per job 0.5 s – 6 s saved |

Moving-mark case (the picture around the mark changes every frame): **120.2 s → 120.3 s, no gain**, quality
identical (32.69 dB). Reuse only helps where reuse is legitimate. This is the honest limit of change 1.

## 4. Do the targets hold?

| Video | Target | Result on this machine | Verdict |
|---|---|---|---|
| 5 s 480p | 5–10 s | 93.7 s (was 132.6 s) | **Not achieved** |
| 5 s 720p | 10–20 s | 160.6 s (was 200.8 s); 34.8 s best case (fixed mark) | **Not achieved** |
| 10 s 720p, 15 s 1080p | 15–30 s / aggressive | not run (would scale linearly with frames) | **Not achieved** |

The cause is the CPU: LaMa needs ~0.7 s per generated fill here. A GPU should cut that by roughly an order of
magnitude, which is the only route to the targets for marks that must be generated every frame – but **that is an
expectation, not a measurement**; run the commands in §6 on a GPU machine to find out.

## 5. Benchmark table (every row is a real run; 5.2 s clips, 155 frames, balanced quality)

| Clip | Marks | Before | After | Faster | LaMa runs before → after | Quality vs. original |
|---|---|---|---|---|---|---|
| Real footage 480p | 2 (corner + caption) | 132.6 s | 93.7 s | 1.4× | 310 → 185 | not scored: no real logo in this footage, so there is no truth to compare with |
| Real footage 720p | 2 (corner + caption) | 200.8 s | 160.6 s | 1.25× | 310 → 183 | not scored (same reason) |
| 720p, logo drawn on still-camera footage, **fixed** | 1 | 128.1 s | 34.8 s | **3.7×** | 155 → 28 | 17.35 → 17.39 dB (same); flicker 0.423 → 0.419 |
| 720p, logo drawn on footage, **moving** | 1 | 120.2 s (clean re-run) | 120.3 s | 1.0× | 155 → 147 | 32.69 → 32.69 dB (identical) |

Throughput after: 1.65 fps (480p), 0.97 fps (720p). Peak process memory is unchanged (1.5 GB at 480p, 2.6 GB at 720p).
The 128 s / 120 s baselines come from runs with nothing else on the machine; an earlier 158 s reading of the moving
case was disturbed by a test run and is not used.

## 6. Things deliberately not done (and why)

* **FP16 / bf16**: bf16 autocast fails inside the TorchScript LaMa (FFT dtype); cuFFT half precision needs
  power-of-two sizes. Cannot be verified without a GPU, so it is not enabled.
* **NVENC / NVDEC**: encode+decode are ~1 % of the time here; not worth an unverified code path.
* **channels_last, freeze/optimize, batching**: measured, no gain on CPU (`scripts/bench_lama.py`).
* **Smaller LaMa crops**: 3× faster but visibly worse fill – a quality loss, so not applied.
* **Alternative models (ONNX/TensorRT, video-inpainting models)**: no second model is shipped; the engine interface
  (`ImageInpaintingEngine`) is interchangeable, `bench_lama.py` is the place to compare. No model was swapped without
  a comparison.
* **Automatic FAST/BALANCED/HIGH profile selection**: the three quality levels exist and now change how aggressively fills are
  reused, but choosing one automatically from resolution/duration/hardware is **not implemented**.
* **Server-side AI watermark detection**: there is none; detection is the browser-side heuristic plus tracking.
* **Per-job benchmark record in the database**: the render log line now carries frames, seconds and fps; a stored
  per-job record was not added.

## 7. Reproduce / run on a GPU machine

```bash
# real pipeline, per-stage timings, CPU/RSS (and GPU util / VRAM via nvidia-smi when present)
python scripts/bench_pipeline.py clip.mp4 --mask '[[x,y,w,h]]' --label mytest --out result.json
# compare a restored video with the original in the marked area (PSNR + flicker)
python scripts/quality_compare.py original.mp4 restored.mp4 --mask '[[x,y,w,h]]'
# LaMa variants (fp32 / fp16 / crop sizes / batching)
python scripts/bench_lama.py
```

`x,y,w,h` are fractions of the frame. On a GPU machine, check `device` and `gpu` in the JSON to confirm the model really ran on CUDA.

## 8. Automatic mode: measured additions (see AUTOMATION.md)

Detection adds 1.5–2.5 s to a 5 s 960×454 clip and about 11 s to a 5 s 720×1280 60 fps clip (survey decode 2.3 s, tracking
4.2 s, repeats search 2.9 s); the post-render quality check adds 1.4–4.6 s. End to end on the real Shorts clip: 12.8 s
analysis+detection, 157.0 s render, 174.3 s after the upload finished (the manual flow's render of the same clip with
hand-drawn marks was 160.6 s, §5). The render is still ~80 % of the time and is bounded by LaMa on 2 CPU cores; the
15–20 s target is not met on this machine and no GPU numbers exist.
