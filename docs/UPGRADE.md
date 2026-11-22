# Restoration engine upgrade: status (22 Nov 2026)

Honest ledger of the "advanced restoration upgrade" request. "Verified" means run here; this box has 2 vCPU, 8 GB and no GPU.

## Done and verified
| Item | Evidence |
|---|---|
| Audit of the existing pipeline (detect → track → engine → render → QC) | `docs/ARCHITECTURE.md`, this file |
| Baseline benchmark, 166-frame 720×1280 clip, one mark | ≈113 s total, LaMa ≈0.6 s/frame (137 generated, 29 reused), CPU mean 177 %, RSS peak ≈1.78 GB |
| **Input formats**: MP4, MOV, M4V, WebM, MKV, AVI (ffprobe sniffing of the real container, fake extensions refused) | `tests/test_probe.py`, `tests/test_projects.py`, `tests/test_formats.py` |
| **Export**: MP4/MOV/WebM/MKV × H.264/H.265/VP9/AV1 (valid combinations only), Original/720p/1080p/1440p/4K, Fast/Balanced/High, audio copied when the container allows | `tests/test_formats.py` renders every combination and checks codec, frame count, size, audio. AV1 needs an AV1 encoder in the server's FFmpeg; if absent the render says so |
| UI offers the new formats | 251 frontend tests, `tsc` clean |
| **Automatic QC retry**: after a render, residual/flicker are measured; if flagged, one more render at High, the better-scoring result is kept, a failed retry never loses the first result, and the person is told what happened (`QC_RETRY=false` turns it off; skipped when over half the time limit is used) | `tests/test_render_api.py` (6 new tests) |
| **Preservation check**: PSNR (worst frame) and SSIM (mean) of the picture outside the marked areas against the original, warns below 30 dB / 0.9 and feeds the retry decision | `tests/test_render_checks.py` (3 new tests) |
| Model evaluation with licences, memory, quality evidence | `docs/MODELS.md` |
| Full backend suite before the retry change | 714 passed, 1 skipped (SQLite) |

## Not done in this pass
- Moving, resizing and animated-mark masks refined to exact outlines inside the app's detector. The exact-outline method (median gradients over time, integrated, high-pass thresholds) worked on individual clips by hand; it is not wired into `app/detect`.
- LPIPS (a learned perceptual metric; needs a model download). PSNR and SSIM against the original *outside* the marked areas are now part of the QC (below).
- Automatic strategy selection beyond `auto` (overlay inversion → frame reconstruction → LaMa).
- ProPainter/E2FGVI adapters (non-commercial licences; interface is ready, see `docs/MODELS.md`).

## Needs GPU, models or deployment config (unmeasured)
- The 15–20 s target for a 5–6 s clip. On this CPU box a 5 s 720p clip takes minutes (`docs/PERFORMANCE.md`). No GPU number exists; run `scripts/bench_pipeline.py` on a CUDA host before claiming one.
- LaMa on CUDA/FP16/TensorRT, NVENC (used only when a test encode proves it works), 4K tiling speed.
- AV1 encoding speed (SVT-AV1 or libaom, whichever the host FFmpeg has).

## Limits that no software removes
A mark that never moves on a still camera hides pixels that are in no frame. Any engine then generates a plausible fill; it does not recover the original, and the render notes say so. Use removal only on content you own or may edit.
