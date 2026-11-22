# Video restoration models: evaluation

Checked 22 Nov 2026. Everything in the "Source" column was read from the cited page that day; nothing here was benchmarked on a GPU, because the development box has none (2 vCPU, 8 GB, no CUDA).

## What the engine can plug in

`InpaintingEngine` (`app/ai/base.py`) is the swappable interface: `process(frames, masks, options)` streams frames in and out, and an engine either works or raises. `AI_PROVIDER` picks one (`auto`, `reconstruct`, `lama`, `local`, `video`, `api`). A model that needs a clip of frames at once (ProPainter, E2FGVI) belongs behind `video`; a per-frame model (LaMa) belongs behind `lama`/`local`. Nothing below is bundled except LaMa.

## Comparison

| Model | Kind | Licence | Memory (as published) | Quality evidence | Fit for CleanFrame |
|---|---|---|---|---|---|
| **Reconstruction from other frames** (built in) | Classical: motion-aligned copy from frames that show the hidden pixels | Ours | CPU, ~1.8 GB peak at 720p | Exact where another frame shows the pixels (33.8 dB vs clean footage in the repo's e2e run) | Default. Cannot invent anything; refuses when nothing shows the area |
| **LaMa** (built in, ONNX) | Per-frame generative image inpainting | Apache-2.0 | ~0.6 s/frame at 720p on 2 vCPU (measured) | Good single-frame texture; no temporal model, so the repo adds fill reuse + shimmer smoothing | Fallback for pixels no frame shows. Commercial use is fine |
| **ProPainter** | Flow propagation + transformer, clip-based | NTU S-Lab 1.0, non-commercial only | fp32/fp16: 1280×720 50 frames 28 G / 19 G; 720×480 50 frames 11 G / 7 G; 320×240 3 G / 2 G | Highest PSNR of the feed-forward models on DAVIS/YouTube-VOS (29.55 / 31.70 dB at 240p, ahead of E2FGVI 28.71 / 31.18); in a 2026 recoverability study its LPIPS was worse (0.273 vs 0.231 for E2FGVI-HQ) | Best feed-forward option for personal/non-commercial use. A commercial service needs the authors' permission |
| **E2FGVI / E2FGVI-HQ** | End-to-end flow-guided transformer | CC BY-NC 4.0 (from earlier research, not rechecked today) | Lower than ProPainter; a survey on a T4 found it the most deployable of the older models | 8.8 FPS offline at 432×240 (paper table via arXiv 2403.16161); best LPIPS in the 2026 study among the non-diffusion models | Cheaper alternative to ProPainter; also non-commercial |
| **DiffuEraser** | Diffusion with ProPainter prior | Needs ProPainter prior, so non-commercial | Heavy (diffusion) | Better texture and consistency than ProPainter in its paper; 10 s of 540p/25 FPS took ~200 s on an L20 | Too slow for the 15–20 s goal; consider only as an optional "highest quality" tier |
| **MiniMax-Remover** | Diffusion remover | CC BY-NC 4.0 (earlier research) | Heavy | Best LPIPS (0.194) in the 2026 study, lower PSNR than ProPainter | Same limits as DiffuEraser |

Sources: ProPainter repository (licence text, memory table, removal of its watermark demo); arXiv 2309.03897 (ProPainter vs E2FGVI), 2403.16161 (FPS table), 2501.10018 (DiffuEraser timing), 2611.01578 (recoverability study), 2401.17883 (T4 survey), 2412.08975 (DAVIS/YTVI table). One mirror listing ProPainter as BSD-3-Clause was ignored: the upstream repository says S-Lab 1.0.

## What follows

1. **No model is the default for a still mark on a still camera.** The pixels are in no frame. Any model, including ProPainter, produces a plausible fill, not the original. The render notes already say so, and that wording stays.
2. **Licence decides the commercial path.** LaMa is the only model here that can ship in a paid service. ProPainter/E2FGVI are fine for the owner's own non-commercial use and can be supplied through `AI_PROVIDER=video` with the user's own weights.
3. **ProPainter's own authors removed their watermark-removal demo for misuse reasons.** CleanFrame keeps its ownership notice on every removal for the same reason.
4. **Speed goal (15–20 s for a 5–6 s clip) is not established.** ProPainter at 720p needs 19 GB (fp16, 50 frames) or sub-video chunking; on a datacentre GPU that may fit the goal, but nothing was measured here. Until someone runs `scripts/bench_pipeline.py` on a CUDA host, `docs/PERFORMANCE.md` numbers (CPU only) are the only ones the project claims.
