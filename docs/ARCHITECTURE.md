# CleanFrame AI — Architecture

Workspace was empty at the start of Phase 1, so this is a greenfield build.

## 1. Recommended architecture

```
Browser (Next.js)
   │  REST (JWT) + SSE for progress
   ▼
FastAPI (api)  ──►  PostgreSQL (users, projects, masks, jobs)
   │   │
   │   └──►  S3-compatible storage (uploads, proxies, outputs)  ◄── signed URLs to browser
   ▼
Redis (queue + progress pub/sub)
   ▼
Celery worker(s)  ── CPU worker (analysis, tracking, encode)
                  └─ GPU worker (inpainting)
```

Design rules:
- The API never processes video. It validates, stores metadata, and enqueues jobs.
- Browser uploads go directly to object storage (multipart, resumable); the API only issues presigned part URLs.
- Progress is written by workers per real stage and per real frame count, then streamed to the UI. No simulated percentages.
- All AI behind `InpaintingEngine`. Providers are chosen by `AI_PROVIDER`.

## 2. Technology stack

| Layer | Choice | Reason |
|---|---|---|
| Frontend | Next.js (App Router), React, TypeScript, Tailwind | Specified |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic | Typed, async, validated |
| Queue | Redis + **Celery** | Mature, supports queues per worker type (cpu / gpu), retries, revocation |
| DB | PostgreSQL | Specified |
| Storage | S3-compatible (MinIO locally) | Presigned multipart uploads and downloads |
| Video | FFmpeg / ffprobe | Metadata, proxy, frames, audio, mux, encode |
| CV | OpenCV | Tracking, optical flow, mask ops |
| AI | PyTorch | GPU inpainting models |
| Auth | Email/password (argon2) + JWT; Google OAuth later | |
| Deploy | Docker Compose; GPU worker with NVIDIA runtime | |

Celery over RQ: we need separate CPU/GPU queues and task retries/timeouts.

## 3. Folder structure

```
cleanframe-ai/
├── frontend/            app/ components/ hooks/ lib/ services/ types/ styles/
├── backend/
│   ├── app/
│   │   ├── api/         routers (projects, auth, uploads, admin)
│   │   ├── auth/        hashing, JWT, dependencies
│   │   ├── models/      SQLAlchemy models
│   │   ├── schemas/     Pydantic request/response models
│   │   ├── services/    UsageService, ProjectService, RetentionService
│   │   ├── storage/     S3 client, signed URLs
│   │   ├── video/       ffprobe, proxy, frames, audio, mux, verify
│   │   ├── tracking/    mask propagation (optical flow / CSRT)
│   │   ├── ai/          InpaintingEngine + adapters, detection
│   │   ├── workers/     Celery app + tasks
│   │   ├── config.py
│   │   └── main.py
│   └── tests/
├── docker/
├── scripts/
├── docs/
├── .env.example
└── README.md
```

## 4. Database

- `users(id, email, name, password_hash, role, created_at)`
- `projects(id, user_id, name, original_file_key, output_file_key, status, duration, width, height, fps, codec, created_at, updated_at, expires_at)`
- `masks(id, project_id, mask_data JSONB, frame_start, frame_end, tracking_status, created_at)`
- `jobs(id, project_id, type, status, progress, stage, frames_done, frames_total, error, started_at, completed_at)`
- `usage_events(id, user_id, project_id, kind, amount, created_at)` for the billing-ready `UsageService`

Storage keys, not URLs, are stored. URLs are signed on demand. Project statuses: UPLOADED, ANALYZING, MASKING, TRACKING, PROCESSING, ENCODING, COMPLETED, FAILED, EXPIRED.

Mask format (`mask_data`, version 1, as built in Phase 5): one document per project, replaced on save (`PUT /api/projects/{id}/mask`; idempotent, so PUT rather than POST).

```
{ "version": 1,
  "regions": [ { "id", "name", "type": "rectangle" | "polygon" | "brush",
                 "keyframes": [ { "frame", "rect": {x,y,w,h} | "points": [[x,y],...] | "strokes": [{radius, points}], "erase": [strokes] } ] } ] }
```

- Coordinates are normalized 0-1 of the frame; brush radius is a fraction of frame width, so masks are resolution independent (proxy to source).
- A single keyframe applies to the whole video. Between keyframes, rectangles and same-count polygons interpolate linearly; brush strokes hold. Tracking (Phase 6) is stored separately (below) and never rewrites the person's keyframes.
- Server limits (pydantic, `extra="forbid"`): 20 regions, 200 keyframes per region, 500 polygon points, 500 strokes per keyframe, 2000 points per stroke, 50,000 points total, radius 0.001-0.25, keyframe frames strictly increasing and below the video's frame count. The editor checks the same limits before saving.
- Saving resets `tracking_status` to `none`, since tracked results no longer match the marks. The request body for all API routes is capped by `MAX_REQUEST_BODY_BYTES` (default 2 MiB; 413 beyond).

## 5. Video processing

1. ffprobe validates real content (not browser MIME) and extracts metadata.
2. Proxy generated for the editor in the same analysis task as ffprobe (Phase 5): H.264 yuv420p, at most 720 px tall, even dimensions, constant frame rate with the source's frame rate and count (frame N in the proxy is frame N in the source, rotation applied), no audio, faststart. A 640 px JPEG thumbnail is taken from it. Both are stored under `derived/{user_id}/{project_id}/` and served through short-lived signed links from `GET /api/projects/{id}/media`; they are deleted with the project. Failure to build them marks the project FAILED with a readable message. Full-res is untouched until export.
3. Tracking (Phase 6) runs on the proxy and produces per-frame transforms of the marked shapes, see 5a. Masks are drawn at proxy scale and are resolution independent, so they scale to the source.
4. Inpainting works on a **cropped, padded region of interest** around the mask (localized for small watermarks), then composited back (section 6).
5. The final render (Phase 8, `app/video/pipeline.py::render_restored_video`) is described in 5b.


### 5a. Mask tracking (Phase 6)

`POST /api/projects/{id}/track` (202) queues a Celery task on the `cpu` queue; `GET /api/projects/{id}/tracking` returns `{status, frames_done, frames_total, error, started_at, data}` (`data` only when `done`). 409 if there are no saved marks, the project isn't in MASKING, or tracking is already running; 503 (state reverted) if the queue is unreachable.

State machine: `MASKING / none|done|failed` -> `TRACKING / running` -> `MASKING / done` (or `failed` with a readable message). Saving the mask while MASKING resets tracking, since results describe marks that no longer exist. A job still `running` past the worker's limit (`TRACKING_TIME_LIMIT_SECONDS`, default 1500, plus 180 s) is reported failed on the next GET. Deleting the project or mask cancels a running job at its next progress callback.

Algorithm (`backend/app/tracking/`): for each region and each keyframe, features inside the shape are followed with pyramidal Lucas-Kanade (window 15, 3 levels) forward and backward; a forward-backward round-trip check (1.2 px) drops unreliable points, RANSAC fits a similarity transform (0.8 px threshold, tight enough to ignore a panning background behind a static logo). The per-frame estimates are chained and corrected against the keyframe's own corners to stop scale drift. Rectangles keep translation + uniform scale only, so they stay rectangles; polygons and brush strokes take the full similarity. Each segment runs from a keyframe to the next one (forwards and backwards), so a keyframe the person adds acts as a correction.

A segment stops, and records why, when: nothing trackable inside the shape at the start, too few inliers or low-confidence matches for several frames, a single-step scale jump (hard cut), total scale outside 1/6-6x, less than 25% of the shape left in frame, or the video ends. After a loss the last transform holds, so the mark stays where it last was until corrected. Nothing is guessed.

Stored in `masks.tracking_data` (version 1): `regions[id].segments[] = {anchor, step: +-1, until, transforms: [[dx, dy, scale, angle], ...], lost_at?, reason?}`. `transforms[i]` applies to frame `anchor + step*(i+1)`; parameters are in frame-width units about the bounding-box centre of the anchor keyframe's shape (y scaled by height/width). Keyframe frames use the exact keyframe shape. `app/tracking/shapes.py` (Python) and `lib/tracking.ts` (frontend) implement the same "shape at frame N" and are tested against shared cases in `backend/tests/fixtures/shape_cases.json`; later phases use `shape_at` and `rasterize` for the per-frame inpainting mask.

Measured on synthetic H.264 clips with known outlines (mean / minimum overlap with the true logo): translation 0.998 / 0.982, growth 0.98 / 0.95, static logo over panning background 1.00 / 1.00, half-transparent logo 1.00, long curved path (240 f) 0.995, 300 frames translucent over a pan >= 0.98 minimum. A logo that vanishes is detected on the exact frame. About 21 ms per frame for two regions at 1280x720 on CPU.

Known limits: rotation is under-estimated on large turns (about 0.93 mean overlap for 61 degrees over 3 s); the tracker follows image detail inside the marked area, so a loose shape over a moving background follows the background; tracking can't be cancelled yet (renders can); stale-job detection is lazy; the canvas cannot be drawn on by keyboard alone.

### 5b. Final render (Phase 8)

`render_restored_video(source, destination, document, tracking, engine, work_dir, settings, expected_frames, progress)` turns the original upload plus the person's marks into the restored file. It is plain library code with no database or storage access; `services/rendering.py` wraps it in a worker job (section 7).

Flow: check the engine is available (fail in a second) -> `inspect_source` (ffprobe) -> decode with FFmpeg to raw BGR over a pipe -> per-frame masks at the **original's full resolution** -> engine -> encode video-only through a second FFmpeg pipe -> mux the original audio -> verify -> move into place. Only one frame at a time is in flight, so memory does not grow with length.

- **Masks.** `MaskSequence` is the one definition of "mask at frame N": the editor's keyframes, or the tracked motion where tracking ran, rasterised at the real size (shapes are fractions of the frame, so nothing is scaled from the preview). All regions are removed together. A mark that does not move shares one cached mask.
- **Frame numbers.** The preview is a constant-frame-rate timeline and is the authority. The decoder uses the same rule (`-fps_mode cfr`), and analysis now stores the proxy's frame count and rate on the project (this fixed a latent mismatch for variable-frame-rate uploads: 90 stored frames, 100 on the timeline). If the decoded count differs from the project's by more than one the render stops, because the masks would be on the wrong frames.
- **Colour.** The source's matrix and range are read (untagged: BT.709 above 576 lines / 1024 wide, BT.601 below, the convention players use), used to convert to BGR, used again to convert back, and written into the output's tags. Measured round trip at high quality: mean difference about 1 level, with about one level of darkening from 8-bit rounding. HDR (PQ/HLG) is refused rather than washed out; 10-bit is read and written as 8-bit with a warning.
- **Size and timing.** Odd dimensions lose one pixel row/column (4:2:0 needs even sizes) and the result says so. `output_height` (the shorter side: 720, 1080, 1440 or 2160) only ever scales down; the default keeps the source's size, frame rate and orientation (`docs/RESOLUTION.md`). The exact frame rate (e.g. 30000/1001) is kept. Container metadata (title, location, dates) is not carried over.
- **Encoding.** MP4 = H.264 (preset/CRF by quality: fast veryfast/23, balanced fast/20, high medium/17) or H.265/HEVC; WebM = VP9 (CRF 34/31/28). NVIDIA NVENC is used only when a real test encode on the machine succeeds (`app/video/encoders.py`); otherwise the CPU encoders. Every frame is re-encoded, so untouched areas pick up ordinary codec noise (about 2.5 levels measured against the original); there is no "smart copy" of untouched GOPs yet. NVENC is not used: quality per bit matters more here than speed.
- **Audio.** Stream-copied bit for bit when the container can hold the codec (MP4: aac, mp3, ac3, eac3, opus, flac, alac; WebM: opus, vorbis), otherwise converted once to AAC (MP4) or Opus (WebM) and the result says so. Inputs keep their own timestamps, so audio that starts late (or early) relative to the picture stays that way (tested at packet level). Opus carried into MP4 can end a few milliseconds later than in Matroska.
- **Verification** (`verify.py`), before anything is delivered: size and frame rate as expected; ffprobe `-count_frames` equals the frames written; length matches the frames; audio present, decodes, and its length gap to the picture is within 0.3 s of the original's gap; no audio where none was expected; no decode errors. A **shimmer monitor** compares, on frames where the same pixels are rebuilt twice in a row, how much the rebuilt patch changes against the real pixels beside it. More than half the frames over the line fails the render ("flickers"), more than 15% adds a warning.
- **Failure and cancel.** Errors are `ProcessingError` / `InpaintingError` with messages safe to show; FFmpeg's own output only goes to the log. A progress callback returning False cancels (checked per frame and between stages). Nothing is written to the destination unless every check passed, partial files are removed, and child FFmpeg processes are killed (including if the Python process dies, via `PR_SET_PDEATHSIG`).
- **Honest numbers.** 1080p, dev engine, CPU only: about 295 ms per frame end to end (about 13 s for 45 frames). With a real model the engine dominates the time.

Known limits: a truncated file is rendered up to where it can be read and the result carries a warning (if the decoder reports errors and fewer than 98% of the expected frames arrive the render fails instead); multiple video streams or audio tracks beyond the first are not selected between; the shimmer monitor only judges marks that stay in place between frames; nothing here makes the development engine an AI restoration.

## 6. Inpainting (Phase 7)

All reconstruction sits behind one interface in `backend/app/ai/`, so a model or provider is swapped by configuration (`AI_PROVIDER`) without touching the pipeline:

```python
class InpaintingEngine(ABC):
    def is_available(self) -> bool: ...
    def process(self, frames, masks, options) -> Iterator[Frame]: ...   # frames: HxWx3 uint8 BGR; masks: HxW uint8, 255 = remove
```

Contract: one output frame per input frame, in order, same size; streaming (frames are pulled lazily); a frame with an empty mask comes back unchanged; pixels farther than `edge_dilate_px + feather_px` from the mask are bit-identical to the input; an engine either does the work or raises `InpaintingError` / `EngineUnavailableError` with a message that is safe to show. Nothing silently substitutes another engine.

| Class | `AI_PROVIDER` | What it is |
|---|---|---|
| `DevInpaintingEngine` | `dev` | OpenCV Telea/Navier-Stokes. Real processing, **not AI**: it blurs the area. Labelled `development_only`; refused when `DEV_MODE=false` (startup fails with a clear message). |
| `AutoRemovalEngine` | `auto` | Overlay inversion when the mark allows it, otherwise reconstruction from other frames (then LaMa if configured). Chosen once, from frames surveyed across the clip; the render's notes say which and why. |
| `OverlayInversionEngine` | `overlay` | Only the exact overlay inversion of section 6c; stops with a reason if the mark isn't a fixed semi-transparent logo. |
| `TemporalReconstructionEngine` | `reconstruct` | **Removal from the video's own pixels, no model, no blur.** See "Reconstruction from other frames" below. Areas no frame shows go to LaMa (`RECONSTRUCT_FALLBACK`) or stop the render. |
| `LamaOnnxEngine` (an `ImageInpaintingEngine`) | `lama` | LaMa (Apache-2.0) through ONNX Runtime: a generative fill, one crop at a time. `AI_MODEL_PATH` is an ONNX LaMa export (image `[1,3,H,W]` 0..1 RGB + mask `[1,1,H,W]` -> image; fixed 512x512 or dynamic, 0..1 or 0..255 output are detected). Optional `AI_MODEL_URL` + `AI_MODEL_SHA256` download it once, verified. |
| `LocalModelEngine` (an `ImageInpaintingEngine`) | `local` | A PyTorch model run one crop at a time. `AI_MODEL_PATH` is a `torch.export` program (`.pt2`, export with dynamic height/width) or a TorchScript file. Contract: `forward(image [1,3,H,W], mask [1,1,H,W]) -> [1,3,H,W]`, float32 RGB 0..1, mask 1 = remove, H and W multiples of 8 (padded by reflection). |
| `LocalVideoModelEngine` (a `VideoInpaintingEngine`) | `video` | A PyTorch clip model: `forward(frames [1,T,3,H,W], masks [1,T,1,H,W]) -> [1,T,3,H,W]`. Videos are cut into windows of `AI_VIDEO_CLIP_FRAMES` with `AI_VIDEO_OVERLAP_FRAMES` shared; overlaps are cross-faded so there is no step between windows. |
| `APIModelEngine` | `api` | HTTP: `POST AI_API_URL`, `Authorization: Bearer`, multipart `image` + `mask` PNG + `quality`; the service returns a same-size PNG. Only the crop around the marks is sent. HTTPS required (plain HTTP only to this machine, in development), no redirects followed, bounded response size, retries with backoff on 429/5xx, key never logged or shown. `sends_frames_to_third_party` is true, so the interface must say so before processing. |

Adding a model means writing one subclass: `ImageInpaintingEngine.inpaint_image(crop, mask, options)` or `VideoInpaintingEngine.inpaint_clip(crops, masks, options)`, and one line in `registry.py`. Everything below is shared:

* **Regions of interest** (`roi.py`): the mask is grown slightly (`edge_dilate_px`, to catch a logo's soft edge) with a feathered rim; only the marked area plus `context_padding_px` of context is cropped, reconstructed and blended back. Marks far apart (a corner logo and a timestamp) get separate crops; overlapping crops are merged. Cropping is why a small corner watermark costs a small reconstruction.
* **Colour/lighting** (`consistency.match_colour`): a constant offset between the edge of the patch and the real pixels outside the mask is removed (bounded, with a dead band so noise isn't "corrected" into flicker).
* **Temporal stability** (`consistency.TemporalStabilizer`, for image engines): the motion of the real pixels around the mask is estimated (optical flow + RANSAC), the previous result is carried along with it and blended into the new one. It checks that the motion estimate is right before using it and resets on cuts. Video engines are temporally consistent by construction and don't use it.
* **Hardware** (`device.py`): CUDA when PyTorch can use it, otherwise the CPU, with a sentence explaining why (no GPU, PyTorch build without CUDA, driver error, switched off by `CUDA_ENABLED=false`). Never raises.

`GET /api/engine` (signed-in users) reports `{provider, name, available, development_only, sends_frames_to_third_party, device, reason, expectation}` and never includes paths, addresses or keys. The interface uses it to label development output, warn before pictures are sent to a third party, and set speed expectations.

Measured (synthetic footage with known ground truth, see `tests/test_inpaint_*.py`): the development engine cuts the error inside the marked area to between a fifth and a third of what leaving the watermark in gives; with a per-frame generative stand-in (a different texture every frame), smoothing cuts frame-to-frame shimmer to about a third on a still shot and to about 60% under a 3 px/frame camera pan, without moving away from the truth. The development engine takes about 120-150 ms per 1280x720 frame on a CPU core.

Known limits (development engine): it only smears nearby colour, so textured backgrounds look blurred there; temporal smoothing follows global (camera) motion only, so something moving *behind* a mark isn't restored by an image engine (that needs a video model); video engines crop all frames of a window to one box, so a logo and a timestamp far apart make a large box (feed regions separately if that matters); no production model weights are bundled, `local`, `video` and `api` need you to supply one (PyTorch has deprecated TorchScript, hence `.pt2` is preferred).

### 6b. Reconstruction from other frames (Phase 13)

The development engine blurs. `AI_PROVIDER=reconstruct` does not: it looks for the pixels a mark hides in the *other frames* of the same video and copies the real ones in.

1. **Camera motion** between each pair of neighbouring frames is measured from the pixels outside every mark (corner tracking, forward/back optical flow, a RANSAC similarity transform; identity when the picture hasn't moved; a cut breaks the chain). The pair transforms are chained to line up every frame within `±window` frames (6/10/16 for fast/balanced/high) with the one being rebuilt.
2. **Candidates**: a neighbour supplies a pixel only where its own mark does not cover it and the pixel is inside the picture. Each neighbour is then checked on a ring of real pixels around the hole: if it doesn't match the target there (mean difference over 10 levels) it is not used, so a wrong alignment can't paint wrong content. Exposure differences are measured on the same ring and removed.
3. **Median** of the surviving candidates per pixel (at least two must exist), so something walking through the gap in one frame doesn't leave a ghost.
4. **What no frame shows** (a fixed mark on a still camera hides the same pixels in every frame) is not invented by this engine. It goes to the fallback if one is configured (LaMa, `RECONSTRUCT_FALLBACK=auto|lama`), otherwise, when more than 1% of a frame's marked pixels are left, the render stops with "N% of the marked area in frame X is hidden in every nearby frame...". A smaller remainder (edge pixels) is patched from adjacent pixels and reported.

The render's warnings state the split in plain words: "100% of the marked area was rebuilt from other frames of your video. Nothing was generated." or "40% was rebuilt from other frames; 60% was hidden in every frame and was generated by LaMa, which makes up plausible detail rather than recovering the real one." (`InpaintingEngine.notes()`, appended in `video/pipeline.py`.) `InpaintingEngine.prepare()` loads or downloads the model before any video is decoded.

**Measured** (`tests/test_temporal_engine.py`, `tests/test_reconstruct_render.py`; synthetic textured scene with ground truth, opaque logo, 1.5-level sensor noise):

| Case | Result |
|---|---|
| Panning camera, fixed opaque logo, raw frames | 48.6-51 dB PSNR inside the marked area in every frame (noise floor); OpenCV inpainting is 8+ dB lower |
| Same, through a real encode (x264 crf 10) -> render -> decode | 35.2 dB minimum / 36.8 dB mean (limited by the codec) vs 15.7 / 20.5 dB for the development blur |
| Still camera, logo slides across the picture | > 30 dB in every frame |
| Still camera, fixed logo | **cannot be rebuilt**: the render stops with a message (no fallback) or reports 100% generated (fallback) |
| Cut in the middle of a clip | no pixels taken across the cut |

**What is and is not established.** Real removal is verified here for footage in which the hidden pixels are visible somewhere else in the clip (camera pan/zoom/handheld, a moving or tracked mark). Not verified: LaMa with its real weights (the sandbox could not download them; the adapter is tested with a stand-in ONNX model for sizes, colour order, scaling and compositing, not for quality), GPU speed, strong parallax (one global motion per frame pair), and footage with heavy noise or motion blur. For a mark that never moves on a still camera, no software can recover pixels that are in no frame; LaMa produces a plausible fill, not the original.

**Licences.** LaMa is Apache-2.0. ProPainter (S-Lab 1.0), E2FGVI (CC BY-NC 4.0), MiniMax-Remover (CC BY-NC 4.0) and DiffuEraser (which needs the ProPainter prior) are non-commercial; they can be plugged in through `local`/`video` for non-commercial use or with the authors' permission, but none is bundled or recommended for a commercial service. Check the licence of any model file you supply.

**Model downloads** (`app/ai/weights.py`): HTTPS only (plain http only to this machine in development), SHA-256 required with the URL, redirects refused, size capped, written to a temporary file and moved into place only after the checksum matches; a present file is also checked against the checksum once per process.

### 6c. Exact overlay removal (Phase 14)

A logo laid over a video at partial opacity changes every pixel under it by a fixed rule, `shown = A*W + (1-A)*original`, with `A` the logo's opacity at that pixel and `W` its colour. If `A` and `W` are known the original is `(shown - A*W)/(1-A)`: exact, no model, nothing invented, and it works on a **still camera**, where no other frame shows the hidden pixels. `A` and `W` are the same in every frame while the picture behind them changes, so they can be measured from the clip (after Dekel et al., *On the Effectiveness of Visible Watermarks*, CVPR 2017):

1. The temporal median of the frames' gradients is the logo's own edges (the changing picture averages out). Integrating them (a conjugate-gradient Poisson solve, pinned to zero along the sides of the area that are not the picture's edge) gives the logo's contrast `Q = A*(W - median picture)`.
2. For each candidate overlay colour `W`, `A = Q/(W - median + Q)` per pixel. Two families of candidates are tried: grey (white, black, grey; one number) and tinted (the measured contrast splits across the channels as `Q_c = A*(W_c - picture_c)`, which leaves one free level). The right `W` is the one whose inverted frames show the least trace of the logo (the leftover-edge measure below, plus a clipping penalty); a coarse search over 0..255 is refined in steps of 2.
3. Before anything is changed, four checks must pass: the picture behind the mark changes enough to separate the logo from it; `A` peaks between 6% and 90%; the inverted frames keep little of the logo's edges (the median-over-time gradient along the logo's outline, minus its level on clean picture, must be under 30% of what it was before inverting) and few pixels are outside range; and the mark is the same in every sampled frame. A logo of several colours, a gradient or opaque logo, or a static picture, fails them and the render stops with the reason (or, under `auto`, falls through to reconstruction and says so).

Under `auto` the overlay is measured from a **survey**: before rendering, the pipeline decodes the video once and keeps up to 16-60 frames (as many as fit in 256 MB) spread evenly over the part of the clip where something is marked. The survey also checks, from the masks alone, that the mark never changes. Frames whose picture around the mark is within 2 levels (mean) of one already kept are skipped, so a still card shown for seconds can't outvote the moving picture (without this the opacity of a test logo was estimated at 0.50 for a true 0.40). The survey is a second front-to-back decode that stops after the last frame it needs; it is its own job stage, "Measuring the overlay" (`measuring`, between fetching and rebuilding; shown as skipped for engines that don't study the video first), with no percentage because there is no honest fraction, and it honours cancel. Used directly (`AI_PROVIDER=overlay`, or any caller without a survey), the engine falls back to the first frames. The rest of the video streams through. Only pixels inside the marked area change. **Fading, appearing or disappearing logos (Phase 19):** the measured model has a strength of 1; in each frame the engine finds the strength (0 to 1.6 times the model) that leaves the least edge energy along the logo's outline once inverted (a coarse grid, then a golden-section refinement, on brightness only). Within 0.12 of 1 it uses exactly 1, so a constant logo is inverted as before; at or below 0.10 the logo is taken to be absent and the frame is left untouched; anything between is inverted at that strength. Frames the first measurement finds absent are left out and the measurement is repeated on the rest (if at least 12 remain). The render's notes report how many frames were full, partly faded and absent. Measured on synthetic clips with a 45% white logo: fade in and out over 10 frames each, mean 31+ dB inside the logo and no frame under 26 dB (about 17 dB before); logo missing from the first 14 frames, those frames above 38 dB (untouched) and the rest above 33 dB; real x264 crf 12 encode of a 40% logo fading over 12 frames: above 28 dB.

**Measured** (`tests/test_overlay_engine.py`, `tests/test_reconstruct_render.py`): a 45%-opaque white "WATERMARK" over footage whose content changes every frame, against the true clean frames: 13.9 dB with the logo -> 39.1 dB after inversion on raw frames (peak opacity estimated 0.471 for a true 0.45, colour within 1 level), and 32.8 dB after a real x264 crf 12 encode (14.9 dB before). 30%-opaque white (17.4 -> 40.7 dB), 55%-opaque black (12.1 -> 37.7 dB) and a 45%-opaque red logo (15.5 -> 38.9 dB, colour estimated (31, 40, 254) for a true (30, 40, 255)) also invert. A bluish-orange logo at 45% (a colour close to the picture's own average, which hides the contrast) was refused by the leftover-edge check, after an earlier, looser check had let it through at 16.7 dB, worse than the untouched input: the same measure now decides between candidate colours, and the final check is relative (share of the logo's edge signal left), because video compression leaves a little of any logo in the picture and an absolute threshold either refused normal compressed video (1.8x) or let the bluish logo through (2.5x). Calibrated on the same logos: share left was 0.02 (white, raw), 0.09 (white, x264 crf 12), 0.17 (white, crf 23), 0.03 (red, raw), 0.26 (red, crf 12), 0.40 (red, crf 23), against 1.31 for the wrong bluish case; the limit is 0.30. Heavy compression (crf 23) of a red logo is therefore refused, not guessed. Refused rather than guessed (each tested): a picture that doesn't change, a picture that is static except for one small moving block, a fully opaque logo, a logo of two different colours, a mark that moves, a clip under 16 frames.

**Limits, stated plainly.** (a) The picture under the logo must change somewhere across the clip: a talking head in front of a fixed background under a logo is not separable, and the render says so. (b) One colour and one opacity pattern per pixel (its overall strength may change from frame to frame, see above); logos of several colours, gradients or animated logos are refused. The strength estimate is per frame and independent, so a faded frame is less exact than a full one, and a logo present in fewer than about half of the surveyed frames is not found at all. Colours close to the picture's own average give little contrast to measure and may be refused. (c) The measurement uses the surveyed frames only. A change in the mark's *shape or position* anywhere in the clip is found before rendering (the masks are known without decoding, and the first frame where the mark differs is always surveyed, so `auto` switches to reconstruction up front); a change in its *opacity or colour* between surveyed frames can't be seen from the masks and is only caught by the measurement's checks. A clip whose picture hardly changes anywhere under the mark is refused. (d) A logo that extends beyond the marked area keeps its outer part (only the marked area is ever changed). (e) Compressed video adds noise that `1/(1-A)` amplifies, so the restored area is a little noisier than its surroundings at high opacity. (f) Footage with strong motion blur or heavy compression in the sampled frames lowers the accuracy of the estimate; the checks above are what protect against a wrong one.

### Phase 21: live-stack re-run after Phases 16-20

Real API, Celery worker, scheduler, Redis, S3 stand-in, PostgreSQL 16 and the built web app, driven by Chromium. With `AI_PROVIDER=auto`, `e2e_removal.js` passed 5 of 5: engine status; a 40%-opaque white logo removed (22.2 dB with the logo -> 33.8 dB against the clean footage, after a real x264 encode and download); a fixed opaque logo on a still camera failing with the real reason and the marks kept; and a logo fading in and out over a 72-frame clip (23.3 dB -> 33.8 dB), where the job's stages were observed live as `measuring > reconstructing > verifying > saving` and the notes said the strength changed. With `AI_PROVIDER=dev`, `e2e_full.js` (19 of 19, including the stage list now carrying a skipped `measuring` step) and `e2e_security.js` (9 of 9) passed. Not run: anything with real LaMa weights, a GPU, or footage that was not synthetic.

## 7. Jobs (Phase 9)

Analysis and tracking keep their own small state machines (sections 5 and 5a). The long one, the final render, is a **job**: a row in `jobs` (id, project, type, status, stage, frames, error, settings, result, timestamps) that the API creates and a worker runs. The browser never waits on a request.

```
POST /render -> job queued, project PROCESSING -> Redis/Celery -> worker claims it -> runs render_restored_video
             -> uploads the result -> job completed, project COMPLETED
```

**Endpoints** (all scoped to the signed-in user; another person's project is a 404):
- `POST /api/projects/{id}/render` (202) with `{quality, container, output_height, keep_audio, acknowledge_third_party}`. Unknown fields are refused. 409 if there are no saved marks, the project isn't in MASKING/COMPLETED, or a job already runs (claimed with one conditional UPDATE, so two clicks make one job); 503 if the engine isn't ready (with the reason) or the queue is down (state reverted, job marked failed); 409 until the person confirms when the engine sends frames to a third party; 429 over `MAX_ACTIVE_JOBS_PER_USER` (default 2).
- `GET /api/projects/{id}/render` -> `{job}` (or `{job: null}`), the latest job.
- `POST /api/projects/{id}/render/cancel`.
- `POST /api/projects/{id}/reopen`: a COMPLETED project goes back to MASKING so its marks can be changed. The finished file stays (and `restore_project_status` returns the project to COMPLETED if a later render fails) until a new render replaces it. 409 for any other status.
- `GET /api/projects/{id}/render/events`: server-sent events. The first event is the current state, then one whenever it changes (while rebuilding that is about once a second, as speed and time remaining move), ending when the job does; a comment keep-alive every 15 s; closes after 30 min and the browser reconnects.

**What is reported, and how it is measured.** `stage` is written by the worker when it reaches `preparing` (fetching the original), `measuring` (only the automatic engine: the survey pass of section 6c), `reconstructing` (rebuilding and encoding, one step: the picture is encoded as it is rebuilt), `audio`, `verifying`, `saving`. `frames_done` counts frames that came out of the engine. `percent` is frames done over frames total and exists only while rebuilding; every other step has no honest fraction, so it is `null` and the interface shows the step, not a bar. `speed_fps` and `eta_seconds` come from frames finished since rebuilding began (nothing until 3 frames are done) and cover the rebuilding step only. A queued job reports `jobs_ahead`: queued jobs on the same queue created before it. The `stages` list (analyzing, mask, tracking, the five above) is derived from the job and project: tracking and audio show `skipped` with the reason when they weren't used. Progress writes are throttled to one a second (stage changes and the last frame always land); that same write is the cancel check.

**Queues.** `cpu` and `gpu`. `RENDER_QUEUE=auto` sends `local` and `video` engines to `gpu` and `dev`/`api` to `cpu`; a machine with no GPU can run one worker for both (`celery -A app.workers.celery_app worker -Q cpu,gpu`). The render task has its own limits (`RENDER_TIME_LIMIT_SECONDS`, default 7200, hard limit 300 s later). Messages are acknowledged after the task runs (`acks_late`) with one task prefetched.

**Safe to run twice.** The worker claims a job with `UPDATE ... WHERE status = 'queued'`, so a message the broker delivers again (a worker died before acknowledging) finds the job already taken and does nothing. There is no automatic retry of a failed render: a failure is reported and the person starts a new job. Storage calls retry inside boto3.

**Cancel, failure, lost workers.**
- Cancel while queued ends the job at once; a worker that picks the message up later skips it. Cancel while running sets a flag the worker sees within a second; it stops FFmpeg, removes partial files and sets `cancelled` itself. Deleting the project mid-render has the same effect (the job row is gone, the worker stops, and anything it uploaded after that is removed).
- After a job that makes no result the project returns to what it was: `COMPLETED` if an earlier result exists (kept), otherwise `MASKING`. A new result replaces the old one only after it is uploaded and recorded; the old file is deleted afterwards. Each job writes its own key, `outputs/{user}/{project}/{job}.{ext}`.
- Errors shown to people come from the pipeline and engines (written for people), or one of a few fixed messages (too long, storage trouble, worker lost, generic). Exception text, paths and keys go to the log only.
- A heartbeat thread in the worker writes `heartbeat_at` every `JOB_HEARTBEAT_SECONDS` (15), so a long step with no frames still beats. A running job with no heartbeat for `JOB_LOST_AFTER_SECONDS` (120) is reported failed ("the worker stopped") the next time it is read; a queued job nobody picked up for `JOB_QUEUE_TIMEOUT_SECONDS` (7200) is reported failed too. There is no scheduler for this; it happens on read. When the worker dies, its FFmpeg children die with it.

**Interface.** `RenderPanel` in the editor: engine name and device (from `GET /api/engine`), a development-engine notice, a consent checkbox when frames leave the server, quality / format / size (only sizes below the original) / keep-audio, then the stage list with a progress bar while rebuilding, speed and time remaining for that step, queue position while waiting, cancel, and the result's recorded facts and warnings. `useRenderJob` follows the job over server-sent events and falls back to asking every 2 s if the stream can't be kept; a late older update never overwrites newer progress or an ending. While a job runs, and on a finished video, the marks and Track are locked with the reason shown; "Edit marks" reopens it. The dashboard links PROCESSING and COMPLETED projects to the editor. Encoding is concurrent with rebuilding, so the spec's separate "Final encoding" line is part of "Rebuilding and encoding the video".

**Measured (real API, Celery worker, Redis and S3).** 1080p, 45 frames, tracked, AAC audio, development engine, CPU: completed in 13 s; progress went 2% -> 100% in order. Through the browser: a 960x540, 450-frame clip with a tracked moving logo rendered in 28 s with 26 distinct progress readings, in order, and the bar picked up where it was after a page reload mid-render; a cancel at about 8% stopped the job, kept the earlier result and left exactly one output object in storage; a worker killed with SIGKILL mid-render was reported failed after 12 s (the test used `JOB_LOST_AFTER_SECONDS=12`), the project returned to COMPLETED with its earlier result, no FFmpeg process remained, and a new worker took the next job.

Known limits: the database is the source of truth for progress and the stream polls it once a second per open page (fine for a few hundred pages; Redis pub/sub would cut the load); a job that is lost is only noticed when somebody reads it; no automatic retry; the API container runs migrations on start (fine for one instance, not for several); the compose file and image are validated with `docker compose config` but were not built here; the frontend has no container yet.

## 7b. Preview and export (Phase 10)

**API.** Both routes need the owner's cookie (401 anonymous, 404 for someone else's project, never a hint that it exists).

- `GET /api/projects/{id}/preview` returns signed links to the finished video and to the editor's preview copy of the original, the result's recorded facts (size, frames, fps, container, audio), the job's warnings, and `development_only`. 409 `NO_RESULT` when no render has finished; 410 `FILE_GONE` when the stored file is missing. A project reopened for editing still previews its last good result, because the output is kept until a new render replaces it.
- `POST /api/projects/{id}/export` returns a fresh signed link that is valid for 300 seconds and tells the browser to save the file (`Content-Disposition: attachment`), with a safe name made from the project name (`Beach day1-restored.mp4`: accents folded to ASCII, unsafe characters dropped, 80 characters at most, `video` if nothing is left). It is POST because each call issues a new link and writes a log line, `export link issued`, with project, user and job ids and the size. No video content or file name is logged.
- Playback links last `PRESIGN_TTL_SECONDS` (1 hour). The video bytes never pass through the API: the browser reads storage directly.

**Result page** (`/result/{id}`, behind the same sign-in gate as the editor, not indexed): "Your restored video is ready." with the recorded facts, any warnings, and a plain note when the development engine made it ("not AI: it only blurs"). Buttons: Download MP4, Edit marks, Create another, Delete project (asks first; nothing is claimed deleted until the API answers, then the page leaves for the dashboard).

**Compare view.** Two stacked videos: the restored one (master: sound, time, controls) and the preview copy of the original (silent follower) cut off at a draggable divider. Modes Compare / Original / Restored; zoom 1x, 2x, 4x with drag to pan (the cut stays where the divider is on screen while the picture moves); play and pause, scrub, one frame back or forward, mute when the video has sound. The divider is a slider (arrow keys 2%, Shift 10%, Home and End), Space plays, left and right arrows step a frame. The follower is corrected whenever it is more than 1.5 frames from the master while playing (0.5 paused), on every video frame where the browser supports it.

**Expired links.** A video that fails to load makes the page ask for new links and load both videos again at the same position, at most three times a minute; then it says so. Coming back to a tab whose links are near the end of their life refreshes them too (unit-tested timing rule; not exercised in the browser run).

**Measured (real API, worker, S3 and Chromium).** 320x568 clip with AAC audio, 60 frames: after a render the editor links to the page; the two videos stayed within 9 ms of each other while playing, and were identical when paused and after seeking; the divider moved by mouse and keyboard; pixels left of the divider differ from the restored picture inside the marked area; zoom and pan worked and reset; the downloaded file was named `small-restored.mp4` and probed as H.264 320x568 with AAC and 60 frames; a blocked video request was recovered by new links; no sideways scroll at 390 px; another account got "Project not found"; deleting needed confirmation and left no object in storage (4 before, 0 after).

Limits: the "Original" side is the editor's preview copy (at most 720p, no sound, re-encoded), so next to a full-size result it looks softer, and the page says so; comparison is by position in time, so a video with variable frame rate can be a frame off in the follower for a moment; only MP4 and WebM results can be previewed by the browser; "Download MP4" saves whatever container was rendered (WebM if that was chosen) under that container's extension.

## 8. Security, cleanup and logging (Phase 11)

What exists, what each piece does, and what it does not do.

### Access and ownership

Sign-in is a signed JWT in an HttpOnly, SameSite=Lax cookie (Secure outside development). Every project, job, mask, media, preview and export route loads the project by `(id, owner)`: someone else's project is a 404, the same answer as for one that does not exist. Writes from a browser origin that is not in `CORS_ORIGINS` are refused (403). There is no CSRF token; SameSite=Lax plus that origin check is the protection. No admin routes exist yet, so `require_admin` is unused.

### Rate limits (`app/ratelimit.py`)

Fixed windows. Redis in production (shared by every API process); an in-process counter in development and tests only. The app refuses to start outside development with `RATE_LIMIT_ENABLED=false` or `RATE_LIMIT_STORAGE=memory`. If Redis cannot be reached the limiter lets requests through and logs one error, because Redis is also the job queue and refusing everything would turn a small fault into an outage. Over a limit the answer is 429 with `Retry-After` and a sentence such as "Please wait 4 minutes and try again."

| Rule | Limit | Keyed by |
|---|---|---|
| `api_ip` (every API call) | 900 / minute | client address |
| `login_ip` | 30 / 15 min | client address |
| `login_account` (failed attempts only; cleared by a correct sign-in) | 8 / 15 min | hashed email |
| `register_ip` | 10 / hour | client address |
| `project_create` | 30 / hour | user |
| `upload_urls` | 600 / hour | user |
| `mask_save` | 600 / hour | user |
| `tracking_start` | 60 / hour | user |
| `render_start` | 30 / hour | user |
| `export_link` | 60 / hour | user |

The client address is the socket address unless `TRUSTED_PROXY_COUNT` says how many proxies sit in front, then the matching `X-Forwarded-For` entry. Behind a shared network address (an office, a mobile carrier) several people share the per-address limits.

### What an uploaded file may be

- Size, duration and pixel limits (`MAX_UPLOAD_SIZE` default 2 GiB, `MAX_VIDEO_DURATION_SECONDS`, `MAX_VIDEO_PIXELS` default 4096×2304, `MIN_FREE_DISK_MB`), enforced by the API at upload completion and again by ffprobe in the worker, which reads the real properties. The extension and the declared type are never trusted.
- Upload completion reads the first bytes of the stored object and refuses anything that is not a known video container (MP4/MOV, Matroska/WebM, AVI, MPEG-TS/PS, FLV, ...). The project is marked FAILED with a readable message and a log line; no worker is started for it.
- Every read of a user file (ffprobe, proxy, frame extraction, audio, render) names the demuxer it sniffed (`-f mov` / `-f matroska` ...) and passes `-protocol_whitelist file`, so a text file that ffprobe would otherwise treat as a playlist or concat list cannot make FFmpeg open other files or addresses. Checked against `ffconcat` with a relative path (ffprobe parsed it as `concat` before the forcing, refuses it now).
- Storage keys are server-made UUIDs; the person's file name is only stored as a label and is dropped from logs. FFmpeg is always called with an argument list, never a shell. Working folders are `cf-*` temporary directories removed when the job ends.
- Request bodies over `MAX_REQUEST_BODY_BYTES` (2 MiB) are refused; video bytes never pass through the API.

### Signed links

Storage links are presigned S3 V4 URLs: 1 hour for playback, 300 seconds for a download (with `Content-Disposition: attachment`). Nothing is public. Expired links are recovered by the page asking for new ones (3 per minute).

### Browser and API headers

The Next app sends a Content-Security-Policy (`default-src 'self'`; `connect-src` limited to the API and, via `NEXT_PUBLIC_STORAGE_ORIGIN`, the storage address (read at build time); `object-src 'none'`; `frame-ancestors 'none'`), `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, a `Permissions-Policy` that turns off camera, microphone, geolocation and payment, and HSTS in production. `script-src` keeps `'unsafe-inline'` because Next writes small inline scripts into each page; nonces would make every page dynamic. In production, if `NEXT_PUBLIC_STORAGE_ORIGIN` is empty, media and uploads may go to any https address. The API sends `nosniff`, `X-Frame-Options`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` (unless the route chose otherwise), a `default-src 'none'` policy, HSTS in production, and `X-Request-ID`. The interactive API docs are off outside development.

### Retention and cleanup (`app/services/cleanup.py`)

Every project has `expires_at`. It is set when the project is created and pushed to "now + `VIDEO_RETENTION_HOURS`" (48) when the upload completes and again when a render finishes, so a finished result is downloadable for the whole period. Two things enforce it:

1. A Celery beat task (`cleanup_expired`, every `CLEANUP_INTERVAL_SECONDS`, 900) deletes the original, preview, thumbnail, every render output and any unfinished upload parts for each due project, then marks the project EXPIRED and drops its jobs and marks.
2. A lazy check on every read: a project past its deadline answers 410 "expired" to the editor, result and media routes, and is expired on the spot, so access stops at the deadline even if the sweep is late.

The EXPIRED row stays so the dashboard can say what happened ("The files were deleted after the retention period"); deleting it from the list still works. If storage refuses a delete, nothing is marked and the next sweep retries, so the interface never says "deleted" for files that still exist. A project a worker is still using (live job, tracking, analysis) is skipped until the work ends or has been silent longer than any job may run. The worker also removes leftover `cf-render-*`, `cf-tracking-*` and `cf-analysis-*` folders older than the longest job limit plus an hour, at start-up and in every sweep. Run exactly one `beat`.

The privacy notice shown to people says that temporary processing files are deleted according to the retention policy. Nothing in the app claims immediate deletion.

### Logging (`app/logging_config.py`)

One JSON object per line in production (readable text in development). Every line carries the ids bound for the request or task (`request_id`, `user_id`, `project_id`, `job_id`, `task_id`); jobs log `duration_ms`. Each request writes one access line (method, route template, status, duration, user) without the query string or cookies; health checks are logged only at DEBUG. Scrubbing is applied to every field and message: names that look like secrets (password, token, cookie, signature, key ...) are replaced, upload file names are dropped, URL query strings are removed and `X-Amz-*` parameters are redacted, exception text included. Events logged: user registered, login succeeded and failed (never the email or password), upload started, completed or refused, analysis started and finished, tracking and render started, finished, failed or cancelled, export link issued, project expired or deleted, the retention sweep result, and rate-limit refusals.

### Messages people see (spec section 33)

| Situation | Message |
|---|---|
| Not a video / unreadable | "We couldn't read this video. The file may be corrupted. Please upload another copy." |
| Too large / too long / too many pixels | The limit and the file's value, in words |
| Worker out of memory or disk | "The server ran out of memory or disk space while working on this video. Try a shorter clip or a smaller output size, then try again." |
| Storage failure | "The video file couldn't be read from or saved to storage. Please try again in a moment." |
| Database down | "We can't reach our database right now. Please try again in a moment." |
| Any other unexpected error | "Something went wrong on our side. Please try again in a moment. If it keeps happening, quote reference {request id}." (the exception text is never sent) |
| Too many requests | "... Please wait N minutes and try again." |
| Expired project | "This project has expired and its files were deleted. Upload the video again to start a new one." |
| Worker stopped | "Rendering didn't finish (the worker stopped). Please try again." (tracking has its own wording) |

### Containers

`api`, `web`, `beat`, `worker` and `worker-gpu` run with `no-new-privileges`, all capabilities dropped and a process cap; workers have a memory limit (`WORKER_MEMORY_LIMIT`, 8g). Validated with `docker compose config` only; the images were not built or run here. **Web image (Phase 22):** `docker/frontend.Dockerfile` builds in three stages (install, build with `NEXT_OUTPUT=standalone`, run as the `node` user from the traced standalone server and the static files only). `NEXT_PUBLIC_*` are build arguments because they are compiled into the pages and the security policy. The `.dockerignore` no longer excludes `frontend/` (it keeps `node_modules`, `.next` and local env files out). Checked without Docker by repeating the image's steps: `npm ci`, the standalone build, then running `node server.js` from a directory holding only the standalone output and `.next/static` (69 MB): the page and its CSS were served with the full set of security headers. Not checked: the build inside Docker itself (no base image could be pulled in the sandbox), the `node` user and file ownership, the container health check, or the compose network between `web` and `api`.

### Verified and not verified

Automated: cleanup (16 tests), rate limiter incl. a real Redis (28), logging and error handling (19), authorisation matrix over every route in the OpenAPI schema (51), plus the earlier suites. In a real browser against the live API, worker, beat and S3: security headers present, the whole flow works under the policy with no blocked request, the dashboard states the deletion date, an account locks after repeated wrong passwords with a plain message, a project made due is deleted by the scheduled sweep (files gone from storage, status EXPIRED), expired pages say so, and the logs contain request ids and durations and none of the secrets or file names checked for.

Known limits: moto (the S3 stand-in) does not enforce presigned expiry or signatures, so expiry is verified at the contract level only (the URL carries `X-Amz-Expires` and a V4 signature); registration says when an email already exists (409, rate limited); the limiter fails open if Redis is down; objects orphaned by a crash between a storage write and its database commit are not found by the database-driven sweep, so also set a lifecycle rule on the bucket; no malware scanning; FFmpeg decoders still parse hostile bitstreams (limited by the demuxer allow-list, container limits and keeping FFmpeg patched); logs scrub links and secret-like fields but not arbitrary text inside an exception; "export completed" is logged as "export link issued" because the server cannot see a download finish; tracking cannot be cancelled; the container hardening is untested.

### Phase 23: upload scanning and bucket lifecycle

**Malware scan (optional, `MALWARE_SCAN=clamav`).** After the analysis worker has downloaded an upload, and before ffprobe or FFmpeg open it, the file is streamed to a clamd server (`CLAMAV_HOST`, `CLAMAV_PORT`) with the INSTREAM command (`app/security/malware.py`; the scanner needs no access to our disk or storage). Outcomes: clean -> analysis continues; flagged -> the project fails with "This file was blocked because the security scan flagged it. It was not processed." (the signature name goes to the log, not to the person); the scanner cannot answer (down, timeout, or it refused the stream, for example because clamd's `StreamMaxLength` is smaller than the file) -> the project fails with "We couldn't run the security scan ... please try again", **never** treated as clean, and the person can retry analysis later. A flagged file is not deleted at once: it stays in storage, is never served (only the generated preview is), and is removed with the project at its retention deadline or when the person deletes the project. Off by default; the compose file has a `clamav` service under `--profile scan` with the size limits raised past `MAX_UPLOAD_SIZE`. clamd silently skips files above its `MaxFileSize`, so that limit matters as much as the stream limit.

**Bucket lifecycle (`python -m app.storage.provision`).** The application deletes files at the retention deadline; the bucket now also enforces a backstop: objects under `uploads/`, `derived/` and `outputs/` expire `ceil(VIDEO_RETENTION_HOURS / 24) + STORAGE_LIFECYCLE_MARGIN_DAYS` days after they were written (4 days by default), and unfinished multipart uploads are discarded after `STORAGE_ABORT_UPLOADS_DAYS` (2). The command replaces the bucket's lifecycle configuration with exactly those four rules, is safe to repeat, and never creates the bucket. In development the API applies it at start-up; in production run it once per environment and again after changing retention (the storage key needs permission to set lifecycle configuration). Expiry granularity is one day in S3, so this is a safety net, not the main cleanup.

**Not done: a nonce-based script policy.** The page policy keeps `script-src 'self' 'unsafe-inline'` because Next.js writes small inline scripts into every page. A nonce per response means every page is rendered on demand instead of served as static output, which costs speed and load for a hardening gain that is small here (no user-supplied HTML is ever rendered, `object-src`, `base-uri` and `frame-ancestors` are locked down, and React escapes text). Revisit it if user-generated markup is ever shown.

**Verified (automated):** a stand-in clamd that speaks the real INSTREAM wire format (length-prefixed chunks, zero terminator, NUL-terminated reply): a 300 KB multi-chunk file streams whole and passes; the EICAR test string is reported with its signature; a refused stream, a silent server and a closed port are each "unavailable", not "clean"; through the real upload and analysis path a flagged upload fails with the message above, a clean one reaches MASKING, and a scanner that is down fails closed and succeeds on retry once it is back. Lifecycle: the four rules are set, repeating changes nothing, retention changes the expiry, and every key the app writes falls under a covered prefix (moto stores the rules; it does not run them). **Not verified:** real ClamAV (not installable or pullable here), clamd's behaviour at its size limits, the `clamav` compose service, the lifecycle rules on a real S3 or MinIO, and no malware scanning of the generated preview or render output (they are produced by our own FFmpeg from a scanned original).

### Phase 24: load and capacity

Measured on this sandbox (2 CPU cores, 8 GB RAM, no GPU) with the local stack (`scripts/load/load_test.py`, rate limiting off, PostgreSQL 16, one API process, the `auto` engine). Treat the numbers as the shape of the system on a small machine, not as a capacity guarantee.

**API reads** (20 signed-in people hammering the project list, engine status and account for 15 s, with the load generator on the same machine): 215 requests/s, p50 about 90 ms, p95 about 140 ms, p99 about 200 ms, no errors. **Password hashing is the expensive request**: 20 simultaneous sign-ups took about 3.2 s each and 20 simultaneous sign-ins about 2.1 s, because they queue for two cores. A burst of logins costs CPU the API shares with everything else.

**Whole pipeline** (6 people at once: upload, analysis, marking, render, download; a 60-frame 960x480 clip with a 40% logo over a mark of 774x264 px):

| Worker processes | 6 videos finished in | Throughput | One render takes | Worker memory peak |
|---|---|---|---|---|
| 1 | 252 s | 1.4 videos/min | 41 s | 1.5 GB |
| 2 | 152 s | 2.4 videos/min | 48 s | 2.5 GB |

The queue behaves as designed: jobs start one after another (or two at a time), nobody's job fails, and a waiting person sees "queued" and their position. Two workers on two cores gave 1.7x the throughput while each render got 18% slower, so on a machine of this size two is the useful limit.

**Where a render's time goes.** Measuring the overlay is a one-off cost per video that depends on the size of the marked area, not on the length of the video: about 30 s for 774x264 px (60 sampled frames), then roughly 0.1 s per frame to invert and encode at 960x480. At 1920x960 with a 1550x530 mark the same render took 158 s (about 140 s of it measuring) and the process peaked at 3.0 GB, roughly 3.6 KB of memory per pixel of the marked area; the cap on that area (`MAX_ROI_PIXELS`, 1.5 million px) therefore means about 5.5 GB for a single worker. Set `WORKER_MEMORY_LIMIT` with that in mind; the default 8g fits one render at the cap, not two.

**Changes this produced.** (1) The first run exposed a false refusal: the 960x480 clip (enlarged footage) was refused by the "edge energy along the logo's outline" check at 1.62 against a limit of 1.5, while 90% of the logo's edge signal was in fact gone. Measured on good and bad cases (white and red logos, raw and enlarged 2x and 3x: 1.12-1.28; the hard bluish colour, which must be refused and is refused by the leftover check: 1.21; two colours, opaque and still pictures are refused before this check), the ratio never separates good from bad, so the limit was raised to 2.5 and the leftover check does that job. A test now covers enlarged footage. (2) Time over medians (sorting instead of `np.median`) and weighted sums instead of boolean indexing: a 960x480 render went from 48 s to 38-41 s with identical results.

**Not measured:** more than 6 simultaneous people, more than 2 workers, a GPU, a network between the API and storage, real S3, uploads of hundreds of megabytes (the test clip is 2 MB, so upload time is not representative), and long videos (the per-frame cost is small, but 120 s at 30 fps is 3,600 frames, which at 0.1 s each is about 6 minutes of inversion plus the one-off measuring). Video analysis of large uploads shares the same worker queue as renders in this setup, so a long render delays other people's analysis; with `RENDER_QUEUE` the render and analysis queues can be split. Reproduce with `scripts/load/make_load_media.sh` and `scripts/load/load_test.py`.

## 9. Testing (Phase 12)

Four layers, each run for real, plus what none of them can show.

**Backend (pytest, `backend/tests`).** Authentication, authorisation, upload validation, project and job creation, API validation, probing, mask generation, tracking, the render pipeline (audio kept, encoding, output verification), result and export, cleanup, rate limits, logging, and a matrix that calls every route in the OpenAPI schema as a stranger and as the wrong person. It needs `ffmpeg`/`ffprobe`; storage is mocked (moto) and Redis is real where a test needs it. The default database is SQLite. Set `TEST_DATABASE_URL=postgresql+psycopg://user@host:port/db` (an empty UTF-8 database; the tests drop and recreate its `public` schema, so never point it at data you want) and the same suite runs on PostgreSQL, the production database, including a check that the migrations build exactly what the models describe and downgrade to nothing. Last results: SQLite 547 passed and 1 skipped (the PostgreSQL migration case), PostgreSQL 548 passed.

**Frontend (vitest).** Library logic in Node (upload part handling, masks, tracking, render state, compare, retention, video validation, API client) and components in a DOM (`components/__tests__`, Testing Library): the upload screen (notices, refusal of a text file, a disguised `.exe`, an image, an empty file, an oversized file and several dropped files, a failed upload, resume only once the project exists, the server's reason for an unusable video), the mask tools (tool selection, size control, undo and lock), the region list (selection, rename on Enter, clear, lock), the timeline (position, seeking, stepping, keyframes, lost-tracking markers, lock), the project list (loading, load failure with retry, every status label, which links each state offers, failure reasons, expired projects, confirmed delete and a failed delete) and the sign-in and sign-up error states (lockout message, short password, taken email, a network failure). 163 tests; `tsc --noEmit` and `next build` are clean.

**Browser journeys (`scripts/e2e`, Playwright against the real stack).** `start.sh` brings up Redis, an S3 stand-in, the API, a Celery worker, the scheduler and the built web app on fresh data (SQLite, or the PostgreSQL in `DATABASE_URL`); `run.sh e2e_full.js` then walks one clean-state journey and `run.sh e2e_security.js` checks the security headers, the policy against the real flow, lockout and scheduled deletion. The full journey, 19 steps, on PostgreSQL: landing page and sign-in gate, sign up / out / in, a text file and an oversized upload refused, a non-video called `.mp4` refused by the server with a readable reason, a `../../etc/passwd.mp4` file name stored as `passwd.mp4` with no unsafe storage key, upload, rectangle and polygon marks surviving a reload, a render whose stages run analyze, mask, track, preparing, reconstructing, audio, verifying, saving, the marks locked afterwards (409), the result page with its development-engine notice, the downloaded file probed (H.264, 320x568, 60 frames, AAC) and compared with the source frame by frame (mean difference inside the marks 85 and 47, everywhere else 0.9), a second account getting 404 on ten routes of the first account's project, anonymous 401, a forged cookie and a forged origin refused, the sweep deleting an overdue project's files, 410 and the "expired" pages, deletion leaving no row and no stored file, logs with ids and no private data, and no console errors. `make_media.sh` makes the two small input files.

**Spec section 59, one by one.**

| Item | Where it is checked |
|---|---|
| Unauthorized project access | `test_security_api` (every route, stranger and other person), journey (10 routes, 404; 4 routes, 401) |
| Invalid file uploads | `test_projects` (unsupported extension, empty or negative size, oversize, wrong declared size, missing parts, unreadable contents, duration and resolution limits), `test_security_api` (non-video bytes behind a video name, playlist and concat files), `UploadDropzone.test`, journey (text, disguised, server-refused non-video) |
| Malicious filenames | `test_projects`, `test_security_api` (both slash styles, NUL, a right-to-left override, 240 characters, a CRLF header injection, percent-encoding, SQL text, HTML, a reserved device name, and only three extensions accepted), journey |
| Path traversal | file names never reach a path (keys are UUIDs); `test_security_api`; journey inspects the stored keys; forced demuxers against a hostile `ffconcat` file in `test_security_api` |
| Oversized uploads | API 413 with the limit in words (`test_projects`, journey), browser refusal (`UploadDropzone.test`), request body cap (`test_middleware`) |
| Expired URLs | the link carries a V4 signature and `X-Amz-Expires`, the lifetimes are asserted (`test_result_api`, `test_security_api`); the page recovers from a dead link (Phase 10 browser script, not kept in the repo). **Not enforced here:** the S3 stand-in does not reject an expired signature, so the storage server's own refusal was never observed |
| Expired projects | `test_cleanup`, journey and `e2e_security` (sweep by the scheduler, then 410 and the pages) |
| Rate limiting | `test_ratelimit` (including a real Redis), `test_security_api`, `AuthForm` error state, `e2e_security` (lockout message in the browser) |

**What was found while testing, and fixed.** Tests run in one process lost the application's log output after the migration test ran, because Alembic's logging setup silenced existing loggers; a production-configuration test broke when production began to require rate limiting; the browser run showed the production page blocking uploads to plain-http storage because the allowed storage address is fixed at build time; a vitest trap where a function returned from `beforeEach` is run as teardown made two upload tests fail for the wrong reason; and the PostgreSQL run showed that tests which build their own app left rows behind in a shared database (each test now starts from an empty schema).

**Not covered, and why.** The Docker images were never built (a daemon started here, but no registry could be reached to pull a base image), only `docker compose config`; a real GPU model was never run, so the only engine exercised end to end is the development engine, which blurs and says so; Redis, S3 and PostgreSQL were local stand-ins or local installs rather than managed services; no load or concurrency test beyond one worker and a few users; no cross-browser run (Chromium only); no accessibility audit beyond the labelled controls the component tests rely on; no penetration test.

### Phases 13-15 verification

Backend 591 passed, 1 skipped (SQLite). Live stack (real API, Redis, Celery worker, S3 stand-in, built web app, Chromium): `e2e_full.js` 19/19 on PostgreSQL 16; `e2e_removal.js` 4/4 with `AI_PROVIDER=auto`: a 40%-opaque logo over changing footage went from 22.2 dB to 33.8 dB against the true clean video in the marked area after the full encode/render/download path, the result page showed the "measured from the video itself and inverted" note and no development notice, and a fixed opaque logo on a still camera failed with "hidden in every nearby frame..." shown in the editor with the marks kept. Not covered: LaMa with real weights, a GPU, footage that is not synthetic, other browsers.

## 9b. Local mode (Phase 25)

`LOCAL_MODE=true` (set by `python run_local.py`) swaps every external service for something on this computer, behind the
same interfaces, so the code paths above the swap are the ones the production setup runs:

- **Database**: SQLite (WAL, `busy_timeout`), migrated with Alembic at start-up.
- **Storage**: `LocalStorage` implements `ObjectStorage` on a folder. Upload parts, completion (ETag check), ranged downloads
  and deletes go through `app/api/local_files.py`; links are HMAC-signed, bound to one operation and expire; keys cannot leave
  the storage folder; an oversized part is refused and leaves nothing behind.
- **Jobs**: `app/workers/local.py` runs the same task functions on a thread pool (`LOCAL_WORKERS`, default 1) with a timer
  thread for the retention sweep. On start-up `recover_interrupted()` marks work that was queued or running when the app
  closed as failed (renders, tracking, uploads still being checked), because nothing on threads survives a restart.
- **Secrets**: the session secret is generated once into the data folder (mode 0600). `AI_PROVIDER` defaults to `auto`.
- **Memory**: the overlay measurement subsamples its frames to `OVERLAY_MEMORY_MB` (never below 16). Measured peak for a
  1550x530 px mark in 1920x960 video: 3.0 GB before, 2.3 GB after; unchanged at 256/512 MB because 16 frames is the floor.
- **Not available locally**: hard job time limits, several people at once, HTTPS (localhost over http only; non-localhost
  `API_PUBLIC_URL` is refused).

Verified: backend tests (`tests/test_local_mode.py`), a real-browser run of `e2e_removal.js` against the stack started by
`run_local.py` from a clean checkout (5/5, including download and playback under the page's content security policy). Not
verified: Windows/macOS, a real LaMa model, Docker images.

## 9c. Interface redesign (Phase 26)

Front end only. No API route, database column, job stage or removal engine changed. Flow: Upload → Select → Restore → Export (`components/StepIndicator.tsx`).

- **Design tokens** live in `app/globals.css` (`@theme`): background `#0B0B0C`, surface `#151516`, border `#29292B`, text `#F5F5F5` / `#A1A1AA`, accent `#B7F34A`, Inter Variable. Shared classes: `.surface .btn .btn-primary .btn-ghost .btn-icon .field .seg`. Motion is 150-250 ms and switched off under `prefers-reduced-motion`.
- **Pages:** landing (`Hero`, `SampleCompare`, `HowItWorks`), `/dashboard` (upload), `/projects` (grid, search from 6 projects, delete), `/editor/[id]`, `/result/[id]`.
- **Editor:** `SelectionToolbar` (Rectangle, Brush, Eraser, Automatic detection, plus Adjust and Polygon), `MaskCanvas`, `VideoTimeline`, `RemovalSettings`, `ProcessingProgress`. Undo/redo/reset, saving and tracking are the existing logic. "Remove Watermark" is disabled until a region has real size.
- **Settings mapping (no invented options):** Quality Balanced / High = the existing quality (10 / 16 frames studied). Processing mode "Fast" sends quality `fast` (6 frames, faster encode); "AI reconstruction" uses the chosen quality. The offline fill model is still chosen by the server.
- **Processing screen:** the nine real backend stages are grouped into four (`lib/render.ts: groupStages`). A percentage or time estimate is shown only when the backend reports one; otherwise the bar is indeterminate and shows no number.
- **Automatic detection is a heuristic, not an AI model** (`lib/detect.ts`, `lib/detectVideo.ts`). It samples 12 frames of the preview in the browser and looks for pixels that stay still while the picture around them moves (plus a see-through plate's outline). It only works for a still overlay over moving footage, can miss or mis-box things, and never selects anything: the user chooses "Use selection" or "Adjust manually". If the browser cannot read the video's pixels, or the scene barely moves, it says so.
- **Editor preview has no sound** (it is the proxy video), so the editor's volume control is disabled with an explanation. The result screen has working volume, zoom and fullscreen.
- **Automatic mode (see AUTOMATION.md):** when the analysed project has an `automation` record the upload screen jumps to `/process/{id}`, which shows what was found and the real job progress and moves to the result when it is done. With `AUTO_PROCESS=false` the earlier manual flow (preview, "Continue to selection") is used. Finished jobs navigate to the result automatically.
- **Landing sample:** two real looping clips in `frontend/public/samples/`. They are a generated colour field with a semi-transparent "LOGO" plate; `after.mp4` is that clip processed by this app's own pipeline. Rebuild with `scripts/make_landing_sample.sh <dir>` and `scripts/drive_landing_sample.py before.mp4 after.mp4` (needs a running local API). It is a demonstration clip, not a customer result.
- **Tests:** unit tests for detection, grouping, settings mapping and the new components (`npx vitest run`). Looked at in a real browser on Linux only; Windows, macOS and phones were not tried on real devices.

## 10. Roadmap

All twenty-six phases are built (Phase 26: interface redesign; Phase 25: local mode with no external services, a one-command launcher and a memory budget; Phase 13: reconstruction engine and LaMa adapter; Phase 14: exact overlay inversion and `auto`; Phase 16: coloured logos; Phase 17: overlay measured from frames across the whole clip, mark changes found before rendering; Phase 19: logos that fade or come and go; Phase 20: the survey is a visible job stage; Phase 21: live-stack re-run; Phase 22: web image and compose service; Phase 23: upload scanning and bucket lifecycle; Phase 24: load test, a false refusal fixed, faster measuring). Remaining work: validating LaMa with real weights on a GPU, logos of several colours or varying opacity, parallax-aware alignment, and, outside the spec's first implementation: building and running the images, a nonce-based script policy (declined, see Phase 23), and load tests beyond the small-machine numbers of Phase 24 (more people, more workers, a GPU, real storage, long videos).


## Resolution (720p to 4K)

Uploads and finished videos can be 720p, 1080p, 1440p or 4K (up to 4096×2304 px by default); the finished video keeps the
source's size, frame rate and orientation unless a smaller size is asked for. What changed, how each piece was checked with
FFprobe on real clips, the benchmark numbers and what could not be verified without a GPU are in `docs/RESOLUTION.md`.
Additional exports (`/api/projects/{id}/exports`, section 7b) are one re-encode of the finished file and are only offered at
sizes smaller than it. The editor still works on a 720 px preview copy by design (browser memory); masks are fractions of the
frame, so the full-resolution render is never affected by it.

## Performance

Measured stage times, the optimizations (fill reuse, exact skip of futile footage search, resident warmed-up LaMa) and the honest status of the 10–20 s target are in `docs/PERFORMANCE.md`. Benchmark tools: `scripts/bench_pipeline.py`, `scripts/quality_compare.py`, `scripts/bench_lama.py`.
