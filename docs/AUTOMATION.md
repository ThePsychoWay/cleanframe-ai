# Automatic mode: upload, wait, download

You upload a video. The server finds the overlays on its own, removes them, checks its own result and offers the file.
Nobody draws, selects or clicks "remove". The editor is still there (**Edit selection**) for the cases the detector
could not decide, and a person's own marks always win over automatic ones.

Everything below is measured on the 2-core, no-GPU machine this was built on. Where something is a limit, it says so.

## 1. Flow

```
upload ─▶ analysis (probe, preview, thumbnail)
        ─▶ detection (survey → candidates → follow → judge)   saved as an ordinary mask (source "auto")
        ─▶ render (engine chain, LaMa only where the video can't supply the picture)
        ─▶ quality check (original vs result, per region)
        ─▶ result page (preview, download)
```

* `/process/{id}` is the screen: what was found, the real job progress (frames done, speed), then the result.
* Statuses and job fields are the real ones (`ANALYZING → MASKING → PROCESSING → COMPLETED`); nothing is simulated.
* If nothing is found the project stays in the editor with the plain statement "nothing found". A video is never
  presented as processed without a removal behind it, and the original is never returned as the result.
* If the engine isn't ready (or would send frames to an outside service, which needs consent) the render is not
  started and the project says why.

## 2. How detection works (no neural detector, no OCR, no API)

1. **Survey**: the whole timeline decoded once at ≤384 px wide (stride-sampled beyond 1200 frames).
2. **Evidence**: a high-pass detail stack; a pixel that holds the same detail while the picture around it changes is
   overlay evidence (mean² / mean-abs over windows: whole clip, per shot, ~1 s sliding). Activity gating keeps still
   pictures from producing evidence.
3. **Candidates**: thresholded evidence → connected components → tightened to the strong core → template and support.
4. **Follow**: normalised cross-correlation tracking with a template that is sharpened from the aligned frames; local
   search at nearby scales, a global multi-scale search when the mark is lost (so marks that jump, reappear or change
   size are re-identified), copies of a proven mark elsewhere, and hidden keyframes for the frames where it is absent.
5. **Judge** (the reason is written down either way): steady enough to be an overlay → not a title/caption → does the
   picture around it move (proof) → otherwise, for a small isolated mark in a corner of a part of the picture that never
   moves, "judged by look and position only" (flagged).
6. **Mask**: a polygon hugging the proven support with keyframes at pose changes, `hidden` keyframes while it is off
   screen, jumps kept as jumps. It goes through the same render/tracking/export code as hand-drawn marks.

7. **Rescue of steady lettering** (`detect/glyph.py`, `_persistent_lettering`): on compressed footage a real mark is often
   rejected by the per-frame "steady" test because its correlation wobbles with the codec. A candidate that is lettering
   (≥ 6 separate strokes, coherent stroke orientation), is on screen for ≥ 50 % of the clip, scores strongly as an overlay,
   matches its own template and **does not wander** (95th-percentile position change ≤ 4 survey px: a painted overlay stays
   put, a scene object that looks like lettering drifts with the camera) is accepted. Without the wander rule a plate edge
   in the Coca-Cola test clip was removed.
8. **Refine** (`detect/refine.py`): the mask is grown to the whole mark where the detector only caught its strong core
   (bounded by aspect so it cannot swallow scene texture), a static mark is held at one position, and frames where the
   picture is too washed out to show the mark are kept. "Washed out" now requires the picture *around* the mark to lose
   detail as well; before, a mark that had really disappeared from a plain background was painted over on every frame.
9. **Drifting marks** (`detect/drift.py`): a mark that slides slowly over a part of the picture that does not move is found by
   differencing against the still background, linked into a smooth path, and masked per frame with a rotated rectangle fitted
   to the residual. It is accepted only for a smooth, consistent path that keeps its look.

Captions: text wider than 42 % of the frame, or large area, or a line of text ≥ 20 % wide in the middle of the picture
is treated as the video's own title and left alone. (An earlier run of this feature erased a centred caption and three
year labels on a real clip; those two rules and the corner-only rule for look-based marks are the fix, covered by tests.)

Settings: `AUTO_PROCESS` (default on; off = detection is skipped and the app behaves as before), `AUTO_APPEARANCE`
(default on; off = remove only what motion proves).

## 3. Generated fill: steadiness and seams

When no other frame shows the pixels behind a mark (a mark that sits still on a camera that doesn't move, or a mark that
drifts over moving content), they are generated by LaMa, one frame at a time. Two things were visible in that output and
are now handled in `app/ai/temporal.py` / `app/ai/consistency.py`:

* **Shimmer.** Each frame's patch is a new guess. The generated pixels (only those, never footage-rebuilt ones) are blended
  with the previous frame's finished patch, carried along with the picture's motion. The weight is 0 at the patch edge and
  0.75 six pixels inside, so the edge keeps following the real picture. It is skipped when the real pixels around the patch
  show that the previous frame no longer lines up, and across cuts. Switch: `temporal_smoothing` (default on).
* **Visible box.** The patch was often a little lighter or darker than its surroundings, so its outline showed
  (`match_colour` only corrected constant offsets ≥ 6 levels). `match_seam` now solves a Poisson (gradient-domain) blend: the
  model's texture inside, exactly the real pixels at the edge. Where the patch touches the border of its crop it falls back to
  `match_colour`.

Measured with the production engine and the real LaMa model (`flick.py`-style harness: mean absolute change per frame inside
the mask, grey levels; "ratio" is the QC flicker ratio against the ring around the mask):

| Clip, frames | Original footage | Before | After |
|---|---|---|---|
| Coca-Cola clip, mark over a moving dog (50 frames) | 1.77 | 3.18, ratio 0.93 | **1.80, ratio 0.53** |
| GTA clip, face with watermark (60 frames) | 6.31 (includes the watermark) | 5.56, ratio 0.95 | **2.01, ratio 0.35** |

Unit tests: `tests/test_fill_smoothing.py` (patch change drops below 0.6× with smoothing; nothing outside the patch changes; a
cut isn't smoothed across; a changed surrounding stops the blend; the seam is closed). What this does **not** do: it
cannot make the patch show what really was behind the mark, and a patch that is steadier than the real footage can look
slightly static where something passes behind it.

## 4. Quality check after the render

`app/services/quality.py` re-surveys the original and the result and reports, per removed region: how much of the
overlay's steady structure is still there (`residual`, warn above 0.5) and whether the filled area flickers
(`flicker`, warn above 2.5). Warnings are added to the job and shown on the result page. It can say "part of the
overlay may still be visible"; it cannot say "perfect".

## 5. Measured numbers

All wall-clock, from `scripts/e2e_auto.py` driving the real HTTP flow (upload only), engine chain with real LaMa.

| Clip (what it is) | Frames | Analysis + detection | Render | Total after upload | Notes |
|---|---|---|---|---|---|
| static logo, 960×454, 5 s | 150 | 3.6 s | 55.8 s | 60.7 s | QC clean |
| transparent logo | 150 | 3.1 s | ~62 s | 67.2 s | |
| text overlay | 150 | 3.6 s | ~82 s | 86.0 s | |
| two marks | 150 | 3.1 s | ~140 s | 143.6 s | |
| mark that jumps between places | 150 | 4.1 s | ~44 s | 48.4 s | |
| no overlay | 150 | 3.6 s | none | 3.6 s | "nothing found", nothing rendered |
| real Shorts clip with moving "Vazh", 720×1280 60 fps, with audio | 310 | 12.8 s | 157.0 s | 174.3 s | QC clean; infographic "Vazh" not found (see limits) |

Upload to the local server took 0.03–0.13 s and download 0.01–0.13 s; over a real network those depend on the
connection and are reported separately by the script.

Detection alone (`scripts/bench_detect.py`, synthetic clips with ground truth, `AUTO_APPEARANCE` default):

| Clip | detect s | recall | precision | frames hit (≥70 %) |
|---|---|---|---|---|
| static logo | 1.5 | 0.73 | 0.96 | 150/150 |
| static logo 2288×1080 | 2.1 | 0.75 | 0.96 | 150/150 |
| transparent logo | 1.5 | 1.00 | 0.63 | 150/150 |
| text | 1.5 | 0.97 | 0.94 | 150/150 |
| two marks | 1.6 | 0.80 | 0.73 | 150/150 |
| mark that jumps | 2.2 | 1.00 | 0.63 | 150/150 |
| appears and disappears | 1.8 | 1.00 | 0.62 | 90/90 |
| size change | 1.5 | 0.70 | 0.87 | 85/150 |
| scene changes | 5.8 | 0.43 | 0.94 | 0/300 |
| smoothly drifting mark | 1.3 | 0.00 | – | 0/150 |
| no overlay | 1.5 | – | – | no false removal |

Recall is the share of the true logo area covered; precision the share of the mask that is logo. Round logos fill
only ~78 % of their box, so recall near 0.75 on those is the mask hugging the visible shape.

## 6. Limits (not hidden)

* **Drifting marks over live footage are not detected.** A mark sliding over a still picture is (GTA clip: 0.895 recall, 143 of
  153 frames); a faint mark sliding over moving, high-contrast footage is not (same clip, region B1: not found; synthetic
  `moving_smooth`: not found). Template matching with a still-learned template does no better than chance there. Hand-drawn
  marks still work for it.
* **A mark on a still picture inside the video** (the small fox badge on the PS4 card) can't be told from the picture.
* **Tiny or faint marks** (the small "Vazh" inside the infographic of the test clip) are below the survey resolution.
* **Still pictures** can't prove an overlay. Only a small isolated mark in a corner is removed on look alone, and it is flagged.
* **Scene changes**: recall 0.43 and the check frames sampled can miss; the report says what was found.
* **Speed**: the render, not detection, dominates. On this CPU it is ~0.4 s per 960×454 frame and ~0.5 s per 720×1280
  frame of LaMa work. A 15–20 s total for a 5–6 s video was **not achieved here** and no GPU was available to
  measure; GPU, FP16, NVDEC/NVENC remain unverified (see PERFORMANCE.md §7 to measure them on a GPU machine).
* Judgement of what is a "watermark" versus part of the video can be wrong either way; the editor is the correction.

## 7. Reproduce

```bash
python scripts/make_overlay_clips.py CLIPS_DIR          # synthetic clips + ground truth
python scripts/bench_detect.py CLIPS_DIR --json out.json
python scripts/bench_real.py CLIP.mp4 hand_marks.json   # detector vs a hand-made mask, real clip
AI_MODEL_PATH=big-lama.pt python scripts/bench_fill.py CLIP.mp4 marks.json 100 50 out.npz   # shimmer in generated fill
python run_local.py                                      # then, in another terminal:
python scripts/e2e_auto.py --video clip.mp4 --out /tmp/e2e [--clean twin.mp4]
```
