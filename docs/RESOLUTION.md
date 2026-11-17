# Resolution: 720p, 1080p, 1440p and 4K

CleanFrame AI accepts, processes and delivers 720p, 1080p, 1440p and 4K videos. The finished video keeps the source's
size, frame rate, orientation, pixel format and audio unless the person asks for a smaller size. Nothing here is mocked:
every number below was measured on this machine with the real server, the real LaMa model and FFprobe on the files.

**Test machine:** 2 CPU cores, 8 GB RAM, no GPU, Python 3.13, PyTorch CPU. Anything that needs a GPU is marked
*not verified*.

## 1. What limited the app to 720p (audit)

| Area | What it was | What it is now |
|---|---|---|
| Frontend upload check | `.mp4 .mov .webm` only; 500 MB default limit | adds `.avi`; 2 GiB default (`NEXT_PUBLIC_MAX_UPLOAD_SIZE` follows the server's `MAX_UPLOAD_SIZE`) |
| Frontend output choice | `1080 \| 720`, compared with the video's *height* | 720 / 1080 / 1440 / 4K by the **shorter side** (a 1080×1920 portrait clip is 1080p), only sizes smaller than the original |
| API: render request | `output_height` accepted only 720 or 1080 | 720, 1080, 1440, 2160, or none (= original); `codec` h264 / hevc / vp9 checked against the container |
| API: upload | `MAX_UPLOAD_SIZE` 500 MB; `MAX_VIDEO_PIXELS` 8,294,400 (rejects 4096×2160); AVI refused | 2 GiB; 9,437,184 px (4096×2304); AVI accepted when FFmpeg can read it. All three are settings, not constants |
| Metadata | width, height, fps, duration, codec, bitrate, size already read by ffprobe | unchanged; container and audio codec are now shown too |
| AI inpainting | large fill areas were **shrunk** to fit the model, then enlarged | filled **at the video's own resolution** in overlapping pieces (section 3); a model that cannot take larger input is the only case that shrinks, and that is counted and reported |
| Temporal engine | kept the same number of neighbouring frames whatever their size (a 4K window would cost GBs) | the window shrinks to a memory budget (`TEMPORAL_MEMORY_MB`) and the job's notes say so |
| FFmpeg | H.264 only; output cap compared with the long side | H.264, H.265/HEVC (tagged `hvc1`), VP9; NVENC when a real test encode works; cap by shorter side; never enlarges |
| Storage / disk | no check; a full disk failed half-way | free-space check at upload start (HTTP 507) and before analysis and render (needs about 1.5× / 3.5× the upload) |
| Download | one file | the original-size file by default, plus extra exports at smaller sizes / other formats |
| **Kept on purpose** | the editor's preview copy is at most 720 px tall | still 720p: it exists so a 4K video never has to be decoded by the browser. Marks are saved as fractions of the frame, so the render at full size is unaffected. The result page says the "Original" side is this lower-resolution copy. |

There was no `scale=1280:720`-style filter in the render path; the only scaling is the explicit, never-enlarging
`output_height` option and the new exports.

## 2. Upload and storage

* Uploads are S3-style multipart: 8 MiB parts, each PUT directly to storage (local mode: to the API's signed endpoint),
  so the browser holds one part at a time, never the file. A dropped connection resumes: asking to upload the same
  file again returns the same project and the parts already stored (`GET /api/projects/{id}/upload`).
  Checked on real 4K files: 10 parts, 5 sent, "connection lost", the other 5 sent, upload completed, analysis correct.
* Limits are settings: `MAX_UPLOAD_SIZE`, `MAX_VIDEO_PIXELS`, `MAX_VIDEO_DURATION_SECONDS`, `MIN_FREE_DISK_MB`.
* Detected on upload (ffprobe, never the browser): width, height, fps, duration, codec, bitrate, size, aspect ratio,
  container, audio codec. The upload screen shows `Resolution 3840 × 2160`, `Quality 4K Ultra HD`, `Frame rate 30 FPS`, `Format MP4`.
* Downloads are signed links straight to storage (local mode: a range-capable file response), so nothing is read into
  the API's memory and the browser streams the video.
* Temporary files live in the worker's temp directory and are removed when the job ends; expired projects (and their
  exports) are swept by the retention job.

## 3. How the AI handles big frames

The model is LaMa (TorchScript, accepts any size that is a multiple of 8).

1. The fill is **localized**: only a padded region around the mark is given to the model, not the whole frame.
2. If that region is no larger than 1024 px on its longer side it goes through in one piece.
3. If it is larger (a wide caption on a 4K frame), it is cut into pieces of up to 1024 px that **overlap by 64 px**.
   Each piece keeps only what it generated in its own part of the overlap's middle, so every handover happens
   where both pieces see the area well inside their view. Pieces are filled one after another and later pieces see the
   earlier fill as context, which is what keeps texture and colour continuous across the join. Mask alignment is exact:
   the pieces tile the hole with no gap and no double coverage (tested).
4. Nothing is shrunk. The note on the job says "filled in N overlapping pieces of up to X px at the video's full
   resolution (nothing was shrunk)".
5. Temporal consistency is unchanged: the temporal engine reuses footage from neighbouring frames where the mark
   moves, reuses a fill while its surroundings are unchanged, and the stabiliser blends frame to frame.

This is **not** a claim of "native 4K reconstruction": LaMa invents plausible detail from the surroundings. It does
not recover what was behind the mark, at any resolution.

**GPU memory (code written and unit-tested with a stand-in runtime; not verified on a real GPU).** On CUDA the free VRAM
decides the first piece size (≥ 8 GB free: 1024, ≥ 4 GB: 768, ≥ 2 GB: 512, else 384). If the model still runs out of
memory the piece size is reduced and that fill restarted; only if the smallest piece fails does the job stop, with
*"Your video is 3840 × 2160, but the processing worker doesn't have enough GPU memory to rebuild the marked area, even
in small pieces. Mark a smaller area, or retry on a processing worker with more GPU memory (VRAM). Nothing was lowered
in quality to get around this."* Cached GPU memory is released after an out-of-memory error. FP16 / bf16 is **not
enabled**: it fails inside this TorchScript model (see `docs/PERFORMANCE.md` §6) and could not be tested without a GPU.
LaMa runs one piece at a time, so "batch size" here means piece size; frames are not batched.

## 4. Encoding

| | |
|---|---|
| Containers / codecs | MP4: H.264 (default), H.265/HEVC; WebM: VP9. A codec the server's FFmpeg lacks is not offered |
| Quality | constant-quality (CRF / CQ), never a fixed low bitrate. FAST / BALANCED / HIGH map to preset and CRF in `app/video/encoders.py` |
| NVENC | used only if a real one-frame test encode succeeds on this machine (cached); otherwise the CPU encoder. `VIDEO_ENCODER=cpu` forbids it. **Not verified: no NVIDIA GPU here** |
| Hardware decoding | **not implemented** (it cannot be verified without a GPU, and decoding is about 1 % of the time on this machine) |
| Audio | stream-copied when the container can hold the codec (verified: same codec, rate, channels, duration, bitrate, start offset); otherwise converted once and the result says so |
| Colour | matrix, primaries, transfer and range are read from the source and written to the output; pixel format is kept (verified `yuv420p` = `yuv420p`) |
| One encode | the render encodes once. A requested smaller size is done in that same encode; an extra export re-encodes the *finished file once* (no AI re-run), so a 4K→1080p export is one generation more than the 4K file |

## 5. Extra exports

`GET /api/projects/{id}/exports/options` lists only what is valid: sizes strictly smaller than the finished video
(by shorter side) and codecs this server can encode. `POST …/exports` refuses an enlargement (422) and an export that
equals the finished file (409); asking twice for the same thing returns the same export. A new render drops the old
exports. The result page shows `Original resolution`, `Output resolution`, `Quality` and the "More sizes and formats"
panel, which is hidden when there is nothing to choose (a 720p result in its only format). A 720p source never
offers 1080p or 4K.

## 6. Measured results

Test clips: real footage (cropped to 16:9, 2 s, 60 frames, 30 fps, AAC stereo), upscaled with Lanczos to each size with
temporal grain added, plus an opaque 24 % × 13 % logo plate drawn at a fixed place. A clean twin is the ground truth.
**The footage is upscaled, so it has no real fine detail beyond 720p; these runs test the pipeline's handling of size,
memory, audio and metadata, not 4K sensor detail.** The grain makes the sources very high-bitrate (31–315 Mb/s), so
the outputs (10–88 Mb/s) are smaller than the inputs and the "outside the logo" PSNR is ~34 dB: that is the balanced preset
discarding synthetic grain, not damage.

Driven through the real HTTP API (register → interrupted-and-resumed upload → analysis → mask → render → download →
extra exports) by `scripts/e2e_resolution.py`; every output was checked with FFprobe, not with the app's labels.

| Clip | Source | Output (ffprobe) | Frames | Audio | Render | fps | Peak server RAM | Mean CPU | PSNR in logo / outside | Checks |
|---|---|---|---|---|---|---|---|---|---|---|
| 720p | 1280×720 h264 31 Mb/s | 1280×720 h264 10 Mb/s | 60/60 | aac copied | 84 s | 0.71 | 1.7 GB | 179 % | 13.33 / 31.64 dB | all passed |
| 1080p | 1920×1080 h264 75 Mb/s | 1920×1080 h264 23 Mb/s | 60/60 | aac copied | 143 s | 0.42 | 2.4 GB | 177 % | 14.18 / 33.83 dB | all passed |
| 1440p | 2560×1440 h264 135 Mb/s | 2560×1440 h264 39 Mb/s | 60/60 | aac copied | 231 s | 0.26 | 3.3 GB | 177 % | 14.2 / 33.85 dB | all passed |
| 2160p | 3840×2160 h264 315 Mb/s | 3840×2160 h264 88 Mb/s | 60/60 | aac copied | 738 s | 0.08 | 5.5 GB | 183 % | 14.02 / 33.82 dB | all passed |

| Export from | To | Size (ffprobe) | Frames | Time | PSNR in logo / outside vs scaled truth |
|---|---|---|---|---|---|
| 1080p | 720p | 1280×720 | 60 | 4 s | 14.38 / 35.65 dB |
| 1440p | 1080p | 1920×1080 | 60 | 8 s | 14.38 / 35.5 dB |
| 1440p | 720p | 1280×720 | 60 | 4 s | 14.43 / 36.65 dB |
| 2160p | 1440p | 2560×1440 | 60 | 13 s | 14.16 / 35.92 dB |
| 2160p | 1080p | 1920×1080 | 60 | 9 s | 14.2 / 36.89 dB |
| 2160p | 720p | 1280×720 | 60 | 5 s | 14.24 / 36.94 dB |

Every row also passed: detected size / fps / frame count equal ffprobe's; output size, fps, frame count and pixel format
equal the source's; audio codec, sample rate, channels, duration and bitrate equal the source's (stream copy) and the
audio-video start offset is unchanged (sync); an enlargement request is refused; no size at or above the source is offered.

**Fill quality** (PSNR inside the logo, against the clean twin) is 13.3 dB at 720p and 14.0–14.2 dB at 1080p to 4K:
the fill does not get worse with resolution, but it is a low number at every size because an opaque plate covers a
structured subject (sugar cubes) and LaMa can only make up plausible detail. The filled area is visible as a slightly
different tone. This is the existing model's limit, not something the upgrade changed; it is shown here so the numbers
are not read as "recovered".

**Time does not scale like frames.** On 2 CPU cores: 0.71 fps at 720p, 0.42 at 1080p, 0.26 at 1440p, 0.08 at 4K
(8.8× the time of 720p for 9× the pixels). At this clip's rate a 30 s 4K clip would take about 3 hours on this CPU. The GPU numbers are *not measured*.

**Memory.** Peak resident memory of the server and its FFmpeg children: 1.7 / 2.4 / 3.3 / 5.5 GB at 720 / 1080 / 1440 / 4K.
4K needs a machine with at least 8 GB.

**Encoders at 4K** (24 real 3840×2160 frames, `FrameWriter`, balanced): H.264 → 3840×2160 `avc1`, PSNR 39.1 dB against the
source; HEVC → `hvc1`, 38.2 dB; VP9 in WebM → 35.9 dB; all 24 frames, 30 fps, `yuv420p`. Capped to 1080 (H.264) and 1440 (HEVC)
the sizes were 1920×1080 and 2560×1440.

## 7. Errors people see

Unsupported codec ("This video uses the … codec, which can't be read…"), over the size limit (names the limit), more
pixels than allowed (names the limit), not enough disk (HTTP 507, says what was needed), the model out of memory (above),
a corrupt file, an FFmpeg failure ("The video couldn't be encoded."), a failed upload (resumable). No path lowers a
video to 720p to get around an error.

## 8. Not done / not verified

* GPU, VRAM sizing on a real device, NVENC, FP16: **not verified** (no GPU).
* Hardware decoding: not implemented.
* 4K through the browser editor: not tested in a browser here; the editor uses the 720 px preview by design.
* Render-time output smaller than the source (`output_height`) is covered by unit tests of the size rule and the
  pipeline, and the extra-export path was run end to end at every size; a full 4K render with `output_height=1080` was not run.
* AVI depends on the server's FFmpeg being able to demux and decode the file; the container check and the message are
  tested, a real AVI of every codec is not.
* Automatic FAST/BALANCED/HIGH selection by resolution is not implemented; the person chooses.

## 9. Reproduce

```bash
scripts/make_resolution_clips.sh source.mp4 /tmp/res 2          # clean_* and marked_* at 720p, 1080p, 1440p, 4K
# start the server (LOCAL_MODE=true), then per clip:
python scripts/e2e_resolution.py --marked /tmp/res/marked_2160p.mp4 --clean /tmp/res/clean_2160p.mp4 \
    --label 2160p --out /tmp/res/out --export 1080 --server-pid <uvicorn pid>
python scripts/summarise_resolution.py /tmp/res/out               # the tables above
```

The raw numbers behind section 6 (one JSON per clip) and two screenshots of the real result page are in
`docs/resolution-results/`; `scripts/codec_check_4k.py` is the 4K encoder check.
