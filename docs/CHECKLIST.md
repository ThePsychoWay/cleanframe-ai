# CleanFrame AI — short checklist

Only remove overlays from content you own or have permission to edit.

## Run it on your computer (no API, no accounts, no Docker)

- [ ] Install **Python 3.11+**, **FFmpeg** (`ffmpeg -version` and `ffprobe -version` work) and **Node.js 20+**.
- [ ] Windows: double-click **Start CleanFrame.bat**. macOS: double-click **Start CleanFrame.command**. Linux: `./start.sh`.
      (Or in a terminal: `python run_local.py`.)
- [ ] First run only: wait while it installs, builds and downloads the fill model (about 400 MB; several minutes, needs
      internet once). The window must say **"Generative fill is on"**. Later runs start in seconds, offline.
- [ ] Open the address it prints (http://localhost:3000), create an account (stored only in `cleanframe-data/`), upload a video.
- [ ] Upload the video and wait: it finds the overlays itself, removes them, checks the result and shows **Your video is ready.** Then **Download video**. (Nothing found? It says so and opens the editor: mark the logo with Rectangle, Brush or Eraser and press **Remove Watermark**. **Edit selection** on the result page adjusts what was removed. Details: docs/AUTOMATION.md.)
- [ ] Want to use it from your phone? Double-click **Start CleanFrame for Phone** instead, then open the address it prints on the phone (same Wi-Fi, home network only; see docs/LOCAL.md).
- [ ] Stop with **Ctrl+C**. Your videos stay in `cleanframe-data/` and are deleted automatically after the retention time.

## If your computer is short on memory

- [ ] `python run_local.py --memory-mb 512` (fewer frames studied; minimum 256). A very large mark still needs about 2.3 GB (measured); smaller marks need less.
- [ ] Keep `--workers 1` (the default): one video at a time.
- [ ] Close other heavy programs while a large video renders. A big marked area needs more memory than a small one.
- [ ] Mark only the logo itself, not a big box around it.

## What to expect

- [ ] A **semi-transparent logo that stays put** is taken off exactly (the pixels underneath are restored, not guessed).
- [ ] An **opaque logo** is rebuilt from other frames, but only if the camera moves enough to show what is behind it.
      If neither works, the app **stops with the reason** instead of blurring (the offline fill model, on by default, generates it).
- [ ] Nothing is sent anywhere: no API key, no cloud, no upload to a third party.

## Things only you can check (not proven in the build environment)

- [ ] Your own footage: try one real video with a short clip first.
- [ ] Windows/macOS: the launcher was run on Linux only.
- [ ] Logos that never move over a still camera (white text, solid logos) are filled by the offline model, which is on
      by default. The area is generated, not recovered. Tested on one real clip on CPU only; GPU and Windows untested.
- [ ] Docker images (`docker/`): prepared but never built, because the build environment could not pull images.
- [ ] Real ClamAV scanning, real S3/MinIO and PostgreSQL deployment: tested against stand-ins only.

More detail: [LOCAL.md](LOCAL.md).
