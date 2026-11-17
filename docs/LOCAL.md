# Running CleanFrame AI on your computer

No API keys, no accounts with anyone, no Docker, PostgreSQL, Redis or cloud storage. Double-click
**`Start CleanFrame.bat`** (Windows) or **`Start CleanFrame.command`** (macOS), or run `python run_local.py`
(Linux: `./start.sh`). Your browser opens the site at http://localhost:3000.

> Only remove overlays from content you own or have permission to edit.

Needs **Python 3.11+**, **FFmpeg** (`ffmpeg` and `ffprobe` on your PATH) and **Node.js 20+** (not needed with `--no-web`).
The first run creates `.venv`, installs the Python packages, installs and builds the web app (a few minutes, internet
needed once). After that it starts in seconds and works offline. Stop it with Ctrl+C.

| Option | Default | Meaning |
|---|---|---|
| `--port` / `--web-port` | 8000 / 3000 | Where the API / the web app listen (localhost only). |
| `--data-dir` | `./cleanframe-data` | Database, videos and the generated secret. Delete the folder to start fresh. |
| `--memory-mb` | 1024 | Memory the overlay measurement may use (minimum 256). |
| `--workers` | 1 | Videos processed at the same time. 1 uses the least memory. |
| `--no-web` | off | API only. |
| `--rebuild` | off | Rebuild the web app (done automatically when a port changes). |
| `--no-generative-fill` | off | Skip the offline fill model (about 400 MB). Logos nothing else can remove are then refused. |

## What runs where

| Normally | Local mode |
|---|---|
| PostgreSQL | a SQLite file (`cleanframe-data/cleanframe.db`, WAL mode) |
| S3 / MinIO | the disk (`cleanframe-data/storage/`); the browser uploads and downloads through the API with signed, expiring links |
| Redis + Celery worker + beat | threads inside the API process; the retention sweep runs on a timer thread |
| `JWT_SECRET` you generate | generated once into `cleanframe-data/secret.key` (owner-readable only) |
| `AI_PROVIDER=dev` blur engine | `AI_PROVIDER=auto`: real removal (exact overlay inversion, otherwise rebuild from other frames) |

Everything is processed on this computer. Nothing is sent to any server, and no key is needed because no outside AI
service is involved. `AI_PROVIDER=api` (a third-party service) is the only mode that would send frames elsewhere, and
local mode never selects it.

## Memory

At rest the API plus the web app use roughly 400 MB (measured: ~310 MB API, ~100 MB web). A render needs more, and the
amount depends on **the size of the marked area**, not the video's length: the overlay is measured from a sample of frames,
and each frame costs memory in proportion to the marked area. `--memory-mb` caps how many frames are studied (never fewer
than 16, which is the minimum for a trustworthy measurement), so a huge marked area can still need more than the number
you give.

**Video size matters too.** Measured with the whole app running locally on 2 CPUs and 2-second clips: a render peaked at
1.7 GB for 720p, 2.4 GB for 1080p, 3.3 GB for 1440p and 5.5 GB for 4K (`docs/RESOLUTION.md`). Plan on 8 GB of RAM for 4K;
`TEMPORAL_MEMORY_MB` limits the part that is under the app's control. 4K is also slow without a GPU (about 12 s per frame).

Measured on this build (2 CPUs, 1920×960 video, a 1550×530 px mark, 60 frames sampled before the change):

| Setting | Peak memory of one render |
|---|---|
| before Phase 25 (60 frames studied) | about 3.0 GB |
| `--memory-mb 1024` (default, 19 frames) | about 2.3 GB |
| `--memory-mb 512` or `256` (16 frames, the floor) | about 2.3 GB (no further gain) |

So for a very large marked area the floor of 16 frames is reached early and about 2.3 GB remains; lowering the setting
cannot go below it. Small and medium marks need far less (memory grows with the marked area), but only the large case
was measured here; the test clips in the build are tiny (a few hundred MB at most, not separately measured).

Tips: mark only the logo, not a large box around it; keep `--workers 1`; close other heavy programs for long renders.

## Videos where the logo and the camera never move

If the logo is solid, or the picture behind it never changes (a still camera on a still scene), nothing in the video shows
what is under the logo, so exact removal and rebuilding from other frames both have to stop. The app then says "100% of the
marked area ... is hidden in every nearby frame". The only way to fill such an area is a generative model that **invents** a
plausible patch. It is **on by default**, so there is nothing to type:

The first start installs PyTorch (CPU, about 200 MB) into `.venv` and downloads the public LaMa model (about 200 MB,
Apache-2.0, checked against its published SHA-256) into `cleanframe-data/models/`. After that every start uses it
automatically, offline. (`--no-generative-fill` skips all of this.) It is used only for areas no other frame can supply, and the finished
video's notes say how much was generated rather than recovered.

Measured on the one real clip I was given (5.2 s, 720x1280, two "Vazh" text marks): both marks removed with no visible
trace on the frames I checked. The small mark in the card took about 3 minutes on 2 CPU cores for 310 frames; a bigger area
takes longer. Generated fill can shimmer on fast-moving footage; judge your own clips.

You can also supply an ONNX LaMa export instead: save it as `cleanframe-data/models/lama.onnx` (no PyTorch needed).

## Using it from a phone or tablet

Double-click **Start CleanFrame for Phone** (or run `python run_local.py --lan`). It prints an address such as
`http://192.168.1.20:3000`; open that in the phone's browser while the phone is on the **same Wi-Fi** as the computer.
Everything still runs on the computer; the phone is only the screen, so uploads travel over your Wi-Fi and the computer does the work.

How it is kept simple and closed: only the web app's port is opened to the network. It forwards `/api/*` to the processing server,
which keeps listening on this computer only, and file links are relative, so they work from whichever address the page was opened at.

Things to know:
* **Trusted networks only.** Anyone on the same Wi-Fi can open that address and make an account. Use it at home, not on public Wi-Fi.
* The first `--lan` start rebuilds the web app (a few minutes), and so does switching back to the normal start. Changing the
  computer's network address (a new Wi-Fi, a restart of the router) does not need a rebuild: just use the address it prints.
* If the phone can't connect: allow Python/Node through the computer's firewall when it asks; make sure the Wi-Fi isn't a
  "guest" network that stops devices seeing each other.
* The phone only needs a browser. Marking an area works with a finger (drag a rectangle; the corner handles are bigger on touch).
* Checked here with a phone-sized, touch-enabled browser against the real app on the network address: sign-up, a 76 MB 4K upload in
  10 parts, analysis, drawing the selection with finger touches, removal, the result page with both videos playing, and the
  download. **Not checked on a real phone** (this machine has none), so the first real-phone run is the real test.

## Limits to know

- **Single user, single computer.** Local mode serves files over plain http on localhost and is refused if `API_PUBLIC_URL`
  is not localhost. For a server shared by several people use the Docker/PostgreSQL setup in the README.
- **No hard time limit on a job** (the Celery setup has one). Cancelling is checked between frames, so a job stuck inside a single step is ended by restarting the app (it is then marked failed).
- **Closing the app stops running work.** Work that was queued or running when the app closed is marked failed the next
  time it starts, with a message to try again; nothing is resumed.
- **SQLite** handles one writer at a time; with `--workers 1` and one person that is not noticeable.
- **Opaque logos on a still camera** cannot be restored from the video alone (nothing shows what is behind them). With
  `--no-generative-fill` the app stops with that reason; by default the area is generated, not recovered.
- Windows and macOS: the launcher is written to be portable but was only run on Linux.

## Where things are

`cleanframe-data/` holds everything. Deleting a project in the app deletes its files; finished and abandoned projects are
removed after the retention time (`VIDEO_RETENTION_HOURS`), as the privacy notice in the app says.
