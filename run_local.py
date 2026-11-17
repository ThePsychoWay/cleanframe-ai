#!/usr/bin/env python3
"""Run CleanFrame AI on this computer with one command:  python run_local.py

No account, API key, Docker, PostgreSQL, Redis or cloud storage is needed. Everything is kept in ./cleanframe-data
(a SQLite file plus the videos), background jobs run inside the API process, and your videos never leave this computer.

First run: creates a virtual environment, installs the Python packages, installs and builds the web app (a few
minutes, needs internet once for the packages). Later runs start in seconds and work offline.

The first run also downloads the offline fill model (about 400 MB) used for logos that sit still over a still picture
(see docs/LOCAL.md); add --no-generative-fill to skip it.

Needs: Python 3.11+, FFmpeg (ffmpeg and ffprobe on PATH) and Node.js 20+. Stop with Ctrl+C.
Only remove overlays from content you own or have permission to edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND, FRONTEND = ROOT / "backend", ROOT / "frontend"
VENV = ROOT / ".venv"
MIN_PYTHON, MIN_NODE = (3, 11), 20
# The public big-lama model (Apache-2.0, TorchScript), published on GitHub by the IOPaint project. Checked after download.
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"


def fail(message: str) -> None:
    print(f"\n  ✗ {message}\n", file=sys.stderr)
    raise SystemExit(1)


def say(message: str) -> None:
    print(f"  {message}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str | Path], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    result = subprocess.run([str(c) for c in command], cwd=cwd, env=env, check=False)  # noqa: S603
    if result.returncode:
        fail(f"This step failed: {' '.join(str(c) for c in command[:4])} (exit code {result.returncode}). The output above says why.")


def check_tools(*, web: bool) -> None:
    if sys.version_info < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed (this is {sys.version.split()[0]}).")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            fail(f"{tool} was not found. Install FFmpeg (https://ffmpeg.org/download.html) and make sure `{tool}` works in a terminal.")
    if web:
        if not shutil.which("node") or not shutil.which("npm"):
            fail("Node.js was not found. Install Node.js 20 or newer (https://nodejs.org), or run with --no-web to use the API only.")
        out = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False).stdout.strip().lstrip("v")  # noqa: S607
        try:
            if int(out.split(".")[0]) < MIN_NODE:
                fail(f"Node.js {MIN_NODE} or newer is needed (this is {out}).")
        except ValueError:
            fail(f"Could not read the Node.js version ({out!r}).")


def ensure_python_packages() -> Path:
    """Creates ./.venv and installs backend/requirements.txt into it once (and again when that file changes)."""
    python = venv_python()
    stamp = VENV / ".requirements-installed"
    wanted = (BACKEND / "requirements.txt").read_text(encoding="utf-8")
    if python.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == wanted:
        return python
    if not python.exists():
        say("Creating a Python environment (.venv) ...")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    say("Installing Python packages (first run only, needs internet) ...")
    run([python, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", "-r", BACKEND / "requirements.txt"])
    stamp.write_text(wanted, encoding="utf-8")
    return python


def heal_blocked_extensions(python: Path) -> None:
    """Some Windows setups (Smart App Control / Application Control policies) refuse to load the compiled speed-ups that
    ship inside SQLAlchemy ("DLL load failed ... blocked this file"). SQLAlchemy 2.1 contains the same code as plain
    Python, so deleting only those compiled files makes it fall back to that, with no change in behaviour (a little
    slower database access, which doesn't matter here). Other blocked packages can't be worked around; they are named."""

    def attempt(module: str) -> str | None:
        result = subprocess.run([str(python), "-c", f"import {module}"], capture_output=True, text=True, check=False)  # noqa: S603
        return None if result.returncode == 0 else result.stderr.strip()

    error = attempt("sqlalchemy")
    if error and ("blocked" in error.lower() or "dll load failed" in error.lower()):
        located = subprocess.run(  # noqa: S603
            [str(python), "-c", "import importlib.util as u, os; print(os.path.dirname(u.find_spec('sqlalchemy').origin))"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        removed = 0
        for compiled in Path(located).rglob("*.pyd") if located else []:
            try:
                compiled.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            say(f"Windows blocked SQLAlchemy's compiled speed-ups; removed {removed} file(s) so it runs as plain Python.")
        error = attempt("sqlalchemy")
    for module in ("sqlalchemy", "pydantic_core", "numpy", "cv2"):
        error = error if module == "sqlalchemy" else attempt(module)
        if error:
            fail(
                f"Python could not load '{module}':\n{error.splitlines()[-1] if error else ''}\n"
                "If this says an Application Control / Smart App Control policy blocked a file, Windows is refusing "
                "unsigned compiled packages on this computer. Turn Smart App Control off (Windows Security > App & browser "
                "control > Smart App Control) or run CleanFrame on a computer where it is off."
            )


def ensure_generative_fill(python: Path, models: Path) -> Path:
    """Installs PyTorch (CPU) into .venv and downloads the LaMa model once. Both stay on this computer afterwards."""
    probe = subprocess.run([str(python), "-c", "import torch"], capture_output=True, check=False)  # noqa: S603
    if probe.returncode:
        say("Installing PyTorch for the generative fill (first time only, about 200 MB) ...")
        install = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--quiet", "torch"]
        # The CPU-only build is much smaller; if that download address can't be reached, the normal one works too.
        attempts = [install] if sys.platform == "darwin" else [[*install, "--index-url", "https://download.pytorch.org/whl/cpu"], install]
        if not any(subprocess.run(command, check=False).returncode == 0 for command in attempts):  # noqa: S603
            fail("PyTorch couldn't be installed (see above). Check your internet connection and run this again, or use --no-generative-fill.")
    target = models / "big-lama.pt"
    if target.is_file() and sha256_of(target) == LAMA_SHA256:
        return target
    models.mkdir(parents=True, exist_ok=True)
    say("Downloading the LaMa model (first time only, about 200 MB) ...")
    partial = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(LAMA_URL, timeout=60) as response, partial.open("wb") as out:  # noqa: S310
            total, done, shown = int(response.headers.get("Content-Length") or 0), 0, -1
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total and done * 10 // total != shown:
                    shown = done * 10 // total
                    say(f"  {shown * 10}%")
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        fail(f"The LaMa model couldn't be downloaded ({error}). Check your internet connection and run the same command again.")
    if sha256_of(partial) != LAMA_SHA256:
        partial.unlink(missing_ok=True)
        fail("The downloaded model doesn't match its published checksum, so it was thrown away. Run the same command again.")
    partial.replace(target)
    return target


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_web_build(plan: Plan) -> None:
    """Installs and builds the web app once. The API address is built into it, so a port change rebuilds it."""
    stamp = FRONTEND / ".next" / "local-build.json"
    wanted = {"api": plan.build_api, "lock": (FRONTEND / "package-lock.json").stat().st_mtime_ns if (FRONTEND / "package-lock.json").exists() else 0}
    try:
        if json.loads(stamp.read_text(encoding="utf-8")) == wanted:
            return
    except (OSError, ValueError):
        pass
    npm = shutil.which("npm") or "npm"
    if not (FRONTEND / "node_modules").is_dir():
        say("Installing the web app's packages (first run only, needs internet) ...")
        run([npm, "ci" if (FRONTEND / "package-lock.json").exists() else "install", "--no-audit", "--no-fund"], cwd=FRONTEND)
    say("Building the web app (first run only, a few minutes) ...")
    env = {**os.environ, **plan.build_env, "NEXT_TELEMETRY_DISABLED": "1", "NODE_ENV": "production"}
    run([npm, "run", "build"], cwd=FRONTEND, env=env)
    stamp.write_text(json.dumps(wanted), encoding="utf-8")


def lan_address() -> str | None:
    """This computer's address on the home network (what a phone types), or None when it isn't on one.

    Connecting a UDP socket sends nothing; it only makes the system pick the network card it would use to reach the internet."""
    for target in ("10.255.255.255", "192.168.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect((target, 9))
                address = probe.getsockname()[0]
        except OSError:
            continue
        if address and not address.startswith(("127.", "0.")):
            return address
    return None


@dataclass(frozen=True)
class Plan:
    """Where everything listens and what address each part is told to use. Pure data, so it can be tested."""

    api_origin: str  # how this script reaches the API (health check)
    web_origin: str  # how this script reaches the web app, and what the browser on this computer opens
    web_host: str | None  # the address the web app listens on (None = the web server's own default)
    build_api: str  # the API address built into the web app
    build_env: dict[str, str]  # extra environment for building the web app
    api_env: dict[str, str]  # extra environment for the API
    phone_url: str | None


def make_plan(api_port: int, web_port: int, lan_ip: str | None, lan: bool, hostname: str = "") -> Plan:
    """Normal: API on :api_port, web app on :web_port, both for this computer only.

    --lan: the web app listens on the network and is the ONLY thing that does; it forwards /api/* to the API, which stays
    on 127.0.0.1. The phone therefore needs one address, nothing else is exposed, and every link the API hands out is
    relative so it works from whichever address the page was opened at."""
    if not lan:
        api, web = f"http://localhost:{api_port}", f"http://localhost:{web_port}"
        return Plan(api, web, None, api, {"NEXT_PUBLIC_API_URL": api, "NEXT_PUBLIC_STORAGE_ORIGIN": api}, {"API_PUBLIC_URL": api, "CORS_ORIGINS": web}, None)
    if not lan_ip:
        raise ValueError("no network address")
    origins = [f"http://{lan_ip}:{web_port}", f"http://localhost:{web_port}", f"http://127.0.0.1:{web_port}"]
    if hostname:
        origins.append(f"http://{hostname}:{web_port}")
    return Plan(
        api_origin=f"http://127.0.0.1:{api_port}",
        web_origin=f"http://localhost:{web_port}",
        web_host="0.0.0.0",  # noqa: S104 - the point of --lan
        build_api=f"same-origin->127.0.0.1:{api_port}",
        build_env={"NEXT_PUBLIC_API_URL": "same-origin", "API_PROXY_TARGET": f"http://127.0.0.1:{api_port}"},
        api_env={"API_PUBLIC_URL": "", "LOCAL_LAN": "true", "CORS_ORIGINS": ",".join(origins)},
        phone_url=f"http://{lan_ip}:{web_port}",
    )


def wait_for(url: str, process: subprocess.Popen[bytes], seconds: int = 90) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # a Windows console that can't show a symbol must not stop the app
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Run CleanFrame AI on this computer (no external services).")
    parser.add_argument("--port", type=int, default=8000, help="API port (default 8000)")
    parser.add_argument("--web-port", type=int, default=3000, help="web app port (default 3000)")
    parser.add_argument("--data-dir", default=str(ROOT / "cleanframe-data"), help="where the database and videos are kept")
    parser.add_argument("--memory-mb", type=int, default=1024, help="memory the overlay measurement may use (default 1024; minimum 256)")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5), help="videos processed at the same time (default 1 = least memory)")
    parser.add_argument(
        "--generative-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="on by default: for logos that can't be removed any other way, use the offline LaMa model (about 400 MB once; it "
        "invents the missing detail). --no-generative-fill skips the download and leaves those logos refused.",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="let a phone or tablet on the same Wi-Fi use it: prints an address to open there. Anyone on that network can then reach "
        "the app, so use it at home, not on public Wi-Fi. Switching this on or off rebuilds the web app once.",
    )
    parser.add_argument("--no-web", action="store_true", help="API only (no Node.js needed)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the web app")
    args = parser.parse_args()

    print("\n  CleanFrame AI — local mode (no accounts, API keys or cloud services)\n")
    check_tools(web=not args.no_web)
    python = ensure_python_packages()
    heal_blocked_extensions(python)
    if args.lan and args.no_web:
        fail("--lan needs the web app; remove --no-web.")
    lan_ip = lan_address() if args.lan else None
    if args.lan and not lan_ip:
        fail("--lan needs this computer to be on a network (Wi-Fi or cable), and none was found. Connect, then run it again.")
    plan = make_plan(args.port, args.web_port, lan_ip, args.lan, socket.gethostname() if args.lan else "")
    api_origin, web_origin = plan.api_origin, plan.web_origin
    if not args.no_web:
        if args.rebuild:
            shutil.rmtree(FRONTEND / ".next", ignore_errors=True)
        ensure_web_build(plan)

    env = {
        **os.environ,
        "LOCAL_MODE": "true",
        "LOCAL_DATA_DIR": args.data_dir,
        **plan.api_env,
        "LOCAL_WORKERS": str(args.workers),
        "OVERLAY_MEMORY_MB": str(max(256, args.memory_mb)),
        "PYTHONUNBUFFERED": "1",
    }
    models = Path(args.data_dir) / "models"
    if args.generative_fill:
        env["AI_MODEL_PATH"] = str(ensure_generative_fill(python, models).resolve())
    else:  # a model installed earlier (or an ONNX LaMa file you saved here) is picked up without any option
        for name in ("big-lama.pt", "lama.onnx"):
            if (models / name).is_file():
                env["AI_MODEL_PATH"] = str((models / name).resolve())
                break
    if "AI_MODEL_PATH" in env:
        say("Generative fill is on: it is used only for areas the video itself can't supply, and the result says so.")
    processes: list[subprocess.Popen[bytes]] = []

    def stop(*_: object) -> None:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGINT, lambda *_: (stop(), sys.exit(0)))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(0)))

    say("Starting the API ...")
    api = subprocess.Popen([str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "warning"], cwd=BACKEND, env=env)  # noqa: S603
    processes.append(api)
    if not wait_for(f"{api_origin}/api/health", api):
        stop()
        fail("The API did not start. The messages above say why (a busy port can be changed with --port).")
    url = api_origin + "/docs"
    if not args.no_web:
        say("Starting the web app ...")
        web_env = {**os.environ, "NODE_OPTIONS": "--max-old-space-size=256", "NEXT_TELEMETRY_DISABLED": "1", "NODE_ENV": "production"}
        web = subprocess.Popen([shutil.which("npm") or "npm", "run", "start", "--", "-p", str(args.web_port), *(["-H", plan.web_host] if plan.web_host else [])], cwd=FRONTEND, env=web_env)  # noqa: S603
        processes.append(web)
        if not wait_for(web_origin, web):
            stop()
            fail("The web app did not start. The messages above say why (a busy port can be changed with --web-port).")
        url = web_origin

    print(f"\n  ✓ Ready:  {url}\n    Your data: {Path(args.data_dir).resolve()}\n    Stop with Ctrl+C.\n")
    if plan.phone_url:
        print(f"  On your phone (same Wi-Fi):  {plan.phone_url}")
        print("  Anyone on this network can open that address, so use it on your home network, not public Wi-Fi.")
        print("  If the phone can't connect, allow Python/Node through this computer's firewall when asked, or check the Wi-Fi\n  isn't a guest network that blocks devices from each other.\n")
    print("  Only remove overlays from content you own or have permission to edit.\n")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        say("A process stopped, shutting down.")
    finally:
        stop()


if __name__ == "__main__":
    main()
