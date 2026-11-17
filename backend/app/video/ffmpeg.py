"""Running FFmpeg safely for the render pipeline.

Arguments are always a list (no shell), only the plain `file` protocol is allowed so a crafted container can't make
FFmpeg fetch URLs or read other local files, and each child process dies with this one (Linux), so a worker that is
killed at its time limit doesn't leave an encoder running.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

GENERIC = "The video couldn't be processed. Please try again, or upload another copy."


class ProcessingError(Exception):
    """Rendering failed. The message is safe to show to the person."""

    def __init__(self, message: str = GENERIC) -> None:
        super().__init__(message)
        self.message = message


class ProcessingCancelled(Exception):
    """The job was cancelled (the project was deleted, or the person stopped it) while rendering."""


def ffmpeg_command(args: list[str]) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-protocol_whitelist", "file", *args]


def _die_with_parent() -> None:  # pragma: no cover - runs in the child between fork and exec
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        except Exception:
            pass


def run_ffmpeg(args: list[str], *, timeout: float, failure: str = GENERIC) -> None:
    """Run FFmpeg to completion. Raises ProcessingError (with `failure`) if it fails or takes too long."""
    try:
        completed = subprocess.run(
            ffmpeg_command(args),
            capture_output=True,
            timeout=timeout,
            check=False,
            text=True,
            preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,
        )
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out after %s s", timeout)
        raise ProcessingError("Processing took too long and was stopped.") from None
    if completed.returncode != 0:
        logger.warning("ffmpeg failed: %s", completed.stderr.strip()[-800:])
        raise ProcessingError(failure)


class FFmpegProcess:
    """A long-running FFmpeg child with a raw-video pipe on one side. Its error output goes to a temporary file
    (never a pipe nobody reads, which would eventually block FFmpeg)."""

    def __init__(self, args: list[str], *, stdin: bool = False, stdout: bool = False) -> None:
        self._stderr: IO[bytes] = tempfile.TemporaryFile(prefix="cf-ffmpeg-")
        self.process = subprocess.Popen(
            ffmpeg_command(args),
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE if stdout else subprocess.DEVNULL,
            stderr=self._stderr,
            preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,
            bufsize=0,
        )

    @property
    def stdin(self) -> Any:
        return self.process.stdin

    @property
    def stdout(self) -> Any:
        return self.process.stdout

    def error_tail(self, limit: int = 800) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0, os.SEEK_END)
            size = self._stderr.tell()
            self._stderr.seek(max(0, size - limit))
            return self._stderr.read().decode("utf-8", "replace").strip()
        except Exception:
            return ""

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            logger.error("ffmpeg didn't exit after kill")
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass

    def close(self) -> None:
        self.kill()
        self._stderr.close()


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("couldn't remove a temporary file", exc_info=True)
