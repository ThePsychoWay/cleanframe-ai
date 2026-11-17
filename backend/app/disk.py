"""Checks that there is room on disk before a big file is written, so a job fails at the start with a clear message
instead of half way through with a cryptic one."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app.config import get_settings

MIB = 1024 * 1024


class InsufficientStorage(Exception):
    """Not enough free disk space. The message is safe to show to the person."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def free_bytes(path: str | Path) -> int:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def ensure_free_space(path: str | Path, needed_bytes: int, *, what: str = "this video") -> None:
    """Raise InsufficientStorage unless `needed_bytes` plus the configured reserve (MIN_FREE_DISK_MB) are free at `path`."""
    reserve = get_settings().min_free_disk_mb * MIB
    available = free_bytes(path)
    if available < needed_bytes + reserve:
        raise InsufficientStorage(
            f"There isn't enough free disk space on the server to work on {what} "
            f"(needs about {(needed_bytes + reserve) // MIB} MB, {available // MIB} MB free). "
            "Free some space or try a smaller video."
        )


def work_dir_root() -> Path:
    configured = get_settings().worker_tmp_dir
    return Path(configured) if configured else Path(tempfile.gettempdir())


def ensure_work_space(source_bytes: int, *, factor: float, what: str = "this video") -> None:
    """Room for a job's temporary files: roughly `factor` times the size of the file it starts from."""
    ensure_free_space(work_dir_root(), int(source_bytes * factor), what=what)
