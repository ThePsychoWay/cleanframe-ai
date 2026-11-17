"""Model files: checked against a published SHA-256 before use, downloaded once if missing.

A model file is code the server will run, so it is never used on trust. The checksum is required whenever a download
URL is configured, the download goes over HTTPS only (plain HTTP to this machine in development), it is written to a
temporary file and moved into place only after the checksum matches, and a file that fails the check is deleted.
URL and checksum come from server configuration, never from a request.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from pathlib import Path

import httpx

from app.ai.base import EngineUnavailableError
from app.ai.remote import url_problem

logger = logging.getLogger(__name__)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024
CHUNK = 1024 * 1024
_lock = threading.Lock()
_verified: dict[tuple[str, int, int, str], bool] = {}


def normalise_sha256(value: str) -> str:
    return value.strip().lower()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def config_problem(url: str, sha256: str, *, allow_insecure_loopback: bool) -> str | None:
    """Why a download can't be configured this way, or None."""
    if sha256 and not SHA256_PATTERN.match(normalise_sha256(sha256)):
        return "AI_MODEL_SHA256 must be 64 hexadecimal characters."
    if url:
        if not sha256:
            return "AI_MODEL_URL needs AI_MODEL_SHA256 so the downloaded file can be verified."
        return url_problem(url, allow_insecure_loopback=allow_insecure_loopback)
    return None


def ensure_weights(path: str, url: str = "", sha256: str = "", *, allow_insecure_loopback: bool = False) -> Path:
    """Return the verified model file, downloading it first when it is missing and a URL is configured."""
    if not path:
        raise EngineUnavailableError("No model file is configured (AI_MODEL_PATH).")
    target = Path(path)
    expected = normalise_sha256(sha256)
    with _lock:
        if target.is_file():
            if expected:
                _check(target, expected)
            return target
        if not url:
            raise EngineUnavailableError("The configured model file wasn't found.")
        problem = config_problem(url, sha256, allow_insecure_loopback=allow_insecure_loopback)
        if problem:
            raise EngineUnavailableError(problem)
        _download(url, target, expected)
        return target


def _check(target: Path, expected: str) -> None:
    stat = target.stat()
    key = (str(target), stat.st_size, stat.st_mtime_ns, expected)
    if _verified.get(key):
        return
    if sha256_of(target) != expected:
        raise EngineUnavailableError("The model file doesn't match its published checksum, so it won't be used.")
    _verified[key] = True


def _download(url: str, target: Path, expected: str) -> None:
    logger.info("downloading the model file", extra={"destination": target.name})
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    size = 0
    try:
        # Redirects are refused: the checksum protects the content, but not where a redirect could send the request.
        with httpx.stream("GET", url, follow_redirects=False, timeout=httpx.Timeout(30.0, read=120.0)) as response:
            if response.status_code != 200:
                raise EngineUnavailableError(f"The model couldn't be downloaded (the server answered {response.status_code}).")
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK):
                    size += len(chunk)
                    if size > MAX_MODEL_BYTES:
                        raise EngineUnavailableError("The model file is larger than the allowed size.")
                    digest.update(chunk)
                    handle.write(chunk)
        if digest.hexdigest() != expected:
            raise EngineUnavailableError("The downloaded model doesn't match its published checksum, so it was discarded.")
        partial.replace(target)  # atomic: other workers never see a half-written file
    except EngineUnavailableError:
        raise
    except (httpx.HTTPError, OSError) as error:
        logger.warning("model download failed: %s", type(error).__name__)
        raise EngineUnavailableError("The model couldn't be downloaded. Check AI_MODEL_URL and the network.") from error
    finally:
        partial.unlink(missing_ok=True)
