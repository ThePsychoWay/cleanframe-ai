"""File storage on this computer's disk, for local mode. It behaves like the S3 storage the rest of the app expects,
including resumable multi-part uploads and time-limited links, with no server to run.

The browser still sends upload parts and fetches videos by link; here the links point at this API
(app/api/local_files.py), and each carries a signed, expiring token naming exactly one operation on one file. Keys are
built by the server from ids, but are checked anyway: nothing can leave the storage folder.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from app.config import Settings
from app.storage.s3 import PRESIGN_TTL_SECONDS, ObjectStorage, StorageError, UploadedPart

logger = logging.getLogger(__name__)

UPLOAD_LINK_SECONDS = 3600


def sign(payload: dict, secret: str) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).rstrip(b"=")
    key = hashlib.sha256(b"cleanframe-local-files:" + secret.encode()).digest()
    mac = base64.urlsafe_b64encode(hmac.new(key, body, hashlib.sha256).digest()).rstrip(b"=")
    return f"{body.decode()}.{mac.decode()}"


def verify(token: str, secret: str, *, operation: str) -> dict:
    """The token's payload, or StorageError when it is forged, expired or for another operation."""
    try:
        body, mac = token.split(".", 1)
        key = hashlib.sha256(b"cleanframe-local-files:" + secret.encode()).digest()
        expected = base64.urlsafe_b64encode(hmac.new(key, body.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
        if not hmac.compare_digest(mac, expected):
            raise ValueError
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except (ValueError, json.JSONDecodeError) as error:
        raise StorageError("This link isn't valid.") from error
    if payload.get("op") != operation or float(payload.get("exp", 0)) < time.time():
        raise StorageError("This link has expired or isn't valid.")
    return payload


class LocalStorage(ObjectStorage):
    def __init__(self, root: Path, *, secret: str, public_url: str, max_part_bytes: int) -> None:
        self.root = root.resolve()
        self._files = self.root / "files"
        self._parts = self.root / "uploads-in-progress"
        self._secret = secret
        self._public = public_url.rstrip("/")
        self.max_part_bytes = max_part_bytes

    @classmethod
    def from_settings(cls, settings: Settings) -> LocalStorage:
        return cls(
            Path(settings.local_data_dir).expanduser() / "storage",
            secret=settings.jwt_secret,
            public_url=settings.api_public_url,
            max_part_bytes=settings.upload_part_size + 1024 * 1024,
        )

    # -- paths ---------------------------------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        path = (self._files / key).resolve()
        if not key or key.startswith("/") or ".." in Path(key).parts or not path.is_relative_to(self._files):
            raise StorageError("That file name isn't allowed.")
        return path

    def _upload_dir(self, upload_id: str) -> Path:
        if not upload_id.isalnum():
            raise StorageError("That upload isn't valid.")
        return self._parts / upload_id

    # -- bucket --------------------------------------------------------------------------------------

    def ensure_bucket(self) -> None:
        self._files.mkdir(parents=True, exist_ok=True)
        self._parts.mkdir(parents=True, exist_ok=True)

    def apply_lifecycle(self, *, expire_after_days: int, abort_uploads_after_days: int) -> None:
        """Nothing to configure for a folder. Old files are removed by the application's own retention sweep."""
        logger.info("storage lifecycle rules don't apply to the local disk storage")

    def lifecycle_rules(self) -> list[dict]:
        return []

    # -- multipart upload ----------------------------------------------------------------------------

    def create_multipart_upload(self, key: str, content_type: str) -> str:
        self.path_for(key)
        self.ensure_bucket()
        upload_id = uuid.uuid4().hex
        directory = self._upload_dir(upload_id)
        directory.mkdir()
        (directory / "meta.json").write_text(json.dumps({"key": key, "content_type": content_type}), encoding="utf-8")
        return upload_id

    def presign_part(self, key: str, upload_id: str, part_number: int) -> str:
        token = sign({"op": "put", "k": key, "u": upload_id, "n": part_number, "exp": time.time() + UPLOAD_LINK_SECONDS}, self._secret)
        return f"{self._public}/api/local-files/upload?token={token}"

    def part_path(self, upload_id: str, part_number: int) -> Path:
        return self._upload_dir(upload_id) / f"{part_number:05d}.part"

    def store_part_etag(self, upload_id: str, part_number: int, etag: str) -> None:
        (self._upload_dir(upload_id) / f"{part_number:05d}.etag").write_text(etag, encoding="utf-8")

    def list_parts(self, key: str, upload_id: str) -> list[UploadedPart]:
        directory = self._upload_dir(upload_id)
        if not directory.is_dir():
            raise StorageError("Couldn't read the upload's progress.")
        parts = []
        for part in sorted(directory.glob("*.part")):
            etag_file = part.with_suffix(".etag")
            if etag_file.is_file():  # a part still being written has no etag yet and doesn't count
                parts.append(UploadedPart(int(part.stem), etag_file.read_text(encoding="utf-8"), part.stat().st_size))
        return parts

    def complete_multipart_upload(self, key: str, upload_id: str, parts: list[tuple[int, str]]) -> None:
        destination = self.path_for(key)
        directory = self._upload_dir(upload_id)
        have = {p.part_number: p for p in self.list_parts(key, upload_id)}
        ordered = sorted(parts)
        if not ordered or any(number not in have or have[number].etag != etag for number, etag in ordered):
            raise StorageError("Storage rejected the uploaded parts.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        assembling = destination.with_name(destination.name + ".assembling")
        try:
            with assembling.open("wb") as out:
                for number, _ in ordered:
                    with self.part_path(upload_id, number).open("rb") as part:
                        shutil.copyfileobj(part, out, 1024 * 1024)
            assembling.replace(destination)  # atomic: nobody sees a half-assembled file
        except OSError as error:
            assembling.unlink(missing_ok=True)
            raise StorageError("Couldn't store the uploaded file.") from error
        shutil.rmtree(directory, ignore_errors=True)

    def abort_multipart_upload(self, key: str, upload_id: str) -> None:
        shutil.rmtree(self._upload_dir(upload_id), ignore_errors=True)

    # -- objects -------------------------------------------------------------------------------------

    def object_size(self, key: str) -> int | None:
        path = self.path_for(key)
        return path.stat().st_size if path.is_file() else None

    def read_head(self, key: str, length: int) -> bytes:
        try:
            with self.path_for(key).open("rb") as handle:
                return handle.read(length)
        except OSError as error:
            raise StorageError("Couldn't read the uploaded file.") from error

    def download_to(self, key: str, destination: Path) -> None:
        source = self.path_for(key)
        if not source.is_file():
            raise StorageError("Couldn't read the uploaded file.")
        destination.unlink(missing_ok=True)
        try:
            os.symlink(source, destination)  # nothing is copied: the worker reads the stored file in place
        except OSError:
            try:
                shutil.copyfile(source, destination)
            except OSError as error:
                raise StorageError("Couldn't read the uploaded file.") from error

    def upload_file(self, source: Path, key: str, content_type: str) -> None:
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".writing")
        try:
            shutil.copyfile(source, partial)
            partial.replace(destination)
        except OSError as error:
            partial.unlink(missing_ok=True)
            raise StorageError("Couldn't store the generated file.") from error

    def presign_get(self, key: str, *, expires_in: int = PRESIGN_TTL_SECONDS, download_as: str | None = None) -> str:
        payload: dict = {"op": "get", "k": key, "exp": time.time() + expires_in}
        if download_as:
            payload["as"] = download_as
        return f"{self._public}/api/local-files/get?token={quote(sign(payload, self._secret))}"

    def delete(self, key: str) -> None:
        try:
            self.path_for(key).unlink(missing_ok=True)
        except OSError as error:
            raise StorageError("Couldn't delete the stored file.") from error

    # -- used by the file routes ---------------------------------------------------------------------

    def verify(self, token: str, *, operation: str) -> dict:
        return verify(token, self._secret, operation=operation)
