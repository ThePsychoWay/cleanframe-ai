"""S3-compatible object storage (AWS S3, MinIO, ...).

Large uploads go straight from the browser to storage using presigned multipart URLs, so video
bytes never pass through the API. The API only creates the upload, signs parts and completes it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

PRESIGN_TTL_SECONDS = 3600


class StorageError(RuntimeError):
    """Storage couldn't do what was asked (unreachable, missing upload, rejected part list)."""


@dataclass(frozen=True)
class UploadedPart:
    part_number: int
    etag: str
    size: int


class ObjectStorage(ABC):
    """What the rest of the application needs from file storage. S3Storage (any S3-compatible store) and
    LocalStorage (a folder on this computer, for local mode) implement it."""

    @abstractmethod
    def ensure_bucket(self) -> None: ...

    @abstractmethod
    def apply_lifecycle(self, *, expire_after_days: int, abort_uploads_after_days: int) -> None: ...

    @abstractmethod
    def lifecycle_rules(self) -> list[dict]: ...

    @abstractmethod
    def create_multipart_upload(self, key: str, content_type: str) -> str: ...

    @abstractmethod
    def presign_part(self, key: str, upload_id: str, part_number: int) -> str: ...

    @abstractmethod
    def list_parts(self, key: str, upload_id: str) -> list[UploadedPart]: ...

    @abstractmethod
    def complete_multipart_upload(self, key: str, upload_id: str, parts: list[tuple[int, str]]) -> None: ...

    @abstractmethod
    def abort_multipart_upload(self, key: str, upload_id: str) -> None: ...

    @abstractmethod
    def object_size(self, key: str) -> int | None: ...

    @abstractmethod
    def read_head(self, key: str, length: int) -> bytes: ...

    @abstractmethod
    def download_to(self, key: str, destination: Path) -> None: ...

    @abstractmethod
    def upload_file(self, source: Path, key: str, content_type: str) -> None: ...

    @abstractmethod
    def presign_get(self, key: str, *, expires_in: int = PRESIGN_TTL_SECONDS, download_as: str | None = None) -> str: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class S3Storage(ObjectStorage):
    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.storage_bucket
        self._region = settings.storage_region
        common = {
            "aws_access_key_id": settings.storage_access_key or None,
            "aws_secret_access_key": settings.storage_secret_key or None,
            "region_name": settings.storage_region,
            "config": Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
        }
        internal = settings.storage_endpoint or None
        public = settings.storage_public_endpoint or internal
        self._client = boto3.client("s3", endpoint_url=internal, **common)
        # Presigned URLs embed the host, so they're signed against the address the browser will use.
        self._presigner = self._client if public == internal else boto3.client("s3", endpoint_url=public, **common)

    # -- bucket -----------------------------------------------------------------------------

    def ensure_bucket(self) -> None:
        """Create the bucket if it's missing. Used in development; production buckets are provisioned."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
                raise StorageError(f"Can't reach storage bucket '{self._bucket}'.") from error
            kwargs = {} if self._region == "us-east-1" else {
                "CreateBucketConfiguration": {"LocationConstraint": self._region}
            }
            self._client.create_bucket(Bucket=self._bucket, **kwargs)
            logger.info("storage bucket created", extra={"bucket": self._bucket})

    def apply_lifecycle(self, *, expire_after_days: int, abort_uploads_after_days: int) -> None:
        """Make the bucket remove what the application should already have removed: unfinished uploads, and anything
        under our prefixes that outlived retention (a worker that died, a delete that failed). Replaces the bucket's
        lifecycle configuration with exactly these rules."""
        rules: list[dict] = [
            {
                "ID": "abort-unfinished-uploads",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": abort_uploads_after_days},
            }
        ]
        for prefix in ("uploads/", "derived/", "outputs/"):
            rules.append(
                {
                    "ID": f"expire-{prefix.rstrip('/')}",
                    "Status": "Enabled",
                    "Filter": {"Prefix": prefix},
                    "Expiration": {"Days": expire_after_days},
                }
            )
        try:
            self._client.put_bucket_lifecycle_configuration(Bucket=self._bucket, LifecycleConfiguration={"Rules": rules})
        except ClientError as error:
            raise StorageError("The bucket's lifecycle rules couldn't be set (does the storage key allow it?).") from error
        logger.info("storage lifecycle rules set", extra={"expire_days": expire_after_days, "abort_days": abort_uploads_after_days})

    def lifecycle_rules(self) -> list[dict]:
        try:
            return list(self._client.get_bucket_lifecycle_configuration(Bucket=self._bucket)["Rules"])
        except ClientError as error:
            if error.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
                return []
            raise StorageError("The bucket's lifecycle rules couldn't be read.") from error

    # -- multipart upload -------------------------------------------------------------------

    def create_multipart_upload(self, key: str, content_type: str) -> str:
        try:
            response = self._client.create_multipart_upload(Bucket=self._bucket, Key=key, ContentType=content_type)
        except ClientError as error:
            raise StorageError("Couldn't start the upload.") from error
        return str(response["UploadId"])

    def presign_part(self, key: str, upload_id: str, part_number: int) -> str:
        return self._presigner.generate_presigned_url(
            "upload_part",
            Params={"Bucket": self._bucket, "Key": key, "UploadId": upload_id, "PartNumber": part_number},
            ExpiresIn=PRESIGN_TTL_SECONDS,
        )

    def list_parts(self, key: str, upload_id: str) -> list[UploadedPart]:
        parts: list[UploadedPart] = []
        marker = 0
        try:
            while True:
                response = self._client.list_parts(
                    Bucket=self._bucket, Key=key, UploadId=upload_id, PartNumberMarker=marker, MaxParts=1000
                )
                parts.extend(
                    UploadedPart(p["PartNumber"], p["ETag"], p["Size"]) for p in response.get("Parts", [])
                )
                if not response.get("IsTruncated"):
                    return parts
                marker = response["NextPartNumberMarker"]
        except ClientError as error:
            raise StorageError("Couldn't read the upload's progress.") from error

    def complete_multipart_upload(self, key: str, upload_id: str, parts: list[tuple[int, str]]) -> None:
        try:
            self._client.complete_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": [{"PartNumber": n, "ETag": etag} for n, etag in sorted(parts)]},
            )
        except ClientError as error:
            raise StorageError("Storage rejected the uploaded parts.") from error

    def abort_multipart_upload(self, key: str, upload_id: str) -> None:
        try:
            self._client.abort_multipart_upload(Bucket=self._bucket, Key=key, UploadId=upload_id)
        except ClientError as error:
            # Already completed or aborted is fine; anything else is worth a log line.
            if error.response["Error"]["Code"] != "NoSuchUpload":
                logger.warning("abort multipart failed", extra={"key": key})

    # -- objects ----------------------------------------------------------------------------

    def object_size(self, key: str) -> int | None:
        try:
            return int(self._client.head_object(Bucket=self._bucket, Key=key)["ContentLength"])
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise StorageError("Couldn't read the stored file.") from error

    def read_head(self, key: str, length: int) -> bytes:
        """The first `length` bytes of a stored file (a ranged read: the rest isn't transferred)."""
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key, Range=f"bytes=0-{length - 1}")
            return bytes(response["Body"].read())
        except ClientError as error:
            raise StorageError("Couldn't read the uploaded file.") from error

    def download_to(self, key: str, destination: Path) -> None:
        try:
            self._client.download_file(self._bucket, key, str(destination))
        except ClientError as error:
            raise StorageError("Couldn't read the uploaded file.") from error

    def upload_file(self, source: Path, key: str, content_type: str) -> None:
        try:
            self._client.upload_file(str(source), self._bucket, key, ExtraArgs={"ContentType": content_type})
        except ClientError as error:
            raise StorageError("Couldn't store the generated file.") from error

    def presign_get(self, key: str, *, expires_in: int = PRESIGN_TTL_SECONDS, download_as: str | None = None) -> str:
        """A time-limited link for the browser to play or show a stored file. Anyone holding it can read it.

        With `download_as` the link makes the browser save the file under that name instead of playing it. The
        name must already be safe (see app.api.result.download_name)."""
        params: dict[str, str] = {"Bucket": self._bucket, "Key": key}
        if download_as:
            params["ResponseContentDisposition"] = f'attachment; filename="{download_as}"'
        return self._presigner.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_in)

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except ClientError as error:
            raise StorageError("Couldn't delete the stored file.") from error


@lru_cache
def get_storage() -> ObjectStorage:
    settings = get_settings()
    if settings.storage_backend == "filesystem":
        from app.storage.local import LocalStorage

        return LocalStorage.from_settings(settings)
    return S3Storage(settings)
