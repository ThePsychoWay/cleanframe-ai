"""The routes that stand in for object storage in local mode: upload parts in, stored files out.

Only registered when STORAGE_BACKEND=filesystem. Every call needs a signed, expiring token made by the API for one
operation on one file (see app/storage/local.py); there is no other way in, so the routes need no sign-in.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from app.storage.local import LocalStorage
from app.storage.s3 import StorageError, get_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/local-files", tags=["local files"])


def _local() -> LocalStorage:
    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
    return storage


Local = Annotated[LocalStorage, Depends(_local)]


@router.put("/upload")
async def upload_part(request: Request, token: str, storage: Local) -> Response:
    try:
        grant = storage.verify(token, operation="put")
        path = storage.part_path(grant["u"], int(grant["n"]))
    except StorageError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    if not path.parent.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This upload no longer exists.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > storage.max_part_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "This part is larger than the upload part size.")

    digest, size = hashlib.md5(usedforsecurity=False), 0
    partial = path.with_name(path.name + ".writing")
    try:
        with partial.open("wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > storage.max_part_bytes:
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "This part is larger than the upload part size.")
                digest.update(chunk)
                out.write(chunk)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    etag = f'"{digest.hexdigest()}"'
    storage.store_part_etag(grant["u"], int(grant["n"]), etag)
    return Response(status_code=status.HTTP_200_OK, headers={"ETag": etag})


@router.get("/get")
def get_file(token: str, storage: Local) -> FileResponse:
    try:
        grant = storage.verify(token, operation="get")
        path = storage.path_for(grant["k"])
    except StorageError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This file no longer exists.")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"}
    name = grant.get("as")
    if name:
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return FileResponse(path, media_type=media_type, headers=headers)  # supports Range, which video playback needs
