"""Sets up what the bucket itself should enforce:  python -m app.storage.provision

Run it once per environment (and again after changing retention). By default it never creates the bucket; production
buckets are provisioned by whoever owns the account. `--create-bucket` creates a missing bucket first (for a self-hosted
MinIO that starts empty); `--cors` sets the bucket's CORS rules to CORS_ORIGINS (for a bucket the browser uploads to). The storage key needs permission to set lifecycle configuration.
"""

from __future__ import annotations

import math
import sys

from app.config import Settings, get_settings
from app.storage.s3 import ObjectStorage, StorageError, get_storage


def expiry_days(settings: Settings) -> int:
    """Whole days after which the bucket removes an object: retention, rounded up, plus the margin."""
    return math.ceil(settings.video_retention_hours / 24) + settings.storage_lifecycle_margin_days


def provision(storage: ObjectStorage | None = None, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    (storage or get_storage()).apply_lifecycle(
        expire_after_days=expiry_days(settings), abort_uploads_after_days=settings.storage_abort_uploads_days
    )
    return expiry_days(settings)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        if "--create-bucket" in args:
            get_storage().ensure_bucket()
        if "--cors" in args:
            settings = get_settings()
            get_storage().apply_cors(settings.cors_origin_list)
        days = provision()
    except StorageError as error:
        print(f"storage provisioning failed: {error}", file=sys.stderr)
        return 1
    print(f"lifecycle rules set: objects expire {days} days after upload; unfinished uploads are discarded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
