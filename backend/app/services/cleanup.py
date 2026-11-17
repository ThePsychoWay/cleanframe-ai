"""Retention: deleting what people uploaded once its time is up.

Every project has an `expires_at` (upload time + VIDEO_RETENTION_HOURS; pushed out when an upload finishes and
when a render finishes, so a result always stays downloadable for the full period). Two things enforce it:

* a periodic sweep (`expire_due_projects`, run by Celery beat) that handles everything that is due, and
* a lazy check whenever somebody reads a project (`app.api.projects._owned_project`), so access stops at the
  deadline even if the sweep is late or isn't running.

Expiring a project deletes every stored file it owns (the original, the preview, the thumbnail, every render
output, an unfinished upload's parts), then marks it EXPIRED and drops its jobs and marks. The row stays so the
person sees what happened; deleting it from the dashboard still works. If storage refuses a delete, nothing is
marked: the project stays due and the next sweep tries again, so the UI never says "deleted" for files that still
exist. A project that is being worked on right now (a live job, tracking, analysis) is left alone until the work
ends, or until it has been silent longer than any job can legitimately run.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Job, Mask, Project, ProjectStatus
from app.models.job import ACTIVE_JOB_STATUSES
from app.storage.keys import output_key, proxy_key, thumbnail_key
from app.storage.s3 import ObjectStorage, StorageError

logger = logging.getLogger(__name__)

# Statuses in which a worker may be touching the project's files without an active Job row.
WORKING_STATUSES = (
    ProjectStatus.ANALYZING.value,
    ProjectStatus.TRACKING.value,
    ProjectStatus.PROCESSING.value,
    ProjectStatus.ENCODING.value,
)
# Slack added to the longest job limit before a silent project is treated as abandoned.
BUSY_GRACE_SECONDS = 600
TEMP_PREFIXES = ("cf-render-", "cf-tracking-", "cf-analysis-")

SessionFactory = Callable[[], Session]


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@dataclass
class SweepReport:
    expired: int = 0
    busy: int = 0
    failed: int = 0
    temp_dirs_removed: int = 0


def retention_deadline(now: datetime | None = None) -> datetime:
    return (now or _now()) + timedelta(hours=get_settings().video_retention_hours)


def extend_retention(project: Project, now: datetime | None = None) -> None:
    """Give the project the full retention period from now (never shortens it)."""
    deadline = retention_deadline(now)
    if _aware(project.expires_at) < deadline:
        project.expires_at = deadline


def is_due(project: Project, now: datetime | None = None) -> bool:
    return project.status != ProjectStatus.EXPIRED.value and _aware(project.expires_at) <= (now or _now())


def project_storage_keys(db: Session, project: Project) -> list[str]:
    """Every object this project may own. Derived keys are deterministic, so they're included even if analysis or a
    render stopped before recording them."""
    keys = [project.original_file_key, proxy_key(project.user_id, project.id), thumbnail_key(project.user_id, project.id)]
    if project.output_file_key:
        keys.append(project.output_file_key)
    for job in db.scalars(select(Job).where(Job.project_id == project.id)):
        container = str(job.settings.get("container", "mp4"))
        keys.append(job.output_file_key or output_key(project.user_id, project.id, job.id, container))
    return list(dict.fromkeys(keys))


def delete_project_files(db: Session, storage: ObjectStorage, project: Project) -> None:
    """Remove every stored file of the project. Raises StorageError if storage refuses; nothing else is changed."""
    if project.status == ProjectStatus.UPLOADING.value and project.upload_id:
        storage.abort_multipart_upload(project.original_file_key, project.upload_id)
    for key in project_storage_keys(db, project):
        storage.delete(key)


def is_busy(db: Session, project: Project, now: datetime | None = None) -> bool:
    """True while a worker is plausibly still working on the project."""
    settings = get_settings()
    now = now or _now()
    window = timedelta(seconds=max(settings.render_time_limit_seconds, settings.tracking_time_limit_seconds) + BUSY_GRACE_SECONDS)
    if project.status in WORKING_STATUSES and _aware(project.updated_at) > now - window:
        return True
    jobs = db.scalars(select(Job).where(Job.project_id == project.id, Job.status.in_(ACTIVE_JOB_STATUSES)))
    return any(_aware(job.updated_at) > now - window for job in jobs)


def expire_project(db: Session, storage: ObjectStorage, project: Project, now: datetime | None = None) -> bool:
    """Delete the project's files and mark it EXPIRED. Returns False (changing nothing) if it's busy or not due.

    Raises StorageError when storage refuses a delete."""
    now = now or _now()
    if not is_due(project, now) or is_busy(db, project, now):
        return False

    delete_project_files(db, storage, project)

    changed = db.execute(
        update(Project)
        .where(Project.id == project.id, Project.status != ProjectStatus.EXPIRED.value)
        .values(
            status=ProjectStatus.EXPIRED.value,
            error=None,
            output_file_key=None,
            proxy_file_key=None,
            thumbnail_file_key=None,
            upload_id=None,
        )
    ).rowcount
    if changed:
        db.execute(delete(Job).where(Job.project_id == project.id))
        db.execute(delete(Mask).where(Mask.project_id == project.id))
    db.commit()
    db.refresh(project)
    if changed:
        logger.info("project expired", extra={"project_id": str(project.id), "user_id": str(project.user_id)})
    return bool(changed)


def expire_due_projects(
    session_factory: SessionFactory, storage: ObjectStorage, *, now: datetime | None = None, limit: int = 200
) -> SweepReport:
    """One pass of the retention sweep. Safe to run from several places at once: every step is idempotent."""
    started = time.monotonic()
    now = now or _now()
    report = SweepReport()
    with session_factory() as db:
        due = list(
            db.scalars(
                select(Project)
                .where(Project.expires_at <= now, Project.status != ProjectStatus.EXPIRED.value)
                .order_by(Project.expires_at)
                .limit(limit)
            )
        )
        for project in due:
            try:
                if expire_project(db, storage, project, now):
                    report.expired += 1
                else:
                    report.busy += 1
            except StorageError:
                db.rollback()
                report.failed += 1
                logger.warning("couldn't delete an expired project's files; will retry", extra={"project_id": str(project.id)}, exc_info=True)
            except Exception:
                db.rollback()
                report.failed += 1
                logger.exception("retention sweep failed for a project", extra={"project_id": str(project.id)})

    report.temp_dirs_removed = remove_stale_temp_dirs()
    logger.info(
        "retention sweep finished",
        extra={
            "expired": report.expired,
            "busy": report.busy,
            "failed": report.failed,
            "temp_dirs_removed": report.temp_dirs_removed,
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return report


def stale_temp_age_seconds() -> int:
    """A working folder older than any job may run is a leftover from a worker that was killed."""
    settings = get_settings()
    return max(settings.render_time_limit_seconds, settings.tracking_time_limit_seconds) + 3600


def remove_stale_temp_dirs(base: Path | None = None, *, older_than_seconds: int | None = None, now: float | None = None) -> int:
    """Delete this application's leftover working folders (cf-render-*, cf-tracking-*, cf-analysis-*) from the
    worker temp directory. Only folders this code creates, only when old. Returns how many were removed."""
    import tempfile

    settings = get_settings()
    root = base or Path(settings.worker_tmp_dir or tempfile.gettempdir())
    cutoff = (now if now is not None else time.time()) - (older_than_seconds if older_than_seconds is not None else stale_temp_age_seconds())
    removed = 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.startswith(TEMP_PREFIXES) or entry.is_symlink() or not entry.is_dir():
            continue
        try:
            if entry.stat().st_mtime > cutoff:
                continue
            shutil.rmtree(entry)
            removed += 1
        except OSError:
            logger.warning("couldn't remove a leftover working folder", extra={"folder": entry.name}, exc_info=True)
    return removed

