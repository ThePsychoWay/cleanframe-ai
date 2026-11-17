"""Background jobs on threads inside the API process, for local mode (TASK_BACKEND=thread). No Redis, no worker process.

Jobs run one after another by default (LOCAL_WORKERS=1), which is also what keeps memory low: a render and the next
upload's analysis take turns. The same task functions the Celery workers run are used, so behaviour is identical, only
the queue is different. The retention sweep runs on its own timer thread.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_stop = threading.Event()
_sweeper: threading.Thread | None = None


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=get_settings().local_workers, thread_name_prefix="cf-job")
        return _executor


def _guarded(name: str, function: Callable[[], object]) -> None:
    from app.logging_config import clear_context, start_context

    start_context(task=name)
    try:
        function()
    except Exception:  # a job handles its own failures; this is the last resort, so the thread survives
        logger.exception("background job crashed", extra={"job": name})
    finally:
        clear_context()


def submit(name: str, function: Callable[[], object]) -> None:
    """Queue a job. Returns at once; the job runs when a thread is free."""
    _pool().submit(_guarded, name, function)


INTERRUPTED = "The app was closed while this was running. Please start it again."
INTERRUPTED_UPLOAD = "The app was closed before this video was checked. Please upload it again."


def recover_interrupted() -> int:
    """Nothing survives a restart on threads, so work that was queued or running when the app closed can never finish:
    report it failed now (with a message the person can act on) instead of leaving it to time out hours later."""
    from datetime import UTC, datetime

    from sqlalchemy import select, update

    from app.db import get_sessionmaker
    from app.models import Job, Mask, Project, ProjectStatus
    from app.models.job import JOB_FAILED, JOB_QUEUED, JOB_RUNNING
    from app.models.mask import TRACKING_FAILED, TRACKING_RUNNING
    from app.services.rendering import restore_project_status

    count = 0
    with get_sessionmaker()() as db:
        job_rows = db.execute(select(Job.id, Job.project_id).where(Job.status.in_((JOB_QUEUED, JOB_RUNNING)))).all()
        for job_id, project_id in job_rows:
            db.execute(update(Job).where(Job.id == job_id).values(status=JOB_FAILED, error=INTERRUPTED, completed_at=datetime.now(UTC)))
            restore_project_status(db, project_id)
            count += 1
        for mask in db.scalars(select(Mask).where(Mask.tracking_status == TRACKING_RUNNING)):
            mask.tracking_status, mask.tracking_error = TRACKING_FAILED, INTERRUPTED
            project = db.get(Project, mask.project_id)
            if project is not None and project.status == ProjectStatus.TRACKING.value:
                project.status = ProjectStatus.MASKING.value
            count += 1
        stuck = (ProjectStatus.UPLOADED.value, ProjectStatus.ANALYZING.value)
        for project in db.scalars(select(Project).where(Project.status.in_(stuck))):
            project.status, project.error = ProjectStatus.FAILED.value, INTERRUPTED_UPLOAD
            count += 1
        db.commit()
    if count:
        logger.warning("work interrupted by the last shutdown was marked failed", extra={"count": count})
    return count


def start_sweeper() -> None:
    """Run the retention sweep every CLEANUP_INTERVAL_SECONDS on a timer thread (once per process)."""
    global _sweeper
    with _lock:
        if _sweeper is not None and _sweeper.is_alive():
            return
        _stop.clear()
        _sweeper = threading.Thread(target=_sweep_loop, name="cf-sweeper", daemon=True)
        _sweeper.start()


def _sweep_loop() -> None:
    from app.workers.tasks import cleanup_expired

    interval = float(get_settings().cleanup_interval_seconds)
    while not _stop.wait(interval):
        _guarded("cleanup_expired", cleanup_expired.run)


def shutdown() -> None:
    """Stop accepting work and let the job in progress finish (called when the API stops)."""
    global _executor, _sweeper
    _stop.set()
    with _lock:
        pool, _executor, _sweeper = _executor, None, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
