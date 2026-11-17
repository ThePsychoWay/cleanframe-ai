import uuid
from typing import Any

from app.config import get_settings
from app.db import get_sessionmaker
from app.services.analysis import analyze_project
from app.services.cleanup import expire_due_projects
from app.services.exporting import run_export
from app.services.rendering import run_render
from app.services.tracking import run_tracking
from app.storage.s3 import get_storage
from app.workers.celery_app import celery_app

_TRACKING_SOFT_LIMIT = get_settings().tracking_time_limit_seconds
_RENDER_SOFT_LIMIT = get_settings().render_time_limit_seconds


@celery_app.task(name="analyze_video")
def analyze_video(project_id: str) -> None:
    analyze_project(uuid.UUID(project_id))


# Tracking is much longer than analysis, so it gets its own limits (the app-wide ones are 240/300 s).
@celery_app.task(name="track_mask", soft_time_limit=_TRACKING_SOFT_LIMIT, time_limit=_TRACKING_SOFT_LIMIT + 120)
def track_mask(project_id: str) -> None:
    run_tracking(uuid.UUID(project_id))


# A render is the longest job. Idempotent: a message delivered twice (a worker died before acknowledging it)
# finds the job already claimed and does nothing, see services/rendering.py.
@celery_app.task(name="render_video", soft_time_limit=_RENDER_SOFT_LIMIT, time_limit=_RENDER_SOFT_LIMIT + 300)
def render_video(job_id: str) -> None:
    run_render(uuid.UUID(job_id))


# An export is one transcode of a finished video. Idempotent like a render: a second delivery finds the job claimed.
@celery_app.task(name="export_video", soft_time_limit=3600, time_limit=3900)
def export_video(job_id: str) -> None:
    run_export(uuid.UUID(job_id))


# Retention: delete what has passed its deadline. Scheduled by Celery beat (see celery_app.py); safe to run twice.
@celery_app.task(name="cleanup_expired", soft_time_limit=600, time_limit=660)
def cleanup_expired() -> dict[str, int]:
    report = expire_due_projects(get_sessionmaker(), get_storage())
    return {"expired": report.expired, "busy": report.busy, "failed": report.failed, "temp_dirs_removed": report.temp_dirs_removed}


class EnqueueError(RuntimeError):
    """The job queue couldn't be reached."""


def _enqueue(task: Any, project_id: uuid.UUID) -> None:
    if get_settings().task_backend == "thread":
        from app.workers import local

        local.submit(task.name, lambda: task.run(str(project_id)))
        return
    # Read the setting at call time so tests can run tasks inline without a broker.
    celery_app.conf.task_always_eager = get_settings().celery_eager
    try:
        task.delay(str(project_id))
    except Exception as error:  # kombu/redis raise several unrelated error types
        raise EnqueueError("The job queue isn't reachable.") from error


def enqueue_analysis(project_id: uuid.UUID) -> None:
    _enqueue(analyze_video, project_id)


def enqueue_tracking(project_id: uuid.UUID) -> None:
    _enqueue(track_mask, project_id)


GPU_PROVIDERS = frozenset({"local", "video"})


def queue_for(provider: str) -> str:
    """Engines that run a model on this machine want a GPU worker; the dev and API engines don't."""
    configured = get_settings().render_queue
    if configured != "auto":
        return configured
    return "gpu" if provider in GPU_PROVIDERS else "cpu"


def enqueue_render(job_id: uuid.UUID, queue: str) -> None:
    if get_settings().task_backend == "thread":
        from app.workers import local

        local.submit("render_video", lambda: render_video.run(str(job_id)))
        return
    celery_app.conf.task_always_eager = get_settings().celery_eager
    try:
        render_video.apply_async(args=[str(job_id)], queue=queue)
    except Exception as error:
        raise EnqueueError("The job queue isn't reachable.") from error


def enqueue_export(job_id: uuid.UUID) -> None:
    if get_settings().task_backend == "thread":
        from app.workers import local

        local.submit("export_video", lambda: export_video.run(str(job_id)))
        return
    celery_app.conf.task_always_eager = get_settings().celery_eager
    try:
        export_video.apply_async(args=[str(job_id)], queue="cpu")
    except Exception as error:
        raise EnqueueError("The job queue isn't reachable.") from error


__all__: list[str] = ["EnqueueError", "enqueue_analysis", "enqueue_export", "enqueue_render", "enqueue_tracking", "queue_for"]
