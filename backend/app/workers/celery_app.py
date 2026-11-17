"""Celery application. Workers consume the `cpu` queue; GPU work gets its own queue in Phase 9."""

from celery import Celery
from celery.signals import setup_logging, task_postrun, task_prerun, worker_ready

from app.config import get_settings


def make_celery() -> Celery:
    settings = get_settings()
    app = Celery("cleanframe", broker=settings.redis_url, include=["app.workers.tasks"])
    app.conf.update(
        task_default_queue="cpu",
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_connection_timeout=3,
        broker_connection_retry_on_startup=True,
        task_soft_time_limit=240,
        task_time_limit=300,
        # Retention sweep. Run one beat process (`celery -A app.workers.celery_app beat`); the task is idempotent, so a
        # second one by mistake only wastes work.
        beat_schedule={
            "retention-sweep": {"task": "cleanup_expired", "schedule": float(settings.cleanup_interval_seconds)},
        },
    )
    return app


celery_app = make_celery()


@worker_ready.connect
def _remove_leftover_working_folders(**_: object) -> None:
    """A worker killed mid-job leaves its working folder behind; clear old ones when a worker starts."""
    from app.services.cleanup import remove_stale_temp_dirs

    remove_stale_temp_dirs()


@setup_logging.connect
def _use_our_logging(**_: object) -> None:
    """Take over logging from Celery so workers write the same structured lines as the API."""
    from app.logging_config import configure_logging

    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)


@task_prerun.connect
def _bind_task(task_id: str | None = None, task: object = None, **_: object) -> None:
    from app.logging_config import start_context

    start_context(task_id=task_id, task=getattr(task, "name", None))


@task_postrun.connect
def _unbind_task(**_: object) -> None:
    from app.logging_config import clear_context

    clear_context()
