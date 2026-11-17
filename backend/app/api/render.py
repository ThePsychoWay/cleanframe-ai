"""Final render: start a job, read its progress (polling or a live stream), cancel it.

Everything is scoped to the signed-in user (see projects._owned_project). The render uses the *saved* marks and
tracking, so the project can't be edited while one runs (the editor only saves in MASKING).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, update

from app.ai.registry import describe_engine
from app.api.editor import _load_mask
from app.api.projects import Db, _owned_project
from app.auth.deps import CurrentUser
from app.config import get_settings
from app.db import get_sessionmaker
from app.models import Job, Project, ProjectStatus
from app.models.job import (
    ACTIVE_JOB_STATUSES,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RENDER,
    JOB_RUNNING,
)
from app.models.mask import TRACKING_DONE
from app.ratelimit import per_user
from app.schemas.render import JobOut, RenderRequest, RenderStatusResponse
from app.services.render_view import to_out
from app.services.rendering import expire_if_lost, restore_project_status
from app.workers.tasks import EnqueueError, enqueue_render, queue_for

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["render"])

STARTABLE = (ProjectStatus.MASKING.value, ProjectStatus.COMPLETED.value)
STREAM_POLL_SECONDS = 1.0
STREAM_KEEPALIVE_SECONDS = 15.0
STREAM_MAX_SECONDS = 1800  # the browser reconnects (EventSource does it by itself) and gets the current state


def _latest_job(db: Db, project_id: uuid.UUID) -> Job | None:
    return db.scalar(select(Job).where(Job.project_id == project_id, Job.type == JOB_RENDER).order_by(Job.created_at.desc()).limit(1))


def _view(db: Db, project: Project, job: Job | None) -> RenderStatusResponse:
    if job is None:
        return RenderStatusResponse(job=None)
    expire_if_lost(db, job)
    db.refresh(project)
    return RenderStatusResponse(job=to_out(db, job, project))


@router.post(
    "/{project_id}/render",
    response_model=RenderStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[per_user("render_start")],
)
def start_render(project_id: uuid.UUID, body: RenderRequest, user: CurrentUser, db: Db) -> RenderStatusResponse:
    project = _owned_project(db, user.id, project_id)
    if project.status == ProjectStatus.PROCESSING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "This video is already being processed.")
    if project.status not in STARTABLE:
        raise HTTPException(status.HTTP_409_CONFLICT, "The video isn't ready to process yet.")
    mask = _load_mask(db, project.id)
    if mask is None or not mask.mask_data.get("regions"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Mark at least one area and save before processing.")
    if not project.frame_count:
        raise HTTPException(status.HTTP_409_CONFLICT, "The video hasn't been analysed yet.")

    engine = describe_engine()
    if not engine.available:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"The reconstruction engine isn't ready: {engine.reason or 'not set up'}"
        )
    if engine.sends_frames_to_third_party and not body.acknowledge_third_party:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"This sends your video frames to an outside service ({engine.name}). Confirm to continue.",
        )

    active = db.scalar(
        select(func.count()).select_from(Job).where(Job.user_id == user.id, Job.type == JOB_RENDER, Job.status.in_(ACTIVE_JOB_STATUSES))
    )
    if (active or 0) >= get_settings().max_active_jobs_per_user:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "You already have videos being processed. Wait for one to finish first."
        )

    # Claim atomically so two clicks (or two tabs) can't start two jobs.
    claimed = db.execute(
        update(Project)
        .where(Project.id == project.id, Project.status.in_(STARTABLE))
        .values(status=ProjectStatus.PROCESSING.value)
    ).rowcount
    if not claimed:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "This video is already being processed.")

    job = Job(
        project_id=project.id,
        user_id=user.id,
        type=JOB_RENDER,
        status=JOB_QUEUED,
        queue=queue_for(engine.provider),
        frames_total=project.frame_count,
        settings=body.model_dump(exclude={"acknowledge_third_party"}),
        engine=engine.provider,
        development_only=engine.development_only,
        tracking_used=mask.tracking_status == TRACKING_DONE,
    )
    db.add(job)
    db.commit()
    logger.info(
        "render requested",
        extra={"job_id": str(job.id), "project_id": str(project.id), "user_id": str(user.id), "queue": job.queue},
    )

    try:
        enqueue_render(job.id, job.queue)
    except EnqueueError:
        logger.exception("couldn't enqueue render", extra={"job_id": str(job.id)})
        job.status, job.error, job.completed_at = JOB_FAILED, "The job queue isn't reachable.", datetime.now(UTC)
        restore_project_status(db, project.id)
        db.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "We couldn't start processing. Try again in a moment."
        ) from None

    db.expire_all()  # an inline (eager) job may already have finished in another session
    project = _owned_project(db, user.id, project_id)
    return _view(db, project, db.get(Job, job.id))


@router.get("/{project_id}/render", response_model=RenderStatusResponse)
def get_render(project_id: uuid.UUID, user: CurrentUser, db: Db) -> RenderStatusResponse:
    project = _owned_project(db, user.id, project_id)
    return _view(db, project, _latest_job(db, project.id))


@router.post("/{project_id}/render/cancel", response_model=RenderStatusResponse)
def cancel_render(project_id: uuid.UUID, user: CurrentUser, db: Db) -> RenderStatusResponse:
    project = _owned_project(db, user.id, project_id)
    job = _latest_job(db, project.id)
    if job is None or job.status not in ACTIVE_JOB_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "Nothing is being processed.")

    if job.status == JOB_QUEUED:
        # No worker has it: stop it here. A worker that picks the message up later finds it isn't waiting.
        stopped = db.execute(
            update(Job).where(Job.id == job.id, Job.status == JOB_QUEUED).values(
                status=JOB_CANCELLED, cancel_requested=True, completed_at=datetime.now(UTC)
            )
        ).rowcount
        if stopped:
            restore_project_status(db, project.id)
            db.commit()
            logger.info("render cancelled while queued", extra={"job_id": str(job.id), "user_id": str(user.id)})
            db.expire_all()
            return _view(db, _owned_project(db, user.id, project_id), db.get(Job, job.id))
        db.rollback()
        db.refresh(job)
    if job.status == JOB_RUNNING:
        # The worker sees the flag within about a second, stops, cleans up and sets the final state itself.
        db.execute(update(Job).where(Job.id == job.id, Job.status == JOB_RUNNING).values(cancel_requested=True))
        db.commit()
        logger.info("render cancel requested", extra={"job_id": str(job.id), "user_id": str(user.id)})
    db.expire_all()
    return _view(db, _owned_project(db, user.id, project_id), db.get(Job, job.id))


@router.post("/{project_id}/reopen", response_model=RenderStatusResponse)
def reopen_for_editing(project_id: uuid.UUID, user: CurrentUser, db: Db) -> RenderStatusResponse:
    """Let the person change the marks of a finished video. The finished file is kept (and stays downloadable)
    until a new render replaces it."""
    project = _owned_project(db, user.id, project_id)
    moved = db.execute(
        update(Project).where(Project.id == project.id, Project.status == ProjectStatus.COMPLETED.value).values(status=ProjectStatus.MASKING.value)
    ).rowcount
    db.commit()
    if not moved:
        raise HTTPException(status.HTTP_409_CONFLICT, "Only a finished video can be reopened for editing.")
    db.expire_all()
    return _view(db, _owned_project(db, user.id, project_id), _latest_job(db, project.id))


def _snapshot(project_id: uuid.UUID) -> str | None:
    """The latest job as JSON, read in its own short session."""
    with get_sessionmaker()() as db:
        job = db.scalar(select(Job).where(Job.project_id == project_id, Job.type == JOB_RENDER).order_by(Job.created_at.desc()).limit(1))
        project = db.get(Project, project_id)
        if job is None or project is None:
            return None
        expire_if_lost(db, job)
        db.refresh(project)
        return to_out(db, job, project).model_dump_json()


async def _events(request: Request, project_id: uuid.UUID) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    started = loop.time()
    last_sent = ""
    last_beat = started
    yield "retry: 3000\n\n"
    while loop.time() - started < STREAM_MAX_SECONDS:
        if await request.is_disconnected():
            return
        current = await run_in_threadpool(_snapshot, project_id)
        if current is None:
            yield "event: none\ndata: {}\n\n"
            return
        if current != last_sent:
            last_sent, last_beat = current, loop.time()
            yield f"event: job\ndata: {current}\n\n"
            if json.loads(current)["status"] not in ACTIVE_JOB_STATUSES:
                return
        elif loop.time() - last_beat >= STREAM_KEEPALIVE_SECONDS:
            last_beat = loop.time()
            yield ": keep-alive\n\n"
        await asyncio.sleep(STREAM_POLL_SECONDS)


@router.get("/{project_id}/render/events")
def render_events(project_id: uuid.UUID, request: Request, user: CurrentUser, db: Db) -> StreamingResponse:
    """Server-sent events: the job's state whenever it changes, ending when the job does. The first event is the
    current state, so a page that reconnects is immediately up to date."""
    project = _owned_project(db, user.id, project_id)
    return StreamingResponse(
        _events(request, project.id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


__all__ = ["JobOut", "router"]
