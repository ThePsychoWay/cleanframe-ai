"""The fully automatic path: upload -> find the overlays -> start the render, with nobody marking anything.

Runs inside the analysis job, right after the preview is made and while the original is still on local disk. What it
finds is saved as an ordinary mask document (source "auto"), so the render, the editor and exports treat it exactly like
marks a person drew, and a person can still open the editor to adjust it.

Everything recorded in `project.automation` is something that happened. When nothing is found the project is simply
left waiting in the editor with that said plainly: a video is never "processed" without a removal behind it.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.ai.registry import describe_engine
from app.config import get_settings
from app.detect.build import document_for
from app.detect.service import removable, survey_and_detect
from app.detect.types import DetectionReport
from app.models import Job, Mask, Project, ProjectStatus
from app.models.job import JOB_QUEUED, JOB_RENDER
from app.schemas.mask import MaskDocument
from app.video.source import inspect_source

logger = logging.getLogger(__name__)

STATE_NOTHING = "nothing_found"
STATE_RENDERING = "rendering"
STATE_FAILED = "detection_failed"
STATE_NO_ENGINE = "engine_unavailable"
STATE_READY = "ready"  # found something; auto-start is off


def detect_project(local: Path, frame_count: int) -> tuple[MaskDocument | None, dict[str, Any]]:
    """Run detection on the original. Returns the mask document (None when nothing was found) and the record to store."""
    started = time.monotonic()
    media = inspect_source(local)
    survey, report = survey_and_detect(local, media, frame_count, appearance=get_settings().auto_appearance)
    keep = removable(report)
    doc = document_for(keep, survey) if keep else None
    record = summarise(report, survey.frames, len(keep))
    record["seconds"] = round(time.monotonic() - started, 2)
    return doc, record


def summarise(report: DetectionReport, frames: Any, removing: int) -> dict[str, Any]:
    overlays = []
    for o in report.overlays:
        where = [int(i) for i in range(len(o.present)) if o.present[i]]
        overlays.append(
            {
                "id": o.id, "kind": o.kind, "decision": "remove" if o.verdict == "remove" else "left",
                "reason": o.reason, "basis": o.basis, "confidence": o.confidence,
                "first_frame": int(frames[where[0]]) if where else None, "last_frame": int(frames[where[-1]]) if where else None,
                "frames_present": o.frames_present, "notes": o.notes,
            }
        )  # fmt: skip
    return {
        "version": 1, "found": len(report.overlays), "removing": removing, "overlays": overlays,
        "notes": report.notes, "timings": report.timings, "samples": report.samples, "stride": report.stride,
    }  # fmt: skip


def apply_detection(db: Session, project: Project, doc: MaskDocument | None, record: dict[str, Any]) -> bool:
    """Store what detection found. Returns True when a render should now be started."""
    if doc is None or not doc.regions:
        record["state"] = STATE_NOTHING
        project.automation = record
        return False
    first, last = doc.frame_span()
    existing = db.scalar(select(Mask).where(Mask.project_id == project.id))
    if existing is not None and existing.source == "manual":
        record["state"] = STATE_READY  # a person's own marks always win
        project.automation = record
        return False
    data = doc.model_dump(mode="json")
    if existing is None:
        db.add(Mask(project_id=project.id, mask_data=data, frame_start=first, frame_end=last, source="auto"))
    else:
        existing.mask_data, existing.frame_start, existing.frame_end, existing.source = data, first, last, "auto"
    record["state"] = STATE_READY if not get_settings().auto_process else STATE_RENDERING
    project.automation = record
    return get_settings().auto_process


def start_render(db: Session, project: Project) -> Job | None:
    """Queue the final render of the saved (automatic) marks, the way the Remove button would. None if it can't start."""
    from app.workers.tasks import EnqueueError, enqueue_render, queue_for  # the task module imports analysis, which imports this

    engine = describe_engine()
    if not engine.available or (engine.sends_frames_to_third_party):
        # A remote engine needs the person's consent to send frames out: never taken automatically.
        state = STATE_NO_ENGINE
        reason = engine.reason or ("the engine sends frames to an outside service, which needs your confirmation" if engine.available else "not set up")
        record = dict(project.automation or {})
        record.update(state=state, state_reason=f"The reconstruction engine isn't ready: {reason}")
        project.automation = record
        return None
    claimed = db.execute(
        update(Project).where(Project.id == project.id, Project.status == ProjectStatus.MASKING.value).values(status=ProjectStatus.PROCESSING.value)
    ).rowcount
    if not claimed:
        return None
    job = Job(
        project_id=project.id, user_id=project.user_id, type=JOB_RENDER, status=JOB_QUEUED, queue=queue_for(engine.provider),
        frames_total=project.frame_count or 0, settings={"quality": "balanced", "container": "mp4", "keep_audio": True},
        engine=engine.provider, development_only=engine.development_only, tracking_used=False,
    )  # fmt: skip
    db.add(job)
    db.commit()
    try:
        enqueue_render(job.id, job.queue)
    except EnqueueError:
        logger.exception("couldn't enqueue the automatic render", extra={"job_id": str(job.id)})
        from datetime import UTC, datetime

        from app.services.rendering import restore_project_status

        job.status, job.error, job.completed_at = "failed", "The job queue isn't reachable.", datetime.now(UTC)
        restore_project_status(db, project.id)
        db.commit()
        return None
    return job


def new_job_id() -> uuid.UUID:  # pragma: no cover - kept for symmetry with API helpers
    return uuid.uuid4()
