"""Tracking through the API: upload a real video with a moving logo, mark it, track it, read the result."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import Mask, Project
from app.schemas.mask import MaskDocument
from app.services import tracking as tracking_service
from app.tracking.raster import rasterize
from app.tracking.shapes import segments_from_json, shape_at
from app.tracking.tracker import TrackingCancelled
from app.workers.tasks import EnqueueError
from tests.conftest import upload_video
from tests.synthetic_video import SyntheticVideo, make_video

FRAMES = 60


def logo_pose(frame: int) -> tuple[float, float, float, float]:
    return (80 + 3 * frame, 150 + 4 * frame, 1.0, 0.0)


@pytest.fixture(scope="module")
def logo_video(tmp_path_factory: pytest.TempPathFactory) -> SyntheticVideo:
    return make_video(
        tmp_path_factory.mktemp("logo") / "logo.mp4", width=320, height=568, frames=FRAMES, pose=logo_pose, sprite_size=(80, 48)
    )


@pytest.fixture
def project(user_client: TestClient, logo_video: SyntheticVideo) -> dict[str, Any]:
    result = upload_video(user_client, logo_video.path)
    assert result["status"] == "MASKING" and result["frame_count"] == FRAMES
    return result


def mask_doc(video: SyntheticVideo, frame: int = 0) -> dict[str, Any]:
    outline = video.normalised_outline(frame)
    xs, ys = [p[0] for p in outline], [p[1] for p in outline]
    rect = {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}
    return {"version": 1, "regions": [{"id": "region_1", "name": "Logo", "type": "rectangle", "keyframes": [{"frame": frame, "rect": rect}]}]}


@pytest.fixture
def marked(user_client: TestClient, project: dict[str, Any], logo_video: SyntheticVideo) -> dict[str, Any]:
    assert user_client.put(f"/api/projects/{project['id']}/mask", json=mask_doc(logo_video)).status_code == 200
    return project


def track(client: TestClient, project: dict[str, Any]):
    return client.post(f"/api/projects/{project['id']}/track")


def state(client: TestClient, project: dict[str, Any]) -> dict[str, Any]:
    response = client.get(f"/api/projects/{project['id']}/tracking")
    assert response.status_code == 200, response.text
    return response.json()


def project_status(client: TestClient, project: dict[str, Any]) -> str:
    return client.get(f"/api/projects/{project['id']}").json()["status"]


# --- the whole path ----------------------------------------------------------------------------


def test_tracking_follows_the_logo_and_returns_to_masking(
    user_client: TestClient, marked: dict[str, Any], logo_video: SyntheticVideo
) -> None:
    assert state(user_client, marked) == {
        "status": "none", "frames_done": 0, "frames_total": 0, "error": None, "started_at": None, "data": None,
    }

    started = track(user_client, marked)
    assert started.status_code == 202, started.text

    result = state(user_client, marked)
    assert result["status"] == "done" and result["error"] is None and result["started_at"]
    assert result["frames_total"] == FRAMES and result["frames_done"] == FRAMES
    assert project_status(user_client, marked) == "MASKING", "the person can keep editing afterwards"

    data = result["data"]
    assert data["frame_count"] == FRAMES and data["aspect"] == pytest.approx(568 / 320)
    assert [(s["anchor"], s["step"], s["until"]) for s in data["regions"]["region_1"]["segments"]] == [(0, 1, FRAMES - 1)]
    assert "lost_at" not in data["regions"]["region_1"]["segments"][0]

    # Score the stored motion against where the logo really is.
    document = MaskDocument.model_validate(user_client.get(f"/api/projects/{marked['id']}/mask").json()["mask_data"])
    segments = segments_from_json(data)["region_1"]
    overlaps = []
    for frame in range(FRAMES):
        shape = shape_at(document.regions[0], frame, segments, data["aspect"])
        got, truth = rasterize(shape, 320, 568), logo_video.outline_mask(frame)
        overlaps.append(np.count_nonzero((got > 0) & (truth > 0)) / np.count_nonzero((got > 0) | (truth > 0)))
    assert np.mean(overlaps) > 0.95 and min(overlaps) > 0.88


def test_the_mask_response_reports_the_tracking_status(user_client: TestClient, marked: dict[str, Any]) -> None:
    track(user_client, marked)
    assert user_client.get(f"/api/projects/{marked['id']}/mask").json()["tracking_status"] == "done"


def test_a_logo_that_disappears_is_reported_with_the_frame(user_client: TestClient, tmp_path: Path) -> None:
    video = make_video(
        tmp_path / "vanish.mp4", width=320, height=568, frames=FRAMES, pose=logo_pose, sprite_size=(80, 48), visible=lambda f: f < 30
    )
    project = upload_video(user_client, video.path)
    assert user_client.put(f"/api/projects/{project['id']}/mask", json=mask_doc(video)).status_code == 200
    track(user_client, project)
    segment = state(user_client, project)["data"]["regions"]["region_1"]["segments"][0]
    assert segment["lost_at"] in (30, 31) and segment["reason"]
    assert len(segment["transforms"]) == segment["lost_at"] - 1


def test_tracking_can_be_run_again_after_it_finished(user_client: TestClient, marked: dict[str, Any]) -> None:
    track(user_client, marked)
    assert track(user_client, marked).status_code == 202
    assert state(user_client, marked)["status"] == "done"


# --- preconditions -----------------------------------------------------------------------------


def test_nothing_to_track_without_saved_marks(user_client: TestClient, project: dict[str, Any]) -> None:
    response = track(user_client, project)
    assert response.status_code == 409 and "Mark at least one area" in response.json()["detail"]
    user_client.put(f"/api/projects/{project['id']}/mask", json={"version": 1, "regions": []})
    assert track(user_client, project).status_code == 409
    assert project_status(user_client, project) == "MASKING"


def test_tracking_needs_a_project_that_is_ready_for_masking(
    user_client: TestClient, marked: dict[str, Any], db: Session
) -> None:
    db.execute(update(Project).where(Project.id == uuid.UUID(marked["id"])).values(status="COMPLETED"))
    db.commit()
    response = track(user_client, marked)
    assert response.status_code == 409 and "ready for masking" in response.json()["detail"]


def test_a_second_start_while_running_is_refused_and_the_marks_are_locked(
    user_client: TestClient, marked: dict[str, Any], db: Session, logo_video: SyntheticVideo
) -> None:
    db.execute(update(Project).where(Project.id == uuid.UUID(marked["id"])).values(status="TRACKING"))
    db.execute(update(Mask).where(Mask.project_id == uuid.UUID(marked["id"])).values(tracking_status="running", tracking_started_at=datetime.now(UTC)))
    db.commit()
    again = track(user_client, marked)
    assert again.status_code == 409 and "already running" in again.json()["detail"]
    saved = user_client.put(f"/api/projects/{marked['id']}/mask", json=mask_doc(logo_video))
    assert saved.status_code == 409, "marks can't change under a running job"
    assert state(user_client, marked)["status"] == "running"


def test_saving_new_marks_discards_the_results(
    user_client: TestClient, marked: dict[str, Any], logo_video: SyntheticVideo
) -> None:
    track(user_client, marked)
    assert state(user_client, marked)["data"] is not None
    assert user_client.put(f"/api/projects/{marked['id']}/mask", json=mask_doc(logo_video)).status_code == 200
    after = state(user_client, marked)
    assert after["status"] == "none" and after["data"] is None and after["frames_done"] == 0 and after["started_at"] is None


# --- failure handling --------------------------------------------------------------------------


def test_an_unreachable_queue_leaves_the_project_untouched(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(_: uuid.UUID) -> None:
        raise EnqueueError("no broker")

    monkeypatch.setattr("app.api.tracking.enqueue_tracking", down)
    response = track(user_client, marked)
    assert response.status_code == 503 and "couldn't start tracking" in response.json()["detail"]
    assert project_status(user_client, marked) == "MASKING"
    assert state(user_client, marked)["status"] == "none"
    monkeypatch.undo()
    assert track(user_client, marked).status_code == 202


def test_a_crashing_job_fails_cleanly_and_can_be_retried(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: Any, **__: Any) -> None:
        raise RuntimeError("codec exploded: /secret/path")

    monkeypatch.setattr(tracking_service, "track_document", explode)
    track(user_client, marked)
    result = state(user_client, marked)
    assert result["status"] == "failed" and result["error"] == tracking_service.GENERIC_FAILURE
    assert "secret" not in str(result), "internal error text must not reach the person"
    assert result["data"] is None
    assert project_status(user_client, marked) == "MASKING"

    monkeypatch.undo()
    track(user_client, marked)
    assert state(user_client, marked)["status"] == "done"


def test_a_worker_that_died_is_reported_after_the_limit_and_not_before(
    user_client: TestClient, marked: dict[str, Any], db: Session
) -> None:
    pid = uuid.UUID(marked["id"])

    def pretend_running(started: datetime) -> None:
        db.execute(update(Project).where(Project.id == pid).values(status="TRACKING"))
        db.execute(update(Mask).where(Mask.project_id == pid).values(tracking_status="running", tracking_started_at=started))
        db.commit()

    pretend_running(datetime.now(UTC) - timedelta(minutes=5))
    assert state(user_client, marked)["status"] == "running", "a long job isn't a lost job"

    pretend_running(datetime.now(UTC) - timedelta(hours=2))
    lost = state(user_client, marked)
    assert lost["status"] == "failed" and lost["error"] == tracking_service.WORKER_LOST
    assert project_status(user_client, marked) == "MASKING"


def test_a_job_whose_project_was_deleted_stops_quietly(
    user_client: TestClient, marked: dict[str, Any], db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = uuid.UUID(marked["id"])
    db.execute(update(Project).where(Project.id == pid).values(status="TRACKING"))
    db.execute(update(Mask).where(Mask.project_id == pid).values(tracking_status="running", tracking_started_at=datetime.now(UTC)))
    db.commit()
    seen: list[bool] = []

    def tracker_that_gets_cancelled(document: MaskDocument, video: Path, *, progress: Any, **_: Any) -> dict[str, Any]:
        assert user_client.delete(f"/api/projects/{marked['id']}").status_code == 204  # person deletes it mid-run
        keep_going = progress(10, 60)
        seen.append(keep_going)
        if not keep_going:
            raise TrackingCancelled
        raise AssertionError("tracking should have been told to stop")

    monkeypatch.setattr(tracking_service, "track_document", tracker_that_gets_cancelled)
    # The delete route refuses nothing for a TRACKING project, so this exercises a real mid-run delete.
    tracking_service.run_tracking(pid)
    assert seen == [False]
    db.expire_all()
    assert db.get(Project, pid) is None and db.scalar(select(Mask).where(Mask.project_id == pid)) is None


def test_a_job_with_nothing_waiting_does_nothing(user_client: TestClient, marked: dict[str, Any], db: Session) -> None:
    tracking_service.run_tracking(uuid.UUID(marked["id"]))  # project is MASKING, mask is 'none'
    tracking_service.run_tracking(uuid.uuid4())  # no such project
    assert state(user_client, marked)["status"] == "none"


# --- access ------------------------------------------------------------------------------------


def test_other_people_cannot_start_or_read_tracking(
    user_client: TestClient, other_client: TestClient, marked: dict[str, Any]
) -> None:
    track(user_client, marked)
    assert track(other_client, marked).status_code == 404
    assert other_client.get(f"/api/projects/{marked['id']}/tracking").status_code == 404
    assert state(user_client, marked)["status"] == "done", "their attempt changed nothing"


def test_tracking_endpoints_require_a_session(client: TestClient) -> None:
    pid = uuid.uuid4()
    assert client.post(f"/api/projects/{pid}/track").status_code == 401
    assert client.get(f"/api/projects/{pid}/tracking").status_code == 401
