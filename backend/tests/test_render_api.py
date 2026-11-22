"""Rendering as a background job, through the API: start, progress, cancel, failure, lost workers."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.ai.base import EngineInfo, InpaintingError
from app.ai.dev import DevInpaintingEngine
from app.errors import RESOURCES_EXHAUSTED
from app.models import Job, Project
from app.services import rendering
from app.storage.s3 import get_storage
from app.video.frames import FrameReader
from app.video.source import inspect_source
from app.workers.tasks import EnqueueError, queue_for
from tests.conftest import upload_video
from tests.render_media import child_processes
from tests.synthetic_video import SyntheticVideo, make_video
from tests.test_render_pipeline import with_audio
from tests.test_tracking_api import mask_doc

FRAMES = 30


def pose(frame: int) -> tuple[float, float, float, float]:
    return (230, 90, 1.0, 0.0)


@pytest.fixture(scope="module")
def source_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[SyntheticVideo, Path]:
    root = tmp_path_factory.mktemp("render_api")
    video = make_video(root / "silent.mp4", width=320, height=568, frames=FRAMES, pose=pose, sprite_size=(80, 48))
    return video, with_audio(video.path, root / "source.mp4", FRAMES / 30)


@pytest.fixture
def marked(user_client: TestClient, source_video: tuple[SyntheticVideo, Path]) -> dict[str, Any]:
    video, path = source_video
    project = upload_video(user_client, path)
    assert project["status"] == "MASKING" and project["frame_count"] == FRAMES
    assert user_client.put(f"/api/projects/{project['id']}/mask", json=mask_doc(video)).status_code == 200
    return project


@pytest.fixture
def queued_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept jobs but never run them: the state while every worker is busy."""
    monkeypatch.setattr("app.api.render.enqueue_render", lambda job_id, queue: None)


def render(client: TestClient, project: dict[str, Any], **body: Any):  # noqa: ANN202
    return client.post(f"/api/projects/{project['id']}/render", json=body)


def job_of(client: TestClient, project: dict[str, Any]) -> dict[str, Any]:
    response = client.get(f"/api/projects/{project['id']}/render")
    assert response.status_code == 200, response.text
    return response.json()["job"]


def project_status(client: TestClient, project: dict[str, Any]) -> str:
    return client.get(f"/api/projects/{project['id']}").json()["status"]


def stored_outputs() -> list[str]:
    storage = get_storage()
    listing = storage._client.list_objects_v2(Bucket=storage._bucket, Prefix="outputs/")  # noqa: SLF001
    return [item["Key"] for item in listing.get("Contents", [])]


# --- the whole path --------------------------------------------------------------------------------


def test_a_render_runs_in_the_background_and_leaves_a_checked_result(
    user_client: TestClient, marked: dict[str, Any], source_video: tuple[SyntheticVideo, Path], tmp_path: Path
) -> None:
    response = render(user_client, marked)
    assert response.status_code == 202
    job = response.json()["job"]  # eager mode: the worker already ran
    assert job["status"] == "completed" and job["percent"] == 100.0
    assert job["frames_done"] == job["frames_total"] == FRAMES
    assert job["engine"] == {"provider": "dev", "development_only": True}
    assert any("development engine" in w for w in job["warnings"])
    assert {s["key"]: s["state"] for s in job["stages"]} == {
        "analyze": "done", "mask": "done", "track": "skipped", "preparing": "done", "measuring": "skipped", "reconstructing": "done",
        "audio": "done", "verifying": "done", "saving": "done",
    }  # fmt: skip
    result = job["result"]
    assert (result["width"], result["height"], result["frames"]) == (320, 568, FRAMES)
    assert result["audio"] == "copied" and result["size_bytes"] > 0 and result["tracking_used"] is False
    assert project_status(user_client, marked) == "COMPLETED"

    # The stored file is the one the job describes, and it plays.
    [key] = stored_outputs()
    assert key.startswith("outputs/") and key.endswith(f"{job['id']}.mp4")
    assert get_storage().object_size(key) == result["size_bytes"]
    local = tmp_path / "result.mp4"
    get_storage().download_to(key, local)
    with FrameReader(local, inspect_source(local)) as reader:
        assert len(list(reader.frames())) == FRAMES


def test_the_chosen_settings_are_used_and_recorded(user_client: TestClient, marked: dict[str, Any]) -> None:
    job = render(user_client, marked, container="webm", quality="fast", keep_audio=False).json()["job"]
    assert job["status"] == "completed", job["error"]
    assert job["settings"] == {"quality": "fast", "container": "webm", "output_height": None, "codec": None, "keep_audio": False}
    assert job["result"]["container"] == "webm" and job["result"]["audio"] == "removed"
    audio = next(s for s in job["stages"] if s["key"] == "audio")
    assert audio["state"] == "skipped" and "removal" in audio["detail"]


def test_a_second_render_replaces_the_first_result(user_client: TestClient, marked: dict[str, Any]) -> None:
    first = render(user_client, marked).json()["job"]
    assert project_status(user_client, marked) == "COMPLETED"
    second = render(user_client, marked, quality="fast").json()["job"]
    assert second["status"] == "completed" and second["id"] != first["id"]
    [key] = stored_outputs()
    assert key.endswith(f"{second['id']}.mp4")  # the earlier file was removed once the new one was safe


def test_the_status_of_a_project_with_no_render_is_empty(user_client: TestClient, marked: dict[str, Any]) -> None:
    assert user_client.get(f"/api/projects/{marked['id']}/render").json() == {"job": None}


# --- who can do what -------------------------------------------------------------------------------


def test_other_peoples_projects_are_invisible(user_client: TestClient, other_client: TestClient, marked: dict[str, Any]) -> None:
    render(user_client, marked)
    pid = marked["id"]
    assert other_client.get(f"/api/projects/{pid}/render").status_code == 404
    assert other_client.post(f"/api/projects/{pid}/render", json={}).status_code == 404
    assert other_client.post(f"/api/projects/{pid}/render/cancel").status_code == 404
    assert other_client.get(f"/api/projects/{pid}/render/events").status_code == 404


def test_signing_in_is_required(client: TestClient) -> None:
    pid = uuid.uuid4()
    assert client.post(f"/api/projects/{pid}/render", json={}).status_code == 401
    assert client.get(f"/api/projects/{pid}/render").status_code == 401
    assert client.get(f"/api/projects/{pid}/render/events").status_code == 401


@pytest.mark.parametrize(
    "body",
    [{"quality": "ultra"}, {"container": "avi"}, {"output_height": 480}, {"keep_audio": "maybe"}, {"extra": 1}, {"output_height": "720p"}],
)
def test_bad_settings_are_refused(user_client: TestClient, marked: dict[str, Any], body: dict[str, Any]) -> None:
    assert render(user_client, marked, **body).status_code == 422
    assert project_status(user_client, marked) == "MASKING"


def test_rendering_needs_marks(user_client: TestClient, source_video: tuple[SyntheticVideo, Path]) -> None:
    project = upload_video(user_client, source_video[1])
    response = render(user_client, project)
    assert response.status_code == 409 and "Mark at least one area" in response.json()["detail"]


def test_rendering_needs_an_analysed_video(user_client: TestClient, db: Session, marked: dict[str, Any]) -> None:
    db.execute(update(Project).where(Project.id == uuid.UUID(marked["id"])).values(status="ANALYZING"))
    db.commit()
    assert render(user_client, marked).status_code == 409


def test_an_engine_that_is_not_ready_says_why_and_starts_nothing(
    user_client: TestClient, marked: dict[str, Any], env: pytest.MonkeyPatch, db: Session
) -> None:
    from app.config import get_settings

    env.setenv("AI_PROVIDER", "local")  # no model file configured
    get_settings.cache_clear()
    response = render(user_client, marked)
    assert response.status_code == 503 and "isn't ready" in response.json()["detail"]
    assert db.scalar(select(Job)) is None and project_status(user_client, marked) == "MASKING"


def test_an_engine_that_sends_frames_out_needs_consent(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch, queued_only: None
) -> None:
    info = EngineInfo("dev", "Outside service", True, False, True, None, None, "")
    monkeypatch.setattr("app.api.render.describe_engine", lambda: info)
    refused = render(user_client, marked)
    assert refused.status_code == 409 and "outside service" in refused.json()["detail"]
    assert project_status(user_client, marked) == "MASKING"
    assert render(user_client, marked, acknowledge_third_party=True).status_code == 202


def test_people_cannot_queue_without_limit(
    user_client: TestClient, marked: dict[str, Any], db: Session, env: pytest.MonkeyPatch, queued_only: None
) -> None:
    from app.config import get_settings

    env.setenv("MAX_ACTIVE_JOBS_PER_USER", "1")
    get_settings.cache_clear()
    me = db.get(Project, uuid.UUID(marked["id"]))
    assert me is not None
    db.add(Job(project_id=me.id, user_id=me.user_id, status="queued"))
    db.commit()
    response = render(user_client, marked)
    assert response.status_code == 429 and project_status(user_client, marked) == "MASKING"


def test_only_one_job_per_project_at_a_time(user_client: TestClient, marked: dict[str, Any], queued_only: None) -> None:
    assert render(user_client, marked).status_code == 202
    again = render(user_client, marked)
    assert again.status_code == 409 and "already being processed" in again.json()["detail"]


# --- waiting, cancelling ---------------------------------------------------------------------------


def test_a_waiting_job_says_so_honestly(user_client: TestClient, marked: dict[str, Any], queued_only: None) -> None:
    job = render(user_client, marked).json()["job"]
    assert job["status"] == "queued" and job["stage"] == "queued"
    assert job["percent"] is None and job["speed_fps"] is None and job["eta_seconds"] is None
    assert job["jobs_ahead"] == 0
    assert all(s["state"] in ("done", "skipped", "pending") for s in job["stages"])
    assert [s["state"] for s in job["stages"] if s["key"] == "reconstructing"] == ["pending"]
    assert project_status(user_client, marked) == "PROCESSING"


def test_cancelling_a_waiting_job_frees_the_project_and_the_worker_ignores_it_later(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session
) -> None:
    job = render(user_client, marked).json()["job"]
    cancelled = user_client.post(f"/api/projects/{marked['id']}/render/cancel").json()["job"]
    assert cancelled["status"] == "cancelled" and cancelled["completed_at"]
    assert project_status(user_client, marked) == "MASKING"
    rendering.run_render(uuid.UUID(job["id"]))  # the message arrives after all
    assert job_of(user_client, marked)["status"] == "cancelled" and stored_outputs() == []
    assert render(user_client, marked).status_code == 202  # and a new one can start


def test_cancelling_when_nothing_runs_is_an_error(user_client: TestClient, marked: dict[str, Any]) -> None:
    assert user_client.post(f"/api/projects/{marked['id']}/render/cancel").status_code == 409


class Scripted(DevInpaintingEngine):
    """The development engine, with something to do when it reaches a given frame."""

    def __init__(self, at: int, action: Any) -> None:
        self.at, self.action, self.seen = at, action, 0

    def process(self, frames: Any, masks: Any, options: Any):  # type: ignore[no-untyped-def]
        for result in super().process(frames, masks, options):
            self.seen += 1
            if self.seen == self.at:
                self.action()
            yield result


@pytest.fixture
def eager_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rendering, "PROGRESS_INTERVAL_SECONDS", 0.0)  # look at the job on every frame


def test_cancelling_a_running_job_stops_it_and_leaves_nothing(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch, eager_progress: None
) -> None:
    before = set(child_processes())
    engine = Scripted(5, lambda: user_client.post(f"/api/projects/{marked['id']}/render/cancel"))
    monkeypatch.setattr(rendering, "get_engine", lambda: engine)
    render(user_client, marked)
    job = job_of(user_client, marked)
    assert job["status"] == "cancelled" and job["error"] is None
    assert job["frames_done"] < FRAMES, "it stopped, it didn't finish"
    reconstruct = next(s for s in job["stages"] if s["key"] == "reconstructing")
    assert reconstruct["state"] == "stopped"
    assert project_status(user_client, marked) == "MASKING"
    assert stored_outputs() == [] and set(child_processes()) <= before


def test_deleting_the_project_mid_render_stops_the_worker_and_leaves_no_file(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch, eager_progress: None, db: Session
) -> None:
    engine = Scripted(5, lambda: user_client.delete(f"/api/projects/{marked['id']}"))
    monkeypatch.setattr(rendering, "get_engine", lambda: engine)
    render(user_client, marked)
    assert user_client.get(f"/api/projects/{marked['id']}").status_code == 404
    assert db.scalar(select(Job)) is None and stored_outputs() == []


# --- failing ---------------------------------------------------------------------------------------


def test_a_failure_the_engine_explains_reaches_the_person_and_the_project_is_editable_again(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise InpaintingError("The model ran out of memory on this frame.")

    monkeypatch.setattr(rendering, "get_engine", lambda: Scripted(3, fail))
    job = render(user_client, marked).json()["job"]
    assert job["status"] == "failed" and job["error"] == "The model ran out of memory on this frame."
    assert next(s for s in job["stages"] if s["key"] == "reconstructing")["state"] == "failed"
    assert project_status(user_client, marked) == "MASKING" and stored_outputs() == []

    monkeypatch.undo()
    assert render(user_client, marked).json()["job"]["status"] == "completed"


def test_an_unexpected_crash_never_shows_internal_details(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash() -> None:
        raise RuntimeError("segfault at /srv/secret/path with key AKIA123")

    monkeypatch.setattr(rendering, "get_engine", lambda: Scripted(2, crash))
    job = render(user_client, marked).json()["job"]
    assert job["status"] == "failed" and job["error"] == rendering.GENERIC_FAILURE
    assert "secret" not in json.dumps(job) and "AKIA" not in json.dumps(job)


def test_a_failed_render_keeps_the_earlier_result(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = render(user_client, marked).json()["job"]

    def crash() -> None:
        raise InpaintingError("no")

    monkeypatch.setattr(rendering, "get_engine", lambda: Scripted(2, crash))
    assert render(user_client, marked, quality="high").json()["job"]["status"] == "failed"
    assert project_status(user_client, marked) == "COMPLETED"  # the first result is still there
    [key] = stored_outputs()
    assert key.endswith(f"{first['id']}.mp4")


def test_a_queue_that_is_down_is_reported_and_leaves_the_project_alone(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch, db: Session
) -> None:
    def down(job_id: uuid.UUID, queue: str) -> None:
        raise EnqueueError("no broker")

    monkeypatch.setattr("app.api.render.enqueue_render", down)
    response = render(user_client, marked)
    assert response.status_code == 503 and "couldn't start processing" in response.json()["detail"]
    assert project_status(user_client, marked) == "MASKING"
    assert db.scalar(select(Job)).status == "failed"
    monkeypatch.undo()
    assert render(user_client, marked).json()["job"]["status"] == "completed"


# --- workers that vanish ---------------------------------------------------------------------------


def _pretend(db: Session, marked: dict[str, Any], **values: Any) -> None:
    db.execute(update(Job).where(Job.project_id == uuid.UUID(marked["id"])).values(**values))
    db.commit()
    db.expire_all()


def test_a_running_job_with_a_recent_heartbeat_is_left_alone(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session
) -> None:
    render(user_client, marked)
    now = datetime.now(UTC)
    _pretend(db, marked, status="running", stage="reconstructing", started_at=now - timedelta(hours=1), heartbeat_at=now - timedelta(seconds=30))
    assert job_of(user_client, marked)["status"] == "running", "a long job is not a lost job"


def test_a_running_job_whose_heartbeat_stopped_is_reported_failed(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session
) -> None:
    render(user_client, marked)
    old = datetime.now(UTC) - timedelta(minutes=10)
    _pretend(db, marked, status="running", stage="reconstructing", started_at=old, heartbeat_at=old)
    job = job_of(user_client, marked)
    assert job["status"] == "failed" and job["error"] == rendering.WORKER_LOST
    assert project_status(user_client, marked) == "MASKING"


def test_a_job_nobody_picked_up_is_reported_after_the_timeout(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session
) -> None:
    render(user_client, marked)
    assert job_of(user_client, marked)["status"] == "queued"
    _pretend(db, marked, created_at=datetime.now(UTC) - timedelta(hours=5))
    job = job_of(user_client, marked)
    assert job["status"] == "failed" and job["error"] == rendering.NOT_PICKED_UP
    assert project_status(user_client, marked) == "MASKING"


def test_a_message_delivered_twice_runs_once(user_client: TestClient, marked: dict[str, Any], db: Session) -> None:
    job = render(user_client, marked).json()["job"]
    [key] = stored_outputs()
    rendering.run_render(uuid.UUID(job["id"]))  # the broker redelivers it
    assert stored_outputs() == [key] and job_of(user_client, marked)["status"] == "completed"


# --- progress is measured --------------------------------------------------------------------------


def test_progress_writes_are_throttled_but_the_last_frame_and_stage_changes_always_land(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session
) -> None:
    from app.db import get_sessionmaker

    job_id = uuid.UUID(render(user_client, marked).json()["job"]["id"])
    _pretend(db, marked, status="running", stage="preparing")
    progress = rendering._Progress(get_sessionmaker(), job_id)  # noqa: SLF001
    assert progress("reconstructing", 1, 100) is True
    for done in range(2, 60):
        assert progress("reconstructing", done, 100) is True
    db.expire_all()
    row = db.get(Job, job_id)
    assert (row.frames_done, row.frames_total, row.stage) == (1, 100, "reconstructing")  # 58 calls, no extra writes
    assert row.reconstruct_started_at is not None

    assert progress("reconstructing", 100, 100) is True
    assert progress("audio", 0, 1) is True
    db.expire_all()
    row = db.get(Job, job_id)
    assert (row.frames_done, row.stage) == (100, "audio")


def test_progress_tells_the_pipeline_to_stop_when_the_job_is_no_longer_wanted(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session
) -> None:
    from app.db import get_sessionmaker

    job_id = uuid.UUID(render(user_client, marked).json()["job"]["id"])
    _pretend(db, marked, status="running", stage="reconstructing")
    progress = rendering._Progress(get_sessionmaker(), job_id)  # noqa: SLF001
    assert progress("reconstructing", 1, 10) is True
    _pretend(db, marked, cancel_requested=True)
    assert progress("audio", 0, 1) is False  # a stage change is always checked


def test_speed_and_time_remaining_come_from_frames_actually_finished() -> None:
    from app.services.render_view import speed_and_eta

    now = datetime.now(UTC)

    def running(done: int, seconds: float, stage: str = "reconstructing", status: str = "running") -> Job:
        return Job(status=status, stage=stage, frames_done=done, frames_total=300, reconstruct_started_at=now - timedelta(seconds=seconds))

    assert speed_and_eta(running(30, 10), now) == (3.0, 90.0)
    assert speed_and_eta(running(2, 10), now) == (None, None), "too early for an honest estimate"
    assert speed_and_eta(running(30, 10, stage="audio"), now) == (None, None)
    assert speed_and_eta(running(30, 10, status="completed"), now) == (None, None)
    assert speed_and_eta(Job(status="running", stage="reconstructing", frames_done=30, frames_total=300), now) == (None, None)


def test_the_heartbeat_keeps_writing_while_a_job_runs(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session
) -> None:
    from app.db import get_sessionmaker

    job_id = uuid.UUID(render(user_client, marked).json()["job"]["id"])
    old = datetime.now(UTC) - timedelta(minutes=1)
    _pretend(db, marked, status="running", heartbeat_at=old)
    with rendering.Heartbeat(get_sessionmaker(), job_id, 0.05):
        time.sleep(0.4)
    db.expire_all()
    beat = rendering.aware(db.get(Job, job_id).heartbeat_at)
    assert beat is not None and beat > old + timedelta(seconds=30)


# --- queues ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(("provider", "queue"), [("dev", "cpu"), ("api", "cpu"), ("local", "gpu"), ("video", "gpu")])
def test_engines_that_need_a_gpu_go_to_the_gpu_queue(provider: str, queue: str, env: pytest.MonkeyPatch) -> None:
    assert queue_for(provider) == queue


def test_the_queue_can_be_forced(env: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    env.setenv("RENDER_QUEUE", "gpu")
    get_settings.cache_clear()
    assert queue_for("dev") == "gpu"


# --- the live stream -------------------------------------------------------------------------------


def _events(client: TestClient, project: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    event = ""
    with client.stream("GET", f"/api/projects/{project['id']}/render/events") as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                found.append((event, json.loads(line.split(":", 1)[1])))
    return found


def test_the_stream_starts_with_the_current_state_and_ends_with_the_job(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.render.STREAM_POLL_SECONDS", 0.05)
    render(user_client, marked)  # finished already (eager)
    events = _events(user_client, marked)
    assert [name for name, _ in events] == ["job"]
    assert events[0][1]["status"] == "completed" and events[0][1]["percent"] == 100.0


def test_a_stream_for_a_project_with_no_render_says_so_and_closes(user_client: TestClient, marked: dict[str, Any]) -> None:
    assert _events(user_client, marked) == [("none", {})]


def test_the_stream_follows_a_job_as_it_changes(
    user_client: TestClient, marked: dict[str, Any], queued_only: None, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    monkeypatch.setattr("app.api.render.STREAM_POLL_SECONDS", 0.05)
    render(user_client, marked)

    def later() -> None:
        time.sleep(0.4)
        now = datetime.now(UTC)
        from app.db import get_sessionmaker

        with get_sessionmaker()() as session:
            session.execute(
                update(Job).values(status="running", stage="reconstructing", frames_done=10, frames_total=FRAMES, heartbeat_at=now, reconstruct_started_at=now)
            )
            session.commit()
        time.sleep(0.4)
        with get_sessionmaker()() as session:
            session.execute(update(Job).values(status="failed", error="stopped for the test", completed_at=datetime.now(UTC)))
            session.commit()

    thread = threading.Thread(target=later)
    thread.start()
    events = _events(user_client, marked)
    thread.join()
    states: list[tuple[str, str]] = []
    for _, data in events:  # while running, speed and time remaining change, so the same state can be sent again
        if not states or states[-1] != (data["status"], data["stage"]):
            states.append((data["status"], data["stage"]))
    assert states == [("queued", "queued"), ("running", "reconstructing"), ("failed", "reconstructing")]
    running = next(data for _, data in events if data["status"] == "running")
    assert running["percent"] == pytest.approx(100 * 10 / FRAMES, abs=0.1)


# --- cleanup ---------------------------------------------------------------------------------------


def test_deleting_a_project_removes_its_results_and_jobs(user_client: TestClient, marked: dict[str, Any], db: Session) -> None:
    render(user_client, marked)
    assert len(stored_outputs()) == 1
    assert user_client.delete(f"/api/projects/{marked['id']}").status_code == 204
    assert stored_outputs() == [] and db.scalar(select(Job)) is None


# --- going back to editing -------------------------------------------------------------------------


def test_a_finished_video_can_be_reopened_for_editing_and_keeps_its_result(user_client: TestClient, marked: dict[str, Any]) -> None:
    assert render(user_client, marked).status_code == 202
    assert project_status(user_client, marked) == "COMPLETED"
    stored = stored_outputs()
    assert len(stored) == 1

    reopened = user_client.post(f"/api/projects/{marked['id']}/reopen")
    assert reopened.status_code == 200 and reopened.json()["job"]["status"] == "completed"
    assert project_status(user_client, marked) == "MASKING"
    assert stored_outputs() == stored  # the finished file stays until a new render replaces it
    assert user_client.put(f"/api/projects/{marked['id']}/mask", json=user_client.get(f"/api/projects/{marked['id']}/mask").json()["mask_data"]).status_code == 200


def test_only_a_finished_video_can_be_reopened(user_client: TestClient, marked: dict[str, Any], other_client: TestClient) -> None:
    assert user_client.post(f"/api/projects/{marked['id']}/reopen").status_code == 409  # still MASKING
    assert render(user_client, marked).status_code == 202
    assert other_client.post(f"/api/projects/{marked['id']}/reopen").status_code == 404
    assert user_client.post(f"/api/projects/{marked['id']}/reopen").status_code == 200
    assert user_client.post(f"/api/projects/{marked['id']}/reopen").status_code == 409  # already reopened


# --- running out of resources ----------------------------------------------------------------------------------


@pytest.mark.parametrize("failure", [MemoryError(), OSError(28, "No space left on device"), OSError(12, "Cannot allocate memory")])
def test_running_out_of_memory_or_disk_is_explained_in_plain_words(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    def fail() -> None:
        raise failure

    monkeypatch.setattr(rendering, "get_engine", lambda: Scripted(2, fail))
    job = render(user_client, marked).json()["job"]
    assert job["status"] == "failed" and job["error"] == RESOURCES_EXHAUSTED
    assert "memory or disk" in job["error"] and "Traceback" not in job["error"]
    assert project_status(user_client, marked) == "MASKING", "the person can try again"


def test_a_resource_problem_hidden_inside_an_engine_error_is_still_recognised(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        try:
            raise MemoryError
        except MemoryError as cause:
            raise InpaintingError("The model failed on this frame.") from cause

    monkeypatch.setattr(rendering, "get_engine", lambda: Scripted(2, fail))
    assert render(user_client, marked).json()["job"]["error"] == RESOURCES_EXHAUSTED


def test_the_stage_list_names_the_measuring_step_only_for_the_engine_that_has_one() -> None:
    from types import SimpleNamespace

    from app.services.render_view import build_stages

    def stages(engine: str, stage: str) -> dict[str, str]:
        job = SimpleNamespace(tracking_used=False, stage=stage, status="running", engine=engine, settings={}, frames_total=0, frames_done=0)
        return {s.key: s.state for s in build_stages(job, SimpleNamespace(audio_codec=None))}  # type: ignore[arg-type]

    now = stages("auto", "measuring")
    assert now["preparing"] == "done" and now["measuring"] == "active" and now["reconstructing"] == "pending"
    assert stages("dev", "reconstructing")["measuring"] == "skipped"


# --- the automatic retry after a flagged check ---------------------------------------------------------


def _report(verdict: str, residual: float = 0.1, flicker: float = 1.0) -> dict[str, Any]:
    problems = [] if verdict == "clean" else ["part of the overlay may still be visible"]
    return {"checked": True, "regions": [{"name": "Logo", "verdict": verdict, "residual": residual, "flicker": flicker, "problems": problems}]}


def test_severity_orders_reports_and_never_prefers_an_unchecked_result_to_a_clean_one() -> None:
    from app.services.quality import retry_quality, severity

    assert severity(_report("clean")) == 0
    assert severity(_report("check", residual=0.9)) > severity(_report("check", residual=0.6)) > severity(_report("clean"))
    assert severity({"checked": False, "regions": []}) > severity(_report("clean"))
    assert severity({"checked": False, "regions": []}) < severity(_report("check", residual=0.6))
    assert retry_quality("balanced", _report("check")) == "high" and retry_quality("fast", _report("check")) == "high"
    assert retry_quality("high", _report("check")) is None  # nothing stronger to try
    assert retry_quality("balanced", _report("clean")) is None
    assert retry_quality("balanced", {"checked": False, "regions": []}) is None


def _scripted_quality(monkeypatch: pytest.MonkeyPatch, reports: list[dict[str, Any]]) -> list[str]:
    calls: list[str] = []
    queue = list(reports)

    def fake(original: Path, restored: Path, document: Any, tracking: Any, frames: int) -> dict[str, Any]:  # noqa: ANN401
        calls.append(restored.name)
        return queue.pop(0)

    monkeypatch.setattr(rendering, "_quality", fake)
    return calls


def test_a_flagged_render_is_rendered_again_and_the_better_one_is_kept(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _scripted_quality(monkeypatch, [_report("check", residual=0.8), _report("clean")])
    job = render(user_client, marked, quality="balanced").json()["job"]
    assert job["status"] == "completed", job["error"]
    assert len(calls) == 2 and len(set(calls)) == 2
    assert any("rendered again at high quality" in w and "second render checked better" in w for w in job["warnings"])
    assert job["result"]["quality"]["regions"][0]["verdict"] == "clean"
    assert not any("Look at it before" in w for w in job["warnings"])


def test_a_retry_that_checks_worse_keeps_the_first_render(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_quality(monkeypatch, [_report("check", residual=0.6), _report("check", residual=0.95)])
    job = render(user_client, marked, quality="balanced").json()["job"]
    assert job["status"] == "completed"
    assert any("did not check better, so the first one was kept" in w for w in job["warnings"])
    assert job["result"]["quality"]["regions"][0]["residual"] == 0.6
    assert any("Look at it before" in w for w in job["warnings"])  # the person still hears the first result was flagged


def test_a_clean_render_and_a_high_quality_render_are_not_repeated(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _scripted_quality(monkeypatch, [_report("clean")])
    assert render(user_client, marked, quality="balanced").json()["job"]["status"] == "completed"
    assert len(calls) == 1
    calls = _scripted_quality(monkeypatch, [_report("check", residual=0.9)])
    job = render(user_client, marked, quality="high").json()["job"]
    assert len(calls) == 1 and any("Look at it before" in w for w in job["warnings"])


def test_a_failing_retry_never_loses_the_first_result(
    user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = rendering.render_restored_video
    runs = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        runs["n"] += 1
        if runs["n"] == 2:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(rendering, "render_restored_video", flaky)
    _scripted_quality(monkeypatch, [_report("check", residual=0.7)])
    job = render(user_client, marked, quality="balanced").json()["job"]
    assert job["status"] == "completed" and runs["n"] == 2
    assert any("failed, so the first one was kept" in w for w in job["warnings"])
    assert len(stored_outputs()) == 1


def test_the_retry_can_be_switched_off(user_client: TestClient, marked: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QC_RETRY", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        calls = _scripted_quality(monkeypatch, [_report("check", residual=0.9)])
        assert render(user_client, marked, quality="balanced").json()["job"]["status"] == "completed"
        assert len(calls) == 1
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
