# ruff: noqa: F811  (pytest fixtures imported from test_render_api are re-declared as test arguments)
"""Retention: expired projects lose their files and their access, busy ones are left alone, failures are retried."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db import get_sessionmaker
from app.models import Job, Mask, Project
from app.models.job import JOB_RUNNING
from app.services.cleanup import expire_due_projects, remove_stale_temp_dirs
from app.storage.s3 import StorageError, get_storage
from tests.conftest import register
from tests.test_render_api import marked, render, source_video  # noqa: F401  (fixtures)


def all_keys() -> list[str]:
    storage = get_storage()
    listing = storage._client.list_objects_v2(Bucket=storage._bucket)  # noqa: SLF001
    return [item["Key"] for item in listing.get("Contents", [])]


def make_due(db: Session, project_id: str, *, hours_ago: float = 1) -> None:
    db.execute(update(Project).where(Project.id == _uuid(project_id)).values(expires_at=datetime.now(UTC) - timedelta(hours=hours_ago)))
    db.commit()


def _uuid(value: str):  # noqa: ANN202
    import uuid

    return uuid.UUID(value)


@pytest.fixture
def finished(user_client: TestClient, marked: dict[str, Any]) -> dict[str, Any]:  # noqa: F811
    assert render(user_client, marked).status_code == 202
    assert user_client.get(f"/api/projects/{marked['id']}").json()["status"] == "COMPLETED"
    return marked


def sweep() -> Any:
    return expire_due_projects(get_sessionmaker(), get_storage())


# --- the sweep -------------------------------------------------------------------------------------


def test_the_sweep_deletes_every_file_and_marks_the_project_expired(
    user_client: TestClient, db: Session, finished: dict[str, Any]
) -> None:
    before = all_keys()
    assert any(k.startswith("uploads/") for k in before) and any(k.startswith("outputs/") for k in before)
    assert any(k.startswith("derived/") for k in before)
    make_due(db, finished["id"])

    report = sweep()

    assert (report.expired, report.busy, report.failed) == (1, 0, 0)
    assert all_keys() == []
    body = user_client.get(f"/api/projects/{finished['id']}").json()
    assert body["status"] == "EXPIRED"
    db.expire_all()
    assert db.scalars(select(Job)).all() == [] and db.scalars(select(Mask)).all() == []


def test_projects_that_are_not_due_are_left_alone(user_client: TestClient, finished: dict[str, Any]) -> None:
    keys = all_keys()
    report = sweep()
    assert report.expired == 0 and all_keys() == keys
    assert user_client.get(f"/api/projects/{finished['id']}").json()["status"] == "COMPLETED"


def test_an_expired_project_stops_working_but_can_still_be_deleted(
    user_client: TestClient, db: Session, finished: dict[str, Any]
) -> None:
    make_due(db, finished["id"])
    sweep()
    pid = finished["id"]
    for path in ("media", "mask", "render", "preview"):
        response = user_client.get(f"/api/projects/{pid}/{path}")
        assert response.status_code == 410, (path, response.text)
        assert "expired" in response.json()["detail"]
    assert user_client.post(f"/api/projects/{pid}/export").status_code == 410
    assert user_client.post(f"/api/projects/{pid}/render", json={}).status_code == 410
    assert user_client.delete(f"/api/projects/{pid}").status_code == 204
    assert user_client.get(f"/api/projects/{pid}").status_code == 404


def test_expiry_does_not_show_other_peoples_projects_as_existing(
    user_client: TestClient, other_client: TestClient, db: Session, finished: dict[str, Any]
) -> None:
    make_due(db, finished["id"])
    sweep()
    assert other_client.get(f"/api/projects/{finished['id']}/media").status_code == 404
    assert other_client.get(f"/api/projects/{finished['id']}").status_code == 404


# --- no waiting for the sweep ----------------------------------------------------------------------


def test_access_stops_at_the_deadline_even_if_the_sweep_has_not_run(
    user_client: TestClient, db: Session, finished: dict[str, Any]
) -> None:
    make_due(db, finished["id"])
    response = user_client.get(f"/api/projects/{finished['id']}/preview")
    assert response.status_code == 410
    assert all_keys() == [], "the files go when somebody asks, not only when the sweep runs"
    assert user_client.get(f"/api/projects/{finished['id']}").json()["status"] == "EXPIRED"


def test_the_dashboard_list_shows_expired_projects_as_expired(
    user_client: TestClient, db: Session, finished: dict[str, Any]
) -> None:
    make_due(db, finished["id"])
    listed = user_client.get("/api/projects").json()
    assert [p["status"] for p in listed] == ["EXPIRED"]
    assert all_keys() == []


# --- what must not be deleted, and what must be retried --------------------------------------------


def test_a_project_a_worker_is_using_is_not_deleted_under_it(
    user_client: TestClient, db: Session, finished: dict[str, Any]
) -> None:
    job = db.scalars(select(Job)).one()
    db.execute(update(Job).where(Job.id == job.id).values(status=JOB_RUNNING, updated_at=datetime.now(UTC)))
    make_due(db, finished["id"])
    keys = all_keys()

    report = sweep()

    assert (report.expired, report.busy) == (0, 1) and all_keys() == keys
    assert user_client.get(f"/api/projects/{finished['id']}").json()["status"] != "EXPIRED"


def test_a_job_that_has_been_silent_longer_than_any_job_can_run_does_not_hold_the_files_forever(
    user_client: TestClient, db: Session, finished: dict[str, Any], env: pytest.MonkeyPatch
) -> None:
    job = db.scalars(select(Job)).one()
    db.execute(update(Job).where(Job.id == job.id).values(status=JOB_RUNNING, updated_at=datetime.now(UTC) - timedelta(days=2)))
    make_due(db, finished["id"])
    assert sweep().expired == 1


def test_a_storage_failure_changes_nothing_and_the_next_sweep_finishes_the_job(
    user_client: TestClient, db: Session, finished: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_due(db, finished["id"])
    storage = get_storage()
    real_delete = storage.delete
    calls = {"n": 0}

    def flaky(key: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise StorageError("storage is down")
        real_delete(key)

    monkeypatch.setattr(storage, "delete", flaky)
    first = sweep()
    assert (first.expired, first.failed) == (0, 1)
    db.expire_all()
    assert db.get(Project, _uuid(finished["id"])).status == "COMPLETED", "must not claim deletion that didn't happen"

    monkeypatch.setattr(storage, "delete", real_delete)
    second = sweep()
    assert (second.expired, second.failed) == (1, 0)
    assert all_keys() == []


def test_an_unfinished_upload_is_aborted_when_it_expires(user_client: TestClient, db: Session) -> None:
    created = user_client.post("/api/projects", json={"filename": "half.mp4", "size_bytes": 12 * 1024 * 1024})
    assert created.status_code == 201
    storage = get_storage()
    assert storage._client.list_multipart_uploads(Bucket=storage._bucket).get("Uploads")  # noqa: SLF001
    make_due(db, created.json()["project"]["id"])

    assert sweep().expired == 1

    assert not storage._client.list_multipart_uploads(Bucket=storage._bucket).get("Uploads")  # noqa: SLF001
    assert user_client.get(f"/api/projects/{created.json()['project']['id']}").json()["status"] == "EXPIRED"


def test_the_sweep_handles_many_projects_and_reports_honestly(client: TestClient, db: Session) -> None:
    ids = []
    for index in range(3):
        register(client, email=f"user{index}@example.com")
        created = client.post("/api/projects", json={"filename": f"c{index}.mp4", "size_bytes": 6 * 1024 * 1024})
        ids.append(created.json()["project"]["id"])
    make_due(db, ids[0])
    make_due(db, ids[1], hours_ago=30)
    report = sweep()
    assert report.expired == 2
    db.expire_all()
    statuses = {str(p.id): p.status for p in db.scalars(select(Project))}
    assert [statuses[i] for i in ids] == ["EXPIRED", "EXPIRED", "UPLOADING"]


# --- the retention clock -----------------------------------------------------------------------------


def test_a_finished_render_gets_the_full_retention_period_again(
    user_client: TestClient, db: Session, marked: dict[str, Any], env: pytest.MonkeyPatch
) -> None:
    make_due_soon = datetime.now(UTC) + timedelta(minutes=5)
    db.execute(update(Project).where(Project.id == _uuid(marked["id"])).values(expires_at=make_due_soon))
    db.commit()
    assert render(user_client, marked).status_code == 202
    db.expire_all()
    expires = db.get(Project, _uuid(marked["id"])).expires_at
    expires = expires if expires.tzinfo else expires.replace(tzinfo=UTC)
    assert expires - datetime.now(UTC) > timedelta(hours=47)


def test_retention_never_gets_shorter(user_client: TestClient, db: Session, marked: dict[str, Any]) -> None:
    far = datetime.now(UTC) + timedelta(days=30)
    db.execute(update(Project).where(Project.id == _uuid(marked["id"])).values(expires_at=far))
    db.commit()
    assert render(user_client, marked).status_code == 202
    db.expire_all()
    expires = db.get(Project, _uuid(marked["id"])).expires_at
    expires = expires if expires.tzinfo else expires.replace(tzinfo=UTC)
    assert expires - far < timedelta(seconds=1)


# --- leftover working folders ------------------------------------------------------------------------


def test_only_old_working_folders_of_this_application_are_removed(tmp_path: Path) -> None:
    old_age = time.time() - 10 * 3600
    old_render = tmp_path / "cf-render-abc"
    old_tracking = tmp_path / "cf-tracking-xyz"
    fresh = tmp_path / "cf-render-fresh"
    other = tmp_path / "somebody-elses-folder"
    plain_file = tmp_path / "cf-render-file"
    for folder in (old_render, old_tracking, fresh, other):
        folder.mkdir()
        (folder / "frame.raw").write_bytes(b"x" * 10)
    plain_file.write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "cf-render-link"
    link.symlink_to(outside)
    for path in (old_render, old_tracking, other, plain_file, link):
        os.utime(path, (old_age, old_age), follow_symlinks=False)

    removed = remove_stale_temp_dirs(tmp_path, older_than_seconds=3600)

    assert removed == 2
    assert not old_render.exists() and not old_tracking.exists()
    assert fresh.exists() and other.exists() and plain_file.exists()
    assert link.is_symlink() and outside.exists(), "a link is never followed"


def test_a_missing_temp_directory_is_not_an_error(tmp_path: Path) -> None:
    assert remove_stale_temp_dirs(tmp_path / "nope", older_than_seconds=1) == 0


def test_the_sweep_is_scheduled() -> None:
    from app.workers.celery_app import celery_app
    from app.workers.tasks import cleanup_expired

    schedule = celery_app.conf.beat_schedule["retention-sweep"]
    assert schedule["task"] == "cleanup_expired" and schedule["schedule"] >= 60
    assert cleanup_expired.name == "cleanup_expired"
