"""Local mode: SQLite, the disk as file storage, and background jobs on threads. Nothing external is started."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.storage.local import LocalStorage
from app.storage.s3 import StorageError, get_storage
from app.workers import local as local_jobs
from tests.conftest import GOOD_PASSWORD, _clear_caches

MIB = 1024 * 1024


@pytest.fixture
def local_client(env: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    data = tmp_path / "data"
    env.setenv("LOCAL_MODE", "true")
    env.setenv("LOCAL_DATA_DIR", str(data))
    env.setenv("DEV_MODE", "true")
    env.setenv("AI_PROVIDER", "dev")
    env.setenv("DATABASE_URL", f"sqlite+pysqlite:///{data}/cleanframe.db")  # the app creates this file and its tables
    env.setenv("CELERY_EAGER", "false")
    _clear_caches()
    with TestClient(create_app()) as client:
        yield client
    local_jobs.shutdown()
    _clear_caches()


def path_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.path}?{parts.query}"


def wait_for(check, seconds: float = 120.0):
    end = time.time() + seconds
    while time.time() < end:
        value = check()
        if value:
            return value
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_local_mode_fills_in_everything_that_was_left_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "JWT_SECRET", "STORAGE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "d"))
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("sqlite") and settings.storage_backend == "filesystem"
    assert settings.task_backend == "thread" and settings.rate_limit_storage == "memory" and not settings.cookie_secure
    assert len(settings.jwt_secret) >= 32
    assert settings.ai_provider == "auto"  # the real removal engines; "dev" only blurs and has to be asked for
    assert Settings(_env_file=None).jwt_secret == settings.jwt_secret  # kept, so sessions survive a restart
    assert oct((tmp_path / "d" / "secret.key").stat().st_mode & 0o777) == "0o600"
    settings.validate_runtime()


def test_an_explicit_setting_still_wins_in_local_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("TASK_BACKEND", "celery")
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///elsewhere.db")
    settings = Settings(_env_file=None)
    assert settings.task_backend == "celery" and settings.database_url.endswith("elsewhere.db")


def test_the_whole_journey_works_on_the_disk_with_threads(local_client: TestClient, videos: dict[str, Path]) -> None:
    assert isinstance(get_storage(), LocalStorage)
    assert local_client.post("/api/auth/register", json={"email": "ada@example.com", "name": "Ada", "password": GOOD_PASSWORD}).status_code == 201

    data = videos["small"].read_bytes()
    created = local_client.post("/api/projects", json={"filename": "small.mp4", "size_bytes": len(data)}).json()
    project_id = created["project"]["id"]
    urls = local_client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": [1]}).json()["urls"]
    put = local_client.put(path_of(urls[0]["url"]), content=data)
    assert put.status_code == 200 and put.headers["ETag"].startswith('"')
    assert local_client.get(f"/api/projects/{project_id}/upload").json()["uploaded_parts"][0]["size"] == len(data)
    done = local_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": [{"part_number": 1, "etag": put.headers["ETag"]}]})
    assert done.status_code == 202

    # Analysis runs on a thread; the request came back at once.
    wait_for(lambda: local_client.get(f"/api/projects/{project_id}").json()["status"] == "MASKING")
    mask = {"version": 1, "regions": [{"id": "r", "name": "Mark", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.2}}]}]}
    assert local_client.put(f"/api/projects/{project_id}/mask", json=mask).status_code == 200

    started = local_client.post(f"/api/projects/{project_id}/render", json={})
    assert started.status_code == 202
    job = wait_for(lambda: (j := local_client.get(f"/api/projects/{project_id}/render").json()["job"])["status"] in ("completed", "failed") and j)
    assert job["status"] == "completed", job.get("error")

    link = local_client.post(f"/api/projects/{project_id}/export").json()
    video = local_client.get(path_of(link["url"]))
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert "attachment" in video.headers["content-disposition"] and len(video.content) == link["size_bytes"]
    ranged = local_client.get(path_of(link["url"]), headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and len(ranged.content) == 100  # what makes seeking in the player work

    assert local_client.delete(f"/api/projects/{project_id}").status_code == 204
    files = [p for p in (Path(get_settings().local_data_dir) / "storage" / "files").rglob("*") if p.is_file()]
    assert files == [], files


def test_links_cannot_be_forged_reused_for_another_operation_or_used_after_they_expire(local_client: TestClient, tmp_path: Path) -> None:
    storage = get_storage()
    assert isinstance(storage, LocalStorage)
    good = storage.presign_get("derived/u/p/proxy.mp4")
    token = good.split("token=", 1)[1]
    assert local_client.get(f"/api/local-files/get?token={token}x").status_code == 403  # tampered
    assert local_client.get("/api/local-files/get?token=garbage").status_code == 403
    assert local_client.put(f"/api/local-files/upload?token={token}", content=b"x").status_code == 403  # a read link can't write
    expired = storage.presign_get("derived/u/p/proxy.mp4", expires_in=-5)
    assert local_client.get(path_of(expired)).status_code == 403
    assert local_client.get(path_of(good)).status_code == 404  # valid link, nothing stored there


def test_a_key_cannot_leave_the_storage_folder(local_client: TestClient) -> None:
    storage = get_storage()
    for key in ("../secret.key", "/etc/passwd", "a/../../b", ""):
        with pytest.raises(StorageError):
            storage.path_for(key)  # type: ignore[attr-defined]


def test_an_oversized_part_is_refused_and_leaves_nothing_behind(local_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = get_storage()
    assert isinstance(storage, LocalStorage)
    upload_id = storage.create_multipart_upload("uploads/u/p/original.mp4", "video/mp4")
    storage.max_part_bytes = 1000
    url = storage.presign_part("uploads/u/p/original.mp4", upload_id, 1)
    assert local_client.put(path_of(url), content=b"x" * 5000).status_code == 413
    assert storage.list_parts("uploads/u/p/original.mp4", upload_id) == []
    assert not list(storage.part_path(upload_id, 1).parent.glob("*.writing"))


def test_completing_with_a_wrong_etag_is_rejected(local_client: TestClient) -> None:
    storage = get_storage()
    assert isinstance(storage, LocalStorage)
    key = "uploads/u/p/original.mp4"
    upload_id = storage.create_multipart_upload(key, "video/mp4")
    assert local_client.put(path_of(storage.presign_part(key, upload_id, 1)), content=b"hello").status_code == 200
    with pytest.raises(StorageError):
        storage.complete_multipart_upload(key, upload_id, [(1, '"0000"')])
    assert storage.object_size(key) is None


def test_jobs_take_turns_when_there_is_one_worker(local_client: TestClient) -> None:
    order: list[str] = []
    gate = [False]

    def first() -> None:
        wait_for(lambda: gate[0], 10)
        order.append("first")

    local_jobs.submit("first", first)
    local_jobs.submit("second", lambda: order.append("second"))
    time.sleep(0.3)
    assert order == []  # the second job waits for the first
    gate[0] = True
    wait_for(lambda: order == ["first", "second"], 10)


def test_a_job_that_crashes_does_not_stop_the_next_one(local_client: TestClient) -> None:
    done: list[str] = []

    def boom() -> None:
        raise RuntimeError("boom")

    local_jobs.submit("boom", boom)
    local_jobs.submit("fine", lambda: done.append("ok"))
    wait_for(lambda: done == ["ok"], 10)


def test_work_that_was_running_when_the_app_closed_is_marked_failed_on_the_next_start(local_client: TestClient, videos: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy import update

    from app.db import get_sessionmaker
    from app.models import Project

    local_client.post("/api/auth/register", json={"email": "ada@example.com", "name": "Ada", "password": GOOD_PASSWORD})
    data = videos["small"].read_bytes()
    created = local_client.post("/api/projects", json={"filename": "small.mp4", "size_bytes": len(data)}).json()
    project_id = created["project"]["id"]
    urls = local_client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": [1]}).json()["urls"]
    etag = local_client.put(path_of(urls[0]["url"]), content=data).headers["ETag"]
    local_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": [{"part_number": 1, "etag": etag}]})
    wait_for(lambda: local_client.get(f"/api/projects/{project_id}").json()["status"] == "MASKING")
    mask = {"version": 1, "regions": [{"id": "r", "name": "Mark", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.2}}]}]}
    local_client.put(f"/api/projects/{project_id}/mask", json=mask)

    monkeypatch.setattr(local_jobs, "submit", lambda name, function: None)  # the queue "is lost": nothing will ever run
    assert local_client.post(f"/api/projects/{project_id}/render", json={}).status_code == 202
    assert local_client.get(f"/api/projects/{project_id}/render").json()["job"]["status"] == "queued"

    assert local_jobs.recover_interrupted() == 1
    job = local_client.get(f"/api/projects/{project_id}/render").json()["job"]
    assert job["status"] == "failed" and "closed" in job["error"]
    assert local_client.get(f"/api/projects/{project_id}").json()["status"] == "MASKING"  # editable again

    with get_sessionmaker()() as db:  # an upload that was being checked
        db.execute(update(Project).values(status="ANALYZING"))
        db.commit()
    assert local_jobs.recover_interrupted() == 1
    failed = local_client.get(f"/api/projects/{project_id}").json()
    assert failed["status"] == "FAILED" and "upload it again" in failed["error"]
    assert local_jobs.recover_interrupted() == 0  # nothing left to recover


# --- phone access (`run_local.py --lan`) ------------------------------------------------------------------------------------


def test_local_mode_refuses_a_network_address_unless_phone_access_was_asked_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("CORS_ORIGINS", "http://192.168.1.20:3000")
    monkeypatch.setenv("API_PUBLIC_URL", "http://192.168.1.20:3000")
    with pytest.raises(Exception, match="API_PUBLIC_URL must be localhost"):
        Settings(_env_file=None).validate_runtime()
    monkeypatch.setenv("LOCAL_LAN", "true")
    Settings(_env_file=None).validate_runtime()


def test_phone_access_needs_local_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCAL_LAN", "true")
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///x.db")
    monkeypatch.setenv("JWT_SECRET", "x" * 64)
    monkeypatch.setenv("STORAGE_BUCKET", "b")
    with pytest.raises(Exception, match="LOCAL_LAN only applies to local mode"):
        Settings(_env_file=None).validate_runtime()


def test_with_an_empty_public_address_file_links_are_relative_so_they_follow_the_page(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path, secret="s" * 40, public_url="", max_part_bytes=MIB)
    upload_id = storage.create_multipart_upload("k/original.mp4", "video/mp4")
    assert storage.presign_part("k/original.mp4", upload_id, 1).startswith("/api/local-files/upload?token=")
    assert storage.presign_get("k/original.mp4").startswith("/api/local-files/get?token=")
