import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
import requests
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_engine, get_sessionmaker
from app.main import create_app
from app.models import Base
from app.ratelimit import get_limiter
from app.storage.s3 import get_storage

TEST_SECRET = "test-only-secret-" + "x" * 40
ALLOWED_ORIGIN = "http://localhost:3000"
BUCKET = "test-bucket"
MIB = 1024 * 1024
GOOD_PASSWORD = "correct horse battery"


def _clear_caches() -> None:
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    get_storage.cache_clear()
    get_limiter.cache_clear()


def _empty_shared_database() -> None:
    """With a shared PostgreSQL database every test must start (and leave) it empty. SQLite files are per-test."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        return
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    # SQLite by default. With TEST_DATABASE_URL set to a PostgreSQL address the same tests run on PostgreSQL, the
    # database production uses (see docs/ARCHITECTURE.md, Testing).
    monkeypatch.setenv("DATABASE_URL", os.environ.get("TEST_DATABASE_URL") or f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
    monkeypatch.setenv("DEV_MODE", "true")
    monkeypatch.setenv("CORS_ORIGINS", ALLOWED_ORIGIN)
    monkeypatch.setenv("CELERY_EAGER", "true")  # run tasks inline; no broker needed
    monkeypatch.setenv("AUTO_PROCESS", "false")  # the existing tests drive the manual flow; automation has its own tests
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")  # tests that exercise limiting switch it on themselves
    monkeypatch.setenv("STORAGE_BUCKET", BUCKET)
    monkeypatch.setenv("UPLOAD_PART_SIZE", str(5 * MIB))
    monkeypatch.delenv("STORAGE_ENDPOINT", raising=False)
    monkeypatch.delenv("STORAGE_PUBLIC_ENDPOINT", raising=False)
    _clear_caches()
    _empty_shared_database()
    yield monkeypatch
    get_engine.cache_clear()  # releases the app's pooled connections before the schema is dropped
    _clear_caches()
    _empty_shared_database()


@pytest.fixture
def client(env: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        app = create_app()
        Base.metadata.create_all(get_engine())
        with TestClient(app) as test_client:
            yield test_client
        get_engine().dispose()


@pytest.fixture
def db(client: TestClient) -> Iterator[Session]:
    with get_sessionmaker()() as session:
        yield session


def register(client: TestClient, email: str = "ada@example.com", name: str = "Ada Lovelace", password: str = GOOD_PASSWORD):
    return client.post("/api/auth/register", json={"email": email, "name": name, "password": password})


@pytest.fixture
def user_client(client: TestClient) -> TestClient:
    """A client signed in as Ada."""
    assert register(client).status_code == 201
    return client


@pytest.fixture
def other_client(client: TestClient) -> TestClient:
    """A second, independent browser session signed in as a different person."""
    other = TestClient(client.app)
    assert register(other, email="grace@example.com", name="Grace Hopper").status_code == 201
    return other


# --- real video files, generated once per test run ---------------------------------------------


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, timeout=120)


@pytest.fixture(scope="session")
def videos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are required for video tests")
    root = tmp_path_factory.mktemp("videos")
    files = {
        "small": root / "small.mp4",
        "webm": root / "clip.webm",
        "big": root / "big.mp4",  # lossless, over 5 MiB so it needs more than one upload part
        "audio_only": root / "audio_only.mp4",
        "garbage": root / "garbage.mp4",
        "truncated": root / "truncated.mp4",
        "vfr": root / "vfr.mp4",  # variable frame rate: 90 stored frames over 3.33 s, with gaps
    }
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x568:rate=30:duration=2", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(files["small"]))
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x568:rate=25:duration=1", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=1", "-c:v", "libvpx-vp9", "-c:a", "libopus", "-shortest", str(files["webm"]))
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6", "-c:v", "libx264", "-preset",
            "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p", str(files["big"]))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", "-f", "mp4",
            str(files["audio_only"]))
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3", "-vf",
            "setpts='PTS+if(gt(N,40),0.2/TB,0)+if(gt(N,70),0.15/TB,0)'", "-fps_mode", "vfr", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(files["vfr"]))
    files["garbage"].write_bytes(bytes(range(256)) * 64)
    files["truncated"].write_bytes(files["small"].read_bytes()[:1500])
    assert files["big"].stat().st_size > 5 * MIB, "test video must need multiple parts"
    return files


# --- simulate the browser: PUT each part straight to the presigned URL --------------------------


def put_part(url: str, data: bytes) -> str:
    response = requests.put(url, data=data, timeout=30)
    assert response.status_code == 200, response.text
    return response.headers["ETag"]


def upload_parts(client: TestClient, project_id: str, data: bytes, part_numbers: list[int], part_size: int) -> list[dict[str, Any]]:
    urls = client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": part_numbers})
    assert urls.status_code == 200, urls.text
    return [
        {"part_number": u["part_number"], "etag": put_part(u["url"], data[(u["part_number"] - 1) * part_size : u["part_number"] * part_size])}
        for u in urls.json()["urls"]
    ]


def upload_video(client: TestClient, path: Path, filename: str | None = None) -> dict[str, Any]:
    """Create, upload every part and complete. Returns the project as the API reports it afterwards."""
    data = path.read_bytes()
    created = client.post("/api/projects", json={"filename": filename or path.name, "size_bytes": len(data)})
    assert created.status_code == 201, created.text
    body = created.json()
    project_id, upload = body["project"]["id"], body["upload"]
    parts = upload_parts(client, project_id, data, list(range(1, upload["total_parts"] + 1)), upload["part_size"])
    done = client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": parts})
    assert done.status_code == 202, done.text
    fetched = client.get(f"/api/projects/{project_id}")
    assert fetched.status_code == 200
    return fetched.json()
