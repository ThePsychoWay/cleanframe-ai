"""Security checks across the whole API (spec section 59): who can reach what, hostile file names and ids, fake videos,
and the shape of the signed links. Routes are discovered from the application, so a new route that forgets to
require a session or an owner fails here."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import boto3
import pytest
from botocore.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import result as result_api
from app.config import get_settings
from app.main import create_app
from app.models import Project
from app.storage.s3 import PRESIGN_TTL_SECONDS, ObjectStorage
from tests.conftest import MIB, put_part

PUBLIC = {("GET", "/api/health"), ("POST", "/api/auth/register"), ("POST", "/api/auth/login"), ("POST", "/api/auth/logout")}
# Valid-enough bodies, so a request fails on who is asking and not on its shape.
BODIES: dict[tuple[str, str], dict[str, Any]] = {
    ("POST", "/api/projects"): {"filename": "a.mp4", "size_bytes": 6 * MIB},
    ("POST", "/api/projects/{project_id}/upload/urls"): {"part_numbers": [1]},
    ("POST", "/api/projects/{project_id}/upload/complete"): {"parts": [{"part_number": 1, "etag": "x", "size": 1}]},
    ("PUT", "/api/projects/{project_id}/mask"): {"version": 1, "regions": []},
    ("POST", "/api/projects/{project_id}/render"): {},
    ("POST", "/api/projects/{project_id}/exports"): {},
}


def api_routes(client: TestClient) -> list[tuple[str, str]]:
    """Every (method, path) the application serves, read from its OpenAPI description. (Walking `app.routes` isn't
    reliable: included routers are wrapped, and a walk that finds nothing would make every check below pass.)"""
    found = []
    for path, operations in client.app.openapi()["paths"].items():
        if path.startswith("/api/"):
            found.extend((method.upper(), path) for method in operations if method in {"get", "post", "put", "patch", "delete"})
    return sorted(found)


def call(client: TestClient, method: str, template: str, project_id: str | None = None):  # noqa: ANN202
    path = template.replace("{project_id}", project_id or str(uuid.uuid4())).replace("{job_id}", str(uuid.uuid4()))
    body = BODIES.get((method, template))
    return client.request(method, path, json=body) if body is not None else client.request(method, path)


def test_the_route_list_is_what_we_expect_so_the_matrix_below_cannot_silently_cover_nothing(client: TestClient) -> None:
    routes = api_routes(client)
    assert len(routes) >= 25, routes
    assert {r for r in routes if "{project_id}" in r[1]}, "project routes found"
    assert PUBLIC <= set(routes)


# --- authentication: every route ------------------------------------------------------------------------


def test_every_route_except_the_public_ones_refuses_a_caller_without_a_session(client: TestClient) -> None:
    open_routes = []
    for method, template in api_routes(client):
        if (method, template) in PUBLIC:
            continue
        status = call(client, method, template).status_code
        if status != 401:
            open_routes.append((method, template, status))
    assert open_routes == [], f"routes that answered without a session: {open_routes}"


def test_the_api_description_is_not_published_outside_development(env: pytest.MonkeyPatch) -> None:
    env.setenv("DEV_MODE", "false")
    env.setenv("AI_PROVIDER", "api")
    env.setenv("RATE_LIMIT_ENABLED", "true")
    get_settings.cache_clear()
    app = create_app()
    assert app.docs_url is None and app.redoc_url is None and app.openapi_url is None
    get_settings.cache_clear()
    env.setenv("DEV_MODE", "true")
    assert create_app().openapi_url == "/openapi.json"


# --- authorization: every project route, someone else's project -----------------------------------------


@pytest.fixture
def victim_project(user_client: TestClient) -> str:
    created = user_client.post("/api/projects", json={"filename": "private.mp4", "size_bytes": 6 * MIB})
    assert created.status_code == 201
    return str(created.json()["project"]["id"])


def test_another_person_gets_not_found_from_every_project_route_and_changes_nothing(
    user_client: TestClient, other_client: TestClient, victim_project: str, db: Session
) -> None:
    before = {c.key: getattr(db.get(Project, uuid.UUID(victim_project)), c.key) for c in Project.__table__.columns}
    leaks = []
    checked = 0
    for method, template in api_routes(other_client):
        if "{project_id}" not in template:
            continue
        checked += 1
        response = call(other_client, method, template, victim_project)
        if response.status_code != 404:
            leaks.append((method, template, response.status_code))
        elif response.json() != {"detail": "We couldn't find that project."}:
            leaks.append((method, template, response.json()))
    assert checked >= 15, f"only {checked} project routes were checked"
    assert leaks == [], f"routes that treated someone else's project differently from a missing one: {leaks}"

    db.expire_all()
    after = {c.key: getattr(db.get(Project, uuid.UUID(victim_project)), c.key) for c in Project.__table__.columns}
    assert before == after, "a refused request must not change the project"
    assert user_client.get(f"/api/projects/{victim_project}").status_code == 200


def test_a_missing_project_and_someone_elses_look_identical(other_client: TestClient, victim_project: str) -> None:
    missing = other_client.get(f"/api/projects/{uuid.uuid4()}")
    theirs = other_client.get(f"/api/projects/{victim_project}")
    assert (missing.status_code, missing.json()) == (theirs.status_code, theirs.json())


def test_the_projects_list_never_includes_other_peoples_projects(other_client: TestClient, victim_project: str) -> None:
    assert other_client.get("/api/projects").json() == []


# --- what a response may contain -------------------------------------------------------------------------


def test_no_response_exposes_a_storage_path_or_an_upload_id(user_client: TestClient, victim_project: str) -> None:
    texts = [
        user_client.get("/api/projects").text,
        user_client.get(f"/api/projects/{victim_project}").text,
        user_client.get(f"/api/projects/{victim_project}/upload").text,
        user_client.get(f"/api/projects/{victim_project}/mask").text,
        user_client.get(f"/api/projects/{victim_project}/render").text,
        user_client.get(f"/api/projects/{victim_project}/tracking").text,
    ]
    for text in texts:
        for fragment in ("uploads/", "derived/", "outputs/", "original_file_key", "output_file_key", "upload_id", "password"):
            assert fragment not in text, fragment


def test_the_client_cannot_set_fields_it_does_not_own(user_client: TestClient, other_client: TestClient, db: Session) -> None:
    other = other_client.get("/api/auth/me").json()["id"]
    created = user_client.post(
        "/api/projects",
        json={
            "filename": "mine.mp4", "size_bytes": 6 * MIB, "status": "COMPLETED", "user_id": other, "id": str(uuid.uuid4()),
            "expires_at": "2099-01-01T00:00:00Z", "output_file_key": "outputs/x", "original_file_key": "uploads/someone/else",
        },
    )
    assert created.status_code == 201
    project = db.scalars(select(Project)).one()
    assert project.status == "UPLOADING" and str(project.user_id) != other
    assert project.original_file_key == f"uploads/{project.user_id}/{project.id}/original.mp4"
    assert project.output_file_key is None and project.expires_at.year < 2090


# --- hostile file names and ids ---------------------------------------------------------------------------


ACCEPTED_NAMES = [
    "../../etc/passwd.mp4",
    "..\\..\\windows\\system32\\evil.mp4",
    "a/b/c/video.mov",
    "video\x00.exe.mp4",
    "‮gpj.mp4",
    "x" * 240 + ".mp4",
    "a\r\nSet-Cookie: pwned=1.mp4",
    "%2e%2e%2fsecret.mp4",
    "'; DROP TABLE projects; --.mp4",
    "<script>alert(1)</script>.webm",
    "CON.mp4",
]


@pytest.mark.parametrize("filename", ACCEPTED_NAMES)
def test_hostile_file_names_are_stored_as_text_only_and_never_reach_a_path(
    user_client: TestClient, db: Session, filename: str
) -> None:
    created = user_client.post("/api/projects", json={"filename": filename, "size_bytes": 6 * MIB})
    assert created.status_code == 201, created.text
    project = db.scalars(select(Project)).one()
    assert re.fullmatch(rf"uploads/{project.user_id}/{project.id}/original\.(mp4|mov|webm)", project.original_file_key)
    shown = created.json()["project"]["original_filename"]
    assert not re.search(r"[\x00-\x1f\x7f/\\]", shown), repr(shown)
    assert created.headers["content-type"].startswith("application/json")
    assert "set-cookie" not in created.headers
    assert db.scalars(select(Project)).all(), "the table is still there"


@pytest.mark.parametrize("filename", ["video.exe", "video.mp4.exe", "video.sh", "video", ".mp4.", "video.mp4 ", "video.m3u8", "video.svg", "video.html"])
def test_only_the_three_video_extensions_are_accepted(user_client: TestClient, filename: str) -> None:
    response = user_client.post("/api/projects", json={"filename": filename, "size_bytes": 6 * MIB})
    if filename == "video.mp4 ":  # trailing space is trimmed by validation, which makes it a plain .mp4
        assert response.status_code == 201
    else:
        assert response.status_code == 422, (filename, response.text)


@pytest.mark.parametrize(
    "path",
    [
        "/api/projects/..%2f..%2fetc%2fpasswd",
        "/api/projects/../../etc/passwd",
        "/api/projects/%2e%2e/%2e%2e/etc/passwd/media",
        "/api/projects/not-a-uuid",
        "/api/projects/00000000-0000-0000-0000-000000000000%00",
        "/api/projects/' OR '1'='1",
        "/api/projects/1;DROP TABLE users",
    ],
)
def test_ids_that_are_not_ids_never_touch_the_filesystem_or_the_database(user_client: TestClient, path: str) -> None:
    response = user_client.get(path)
    assert response.status_code in {404, 422}
    assert "root:" not in response.text and "passwd" not in response.text.replace("etc/passwd", "")


def test_sql_in_text_fields_is_stored_as_text(user_client: TestClient, client: TestClient) -> None:
    response = user_client.post("/api/projects", json={"filename": "ok.mp4", "size_bytes": 6 * MIB, "name": "x'); DROP TABLE users;--"})
    assert response.status_code == 201 and response.json()["project"]["name"] == "x'); DROP TABLE users;--"
    assert user_client.get("/api/auth/me").status_code == 200


# --- fake and hostile videos --------------------------------------------------------------------------------

NOT_VIDEOS = {
    "png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
    "text": b"hello, this is not a video\n" * 20,
    "script": b"#!/bin/sh\nrm -rf /\n" * 10,
    "windows_exe": b"MZ\x90\x00" + b"\x00" * 100,
    "zip": b"PK\x03\x04" + b"\x00" * 100,
    "html": b"<html><script>alert(1)</script></html>" * 5,
    "hls_playlist": b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://169.254.169.254/latest/meta-data/\n#EXT-X-ENDLIST\n",
    "ffconcat": b"ffconcat version 1.0\nfile /etc/passwd\n",
    "empty_header": b"\x00" * 200,
}


def upload_bytes(client: TestClient, data: bytes, filename: str = "clip.mp4") -> dict[str, Any]:
    created = client.post("/api/projects", json={"filename": filename, "size_bytes": len(data)}).json()
    project_id = created["project"]["id"]
    etag = put_part(client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": [1]}).json()["urls"][0]["url"], data)
    done = client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": [{"part_number": 1, "etag": etag, "size": len(data)}]})
    assert done.status_code == 202, done.text
    return client.get(f"/api/projects/{project_id}").json()


@pytest.mark.parametrize("kind", sorted(NOT_VIDEOS))
def test_a_file_that_is_not_a_video_is_refused_at_once_by_its_contents_whatever_it_is_called(
    user_client: TestClient, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    started: list[Any] = []
    monkeypatch.setattr("app.api.projects.enqueue_analysis", lambda project_id: started.append(project_id))
    project = upload_bytes(user_client, NOT_VIDEOS[kind])
    assert project["status"] == "FAILED"
    assert "couldn't read this video" in project["error"] and "corrupted" in project["error"]
    assert started == [], "no worker time is spent on it"


def test_a_real_looking_header_followed_by_rubbish_fails_analysis_with_a_clear_message_not_a_crash(user_client: TestClient) -> None:
    project = upload_bytes(user_client, b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + bytes(range(256)) * 20)
    assert project["status"] == "FAILED" and "may be corrupted" in project["error"]


def test_a_real_video_still_passes_after_all_that(user_client: TestClient, videos: dict[str, Path]) -> None:
    data = videos["small"].read_bytes()
    assert upload_bytes(user_client, data)["status"] == "MASKING"


def test_a_playlist_hiding_in_a_video_file_is_never_followed_by_the_probe(tmp_path: Path) -> None:
    from app.video.probe import VideoRejected, run_ffprobe

    real = tmp_path / "real.mp4"
    (tmp_path / "list.mp4").write_text(f"ffconcat version 1.0\nfile {real.name}\n")
    real.write_bytes(b"x")
    with pytest.raises(VideoRejected):
        run_ffprobe(tmp_path / "list.mp4")


# --- signed links -------------------------------------------------------------------------------------------


def parse(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_signed_links_are_short_lived_and_the_download_link_is_the_shortest(user_client: TestClient, db: Session) -> None:
    storage: ObjectStorage = __import__("app.storage.s3", fromlist=["get_storage"]).get_storage()
    playback = parse(storage.presign_get("derived/u/p/proxy.mp4"))
    download = parse(storage.presign_get("outputs/u/p/j.mp4", expires_in=result_api.DOWNLOAD_TTL_SECONDS, download_as="x-restored.mp4"))
    assert int(playback["X-Amz-Expires"]) == PRESIGN_TTL_SECONDS <= 3600
    assert int(download["X-Amz-Expires"]) == result_api.DOWNLOAD_TTL_SECONDS <= 300 < PRESIGN_TTL_SECONDS
    assert "attachment" in download["response-content-disposition"]
    assert "X-Amz-Signature" in playback and "X-Amz-Signature" in download


def test_the_signature_covers_the_object_and_the_lifetime_so_a_link_cannot_be_stretched_or_pointed_elsewhere() -> None:
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="k", aws_secret_access_key="s", config=Config(signature_version="s3v4")
    )
    base = client.generate_presigned_url("get_object", Params={"Bucket": "bkt", "Key": "a.mp4"}, ExpiresIn=300)
    other_key = client.generate_presigned_url("get_object", Params={"Bucket": "bkt", "Key": "b.mp4"}, ExpiresIn=300)
    longer = client.generate_presigned_url("get_object", Params={"Bucket": "bkt", "Key": "a.mp4"}, ExpiresIn=3000)
    signature = parse(base)["X-Amz-Signature"]
    assert signature != parse(other_key)["X-Amz-Signature"] != parse(longer)["X-Amz-Signature"]
    assert signature != parse(longer)["X-Amz-Signature"]
    # SigV4 signs the canonical request: key (path) and X-Amz-Expires are both inside it. Any compliant server
    # (AWS S3, MinIO) refuses a link whose key or expiry was edited, and refuses it after X-Amz-Date + X-Amz-Expires.


def test_every_api_issued_link_points_only_at_the_callers_own_files(
    user_client: TestClient, other_client: TestClient, videos: dict[str, Path]
) -> None:
    mine = upload_bytes(user_client, videos["small"].read_bytes())
    link = user_client.get(f"/api/projects/{mine['id']}/media").json()["proxy_url"]
    user_id = user_client.get("/api/auth/me").json()["id"]
    assert f"/derived/{user_id}/{mine['id']}/proxy.mp4" in urlparse(link).path
    assert other_client.get(f"/api/projects/{mine['id']}/media").status_code == 404
