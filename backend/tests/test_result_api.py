# ruff: noqa: F811  (pytest fixtures imported from test_render_api are re-declared as test arguments)
"""Previewing and downloading the finished video: ownership, signed links, names, and what happens when there is nothing."""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.api.result import DOWNLOAD_TTL_SECONDS, download_name
from app.models import Job, Project
from app.storage.s3 import get_storage
from tests.render_media import stream_of
from tests.test_render_api import marked, render, source_video, stored_outputs  # noqa: F401  (fixtures)


@pytest.fixture
def finished(user_client: TestClient, marked: dict[str, Any]) -> dict[str, Any]:  # noqa: F811
    assert render(user_client, marked).status_code == 202
    assert user_client.get(f"/api/projects/{marked['id']}").json()["status"] == "COMPLETED"
    return marked


# --- file names ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "container", "expected"),
    [
        ("holiday", "mp4", "holiday-restored.mp4"),
        ("My Trip 2026", "webm", "My Trip 2026-restored.webm"),
        ('evil".mp4\r\nX-Header: 1', "mp4", "evil.mp4X-Header 1-restored.mp4"),
        ("../../etc/passwd", "mp4", "etcpasswd-restored.mp4"),
        ("日本語", "mp4", "video-restored.mp4"),
        ("   ", "mp4", "video-restored.mp4"),
        ("a" * 300, "mp4", "a" * 80 + "-restored.mp4"),
        ("clip", "exe", "clip-restored.mp4"),
        ("café", "mp4", "cafe-restored.mp4"),
    ],
)
def test_download_names_are_safe_for_a_header(name: str, container: str, expected: str) -> None:
    result = download_name(name, container)
    assert result == expected
    assert not any(c in result for c in '"\\/\r\n;')


# --- preview ---------------------------------------------------------------------------------------


def test_the_preview_links_both_videos_and_describes_the_result(user_client: TestClient, finished: dict[str, Any]) -> None:
    body = user_client.get(f"/api/projects/{finished['id']}/preview").json()
    assert body["container"] == "mp4" and body["frame_count"] == 30 and body["width"] == 320 and body["height"] == 568
    assert body["development_only"] is True and any("development engine" in w for w in body["warnings"])
    assert body["size_bytes"] and body["audio"] == "copied"

    result = requests.get(body["result_url"], timeout=30)
    original = requests.get(body["original_url"], timeout=30)
    assert result.status_code == original.status_code == 200
    assert result.content[4:8] == b"ftyp" and original.content[4:8] == b"ftyp"
    assert len(result.content) == body["size_bytes"]
    assert "attachment" not in result.headers.get("content-disposition", "")  # playable in the page


def test_the_preview_needs_a_finished_video(user_client: TestClient, marked: dict[str, Any]) -> None:
    response = user_client.get(f"/api/projects/{marked['id']}/preview")
    assert response.status_code == 409 and "no finished video" in response.json()["detail"]


def test_a_reopened_video_still_previews_its_earlier_result(user_client: TestClient, finished: dict[str, Any]) -> None:
    assert user_client.post(f"/api/projects/{finished['id']}/reopen").status_code == 200
    assert user_client.get(f"/api/projects/{finished['id']}/preview").status_code == 200


def test_other_people_cannot_preview_or_export(user_client: TestClient, other_client: TestClient, finished: dict[str, Any]) -> None:
    for call in (
        lambda c: c.get(f"/api/projects/{finished['id']}/preview"),
        lambda c: c.post(f"/api/projects/{finished['id']}/export"),
    ):
        assert call(other_client).status_code == 404
    anonymous = TestClient(user_client.app)
    assert anonymous.get(f"/api/projects/{finished['id']}/preview").status_code == 401
    assert anonymous.post(f"/api/projects/{finished['id']}/export").status_code == 401


def test_a_result_whose_file_has_gone_is_reported_not_linked(user_client: TestClient, finished: dict[str, Any]) -> None:
    storage = get_storage()
    for key in stored_outputs():
        storage.delete(key)
    for path, method in (("preview", "get"), ("export", "post")):
        response = getattr(user_client, method)(f"/api/projects/{finished['id']}/{path}")
        assert response.status_code == 410 and "no longer available" in response.json()["detail"]


# --- export ----------------------------------------------------------------------------------------


def test_the_download_link_saves_the_result_under_a_clean_name(user_client: TestClient, finished: dict[str, Any], db: Session) -> None:
    db.execute(update(Project).where(Project.id == uuid.UUID(finished["id"])).values(name='Beach "day"/1'))
    db.commit()
    body = user_client.post(f"/api/projects/{finished['id']}/export").json()
    assert body["filename"] == "Beach day1-restored.mp4" and body["expires_in"] == DOWNLOAD_TTL_SECONDS

    query = parse_qs(urlparse(body["url"]).query)
    assert int(query["X-Amz-Expires"][0]) == DOWNLOAD_TTL_SECONDS or "Expires" in query

    response = requests.get(body["url"], timeout=30)
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="Beach day1-restored.mp4"'
    assert len(response.content) == body["size_bytes"]


def test_the_downloaded_file_is_the_checked_result(user_client: TestClient, finished: dict[str, Any], tmp_path) -> None:  # noqa: ANN001
    body = user_client.post(f"/api/projects/{finished['id']}/export").json()
    path = tmp_path / body["filename"]
    path.write_bytes(requests.get(body["url"], timeout=30).content)
    video = stream_of(path, "video")
    assert (video["width"], video["height"], video["codec_name"]) == (320, 568, "h264")
    assert stream_of(path, "audio")["codec_name"] == "aac"


def test_export_needs_a_finished_video(user_client: TestClient, marked: dict[str, Any]) -> None:
    assert user_client.post(f"/api/projects/{marked['id']}/export").status_code == 409


def test_a_failed_render_does_not_hide_the_earlier_result(user_client: TestClient, finished: dict[str, Any], db: Session) -> None:
    db.execute(update(Job).where(Job.project_id == uuid.UUID(finished["id"])).values(status="failed"))
    db.commit()
    assert user_client.get(f"/api/projects/{finished['id']}/preview").status_code == 409  # no completed job owns the file
