# ruff: noqa: F811  (pytest fixtures imported from test_render_api are re-declared as test arguments)
"""Additional exports of a finished video (another size or format), checked with ffprobe on the downloaded files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

from tests.conftest import upload_video
from tests.render_media import stream_of
from tests.synthetic_video import make_video
from tests.test_render_api import render
from tests.test_render_pipeline import with_audio
from tests.test_tracking_api import mask_doc

FRAMES = 20


@pytest.fixture(scope="module")
def qhd_source(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    root = tmp_path_factory.mktemp("exports")
    video = make_video(root / "silent.mp4", width=2560, height=1440, frames=FRAMES, pose=lambda f: (900, 300, 1.0, 0.0), sprite_size=(240, 120))
    return video, with_audio(video.path, root / "source.mp4", FRAMES / 30)


@pytest.fixture
def finished(user_client: TestClient, qhd_source: tuple[Any, Path]) -> dict[str, Any]:
    video, path = qhd_source
    project = upload_video(user_client, path)
    assert project["status"] == "MASKING", project
    assert user_client.put(f"/api/projects/{project['id']}/mask", json=mask_doc(video)).status_code == 200
    assert render(user_client, project).status_code == 202
    assert user_client.get(f"/api/projects/{project['id']}").json()["status"] == "COMPLETED"
    return project


def download(client: TestClient, project: dict[str, Any], job_id: str, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    link = client.post(f"/api/projects/{project['id']}/exports/{job_id}/link")
    assert link.status_code == 200, link.text
    body = link.json()
    data = requests.get(body["url"], timeout=60)
    assert data.status_code == 200 and len(data.content) == body["size_bytes"]
    target = tmp_path / body["filename"]
    target.write_bytes(data.content)
    return target, body


def test_only_sizes_smaller_than_the_finished_video_are_offered(user_client: TestClient, finished: dict[str, Any]) -> None:
    options = user_client.get(f"/api/projects/{finished['id']}/exports/options").json()
    assert (options["output_width"], options["output_height"]) == (2560, 1440) and options["output_label"] == "1440p QHD"
    assert [s["height"] for s in options["sizes"]] == [1080, 720]  # not 1440 (same), not 2160 (an enlargement)
    assert [(s["width_px"], s["height_px"]) for s in options["sizes"]] == [(1920, 1080), (1280, 720)]
    assert {f["codec"] for f in options["formats"]} >= {"h264"}


def test_a_smaller_export_is_made_from_the_finished_file_with_audio_and_timing_intact(
    user_client: TestClient, finished: dict[str, Any], tmp_path: Path
) -> None:
    made = user_client.post(f"/api/projects/{finished['id']}/exports", json={"height": 1080})
    assert made.status_code == 202, made.text
    job = made.json()
    assert job["status"] == "completed" and job["percent"] == 100.0  # eager mode: the worker already ran
    assert (job["result"]["width"], job["result"]["height"]) == (1920, 1080)

    path, body = download(user_client, finished, job["id"], tmp_path)
    assert body["filename"].endswith("-1080p.mp4")
    video, audio = stream_of(path, "video"), stream_of(path, "audio")
    assert (video["width"], video["height"], video["nb_frames"]) == (1920, 1080, str(FRAMES))
    assert video["r_frame_rate"] == "30/1" and audio["codec_name"] == "aac"

    original = stream_of(finished_path(user_client, finished, tmp_path), "video")
    assert (original["width"], original["height"]) == (2560, 1440) and original["nb_frames"] == video["nb_frames"]


def finished_path(client: TestClient, project: dict[str, Any], tmp_path: Path) -> Path:
    link = client.post(f"/api/projects/{project['id']}/export").json()
    target = tmp_path / "finished.mp4"
    target.write_bytes(requests.get(link["url"], timeout=60).content)
    return target


@pytest.mark.parametrize("height", [1440, 2160])
def test_a_size_that_would_enlarge_the_picture_is_refused(user_client: TestClient, finished: dict[str, Any], height: int) -> None:
    response = user_client.post(f"/api/projects/{finished['id']}/exports", json={"height": height})
    assert response.status_code == 422 and "can only be exported at a smaller size" in response.json()["detail"]


def test_the_same_format_at_the_same_size_is_not_a_new_export(user_client: TestClient, finished: dict[str, Any]) -> None:
    assert user_client.post(f"/api/projects/{finished['id']}/exports", json={}).status_code == 409


def test_a_format_only_export_keeps_the_size_and_changes_the_codec(user_client: TestClient, finished: dict[str, Any], tmp_path: Path) -> None:
    job = user_client.post(f"/api/projects/{finished['id']}/exports", json={"codec": "hevc"}).json()
    assert job["status"] == "completed", job
    path, _ = download(user_client, finished, job["id"], tmp_path)
    video = stream_of(path, "video")
    assert (video["codec_name"], video["width"], video["height"]) == ("hevc", 2560, 1440)


def test_webm_with_vp9_and_converted_audio(user_client: TestClient, finished: dict[str, Any], tmp_path: Path) -> None:
    job = user_client.post(f"/api/projects/{finished['id']}/exports", json={"height": 720, "container": "webm"}).json()
    assert job["status"] == "completed", job
    path, _ = download(user_client, finished, job["id"], tmp_path)
    video, audio = stream_of(path, "video"), stream_of(path, "audio")
    assert (video["codec_name"], video["width"], video["height"]) == ("vp9", 1280, 720) and audio["codec_name"] == "opus"
    assert job["result"]["audio"] == "converted"  # AAC can't live in WebM: said so, not hidden


def test_asking_twice_for_the_same_export_returns_the_same_one(user_client: TestClient, finished: dict[str, Any]) -> None:
    first = user_client.post(f"/api/projects/{finished['id']}/exports", json={"height": 720}).json()
    second = user_client.post(f"/api/projects/{finished['id']}/exports", json={"height": 720}).json()
    assert first["id"] == second["id"]
    listing = user_client.get(f"/api/projects/{finished['id']}/exports").json()["exports"]
    assert [e["id"] for e in listing] == [first["id"]]


def test_exports_belong_to_their_owner(user_client: TestClient, other_client: TestClient, finished: dict[str, Any]) -> None:
    job = user_client.post(f"/api/projects/{finished['id']}/exports", json={"height": 720}).json()
    assert other_client.get(f"/api/projects/{finished['id']}/exports").status_code == 404
    assert other_client.post(f"/api/projects/{finished['id']}/exports", json={"height": 720}).status_code == 404
    assert other_client.post(f"/api/projects/{finished['id']}/exports/{job['id']}/link").status_code == 404


def test_exports_are_dropped_when_a_new_render_replaces_the_finished_video(user_client: TestClient, finished: dict[str, Any]) -> None:
    from tests.test_render_api import stored_outputs

    user_client.post(f"/api/projects/{finished['id']}/exports", json={"height": 720})
    before = len(stored_outputs())
    assert user_client.post(f"/api/projects/{finished['id']}/reopen").status_code == 200
    assert render(user_client, finished).status_code == 202
    assert user_client.get(f"/api/projects/{finished['id']}/exports").json()["exports"] == []
    assert len(stored_outputs()) == before - 1  # the export's file went; the render's own replaced the old render's


def test_options_need_a_finished_video(user_client: TestClient, qhd_source: tuple[Any, Path]) -> None:
    video, path = qhd_source
    project = upload_video(user_client, path)
    assert user_client.get(f"/api/projects/{project['id']}/exports/options").status_code == 409
