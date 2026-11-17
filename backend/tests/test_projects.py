import uuid
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Project, ProjectStatus, User
from app.services.analysis import analyze_project
from app.workers.tasks import EnqueueError
from tests.conftest import BUCKET, MIB, put_part, upload_parts, upload_video


# --- the happy path ----------------------------------------------------------------------------


def test_upload_then_analysis_records_the_files_real_properties(user_client: TestClient, videos: dict[str, Path]) -> None:
    project = upload_video(user_client, videos["small"])

    assert project["status"] == "MASKING"
    assert project["error"] is None
    assert (project["width"], project["height"]) == (320, 568)
    assert project["fps"] == pytest.approx(30)
    assert project["duration"] == pytest.approx(2, abs=0.1)
    assert project["frame_count"] == 60
    assert project["codec"] == "h264"
    assert project["audio_codec"] == "aac"
    assert project["container"] in {"mov", "mp4"}
    assert project["name"] == "small"


def test_webm_is_accepted(user_client: TestClient, videos: dict[str, Path]) -> None:
    project = upload_video(user_client, videos["webm"])
    assert project["status"] == "MASKING"
    assert project["codec"] == "vp9"
    assert project["audio_codec"] == "opus"
    assert project["container"] in {"matroska", "webm"}


def test_multipart_upload_can_be_resumed_after_a_dropped_connection(user_client: TestClient, videos: dict[str, Path]) -> None:
    data = videos["big"].read_bytes()
    created = user_client.post("/api/projects", json={"filename": "big.mp4", "size_bytes": len(data)})
    body = created.json()
    project_id, part_size = body["project"]["id"], body["upload"]["part_size"]
    assert body["upload"]["total_parts"] == 2

    # Part 1 arrives, then the connection drops.
    first = upload_parts(user_client, project_id, data, [1], part_size)

    # Choosing the same file again resumes the same project and reports what storage already holds.
    resumed = user_client.post("/api/projects", json={"filename": "big.mp4", "size_bytes": len(data)})
    assert resumed.status_code == 200
    assert resumed.json()["project"]["id"] == project_id
    assert [p["part_number"] for p in resumed.json()["upload"]["uploaded_parts"]] == [1]
    assert user_client.get(f"/api/projects/{project_id}/upload").json()["uploaded_parts"][0]["etag"] == first[0]["etag"]

    second = upload_parts(user_client, project_id, data, [2], part_size)
    done = user_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": first + second})
    assert done.status_code == 202
    final = user_client.get(f"/api/projects/{project_id}").json()
    assert final["status"] == "MASKING"
    assert (final["width"], final["height"]) == (640, 360)


def test_completing_twice_is_harmless(user_client: TestClient, videos: dict[str, Path]) -> None:
    data = videos["small"].read_bytes()
    project_id = user_client.post("/api/projects", json={"filename": "small.mp4", "size_bytes": len(data)}).json()["project"]["id"]
    parts = upload_parts(user_client, project_id, data, [1], 5 * MIB)
    first = user_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": parts})
    again = user_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": parts})
    assert first.status_code == again.status_code == 202
    assert again.json()["status"] == "MASKING"


# --- request validation ------------------------------------------------------------------------


def test_unsupported_extension_is_rejected_up_front(user_client: TestClient) -> None:
    response = user_client.post("/api/projects", json={"filename": "clip.mkv", "size_bytes": 1000})
    assert response.status_code == 422
    assert response.json()["detail"] == "Unsupported video format. Please upload an MP4, MOV, WebM or AVI file."


@pytest.mark.parametrize("size", [0, -5])
def test_empty_or_negative_size_is_rejected(user_client: TestClient, size: int) -> None:
    assert user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": size}).status_code == 422


def test_oversized_upload_is_rejected_before_any_storage_is_touched(user_client: TestClient, env: pytest.MonkeyPatch) -> None:
    env.setenv("MAX_UPLOAD_SIZE", str(10 * MIB))
    get_settings.cache_clear()
    response = user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 11 * MIB})
    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_malicious_filename_cannot_influence_the_storage_key(user_client: TestClient, db: Session) -> None:
    response = user_client.post("/api/projects", json={"filename": "../../etc/passwd\x00.mp4", "size_bytes": 1000})
    assert response.status_code == 201
    project = db.scalar(select(Project))
    user = db.scalar(select(User))
    assert project is not None and user is not None
    assert project.original_filename == "passwd.mp4"
    assert project.original_file_key == f"uploads/{user.id}/{project.id}/original.mp4"
    assert ".." not in project.original_file_key


def test_part_urls_are_signed_and_bounded(user_client: TestClient) -> None:
    project_id = user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 12 * MIB}).json()["project"]["id"]

    ok = user_client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": [1, 3]})
    assert ok.status_code == 200
    urls = ok.json()["urls"]
    assert [u["part_number"] for u in urls] == [1, 3]
    assert all("X-Amz-Signature=" in u["url"] and "uploadId=" in u["url"] for u in urls)

    assert user_client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": [4]}).status_code == 422
    assert user_client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": []}).status_code == 422
    assert user_client.post(f"/api/projects/{project_id}/upload/urls", json={"part_numbers": list(range(1, 22))}).status_code == 422


# --- completion integrity ----------------------------------------------------------------------


def test_complete_requires_every_part(user_client: TestClient, videos: dict[str, Path]) -> None:
    data = videos["big"].read_bytes()
    body = user_client.post("/api/projects", json={"filename": "big.mp4", "size_bytes": len(data)}).json()
    project_id = body["project"]["id"]
    first = upload_parts(user_client, project_id, data, [1], body["upload"]["part_size"])

    response = user_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": first})
    assert response.status_code == 422
    assert "incomplete" in response.json()["detail"]
    assert user_client.get(f"/api/projects/{project_id}").json()["status"] == "UPLOADING"


def test_complete_rejects_a_claim_storage_cannot_back_up(user_client: TestClient) -> None:
    project_id = user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 1000}).json()["project"]["id"]
    # The browser says it uploaded part 1, but nothing was ever sent.
    response = user_client.post(
        f"/api/projects/{project_id}/upload/complete", json={"parts": [{"part_number": 1, "etag": '"made-up"'}]}
    )
    assert response.status_code == 422
    assert user_client.get(f"/api/projects/{project_id}").json()["status"] == "UPLOADING"


def test_wrong_declared_size_is_caught(user_client: TestClient, videos: dict[str, Path]) -> None:
    data = videos["small"].read_bytes()
    # Declares one byte more than will actually be sent.
    project_id = user_client.post("/api/projects", json={"filename": "small.mp4", "size_bytes": len(data) + 1}).json()["project"]["id"]
    parts = upload_parts(user_client, project_id, data, [1], 5 * MIB)
    response = user_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": parts})
    assert response.status_code == 422
    assert "doesn't match the file size" in response.json()["detail"]


# --- the file's real contents decide, not its name ---------------------------------------------


@pytest.mark.parametrize(
    ("video", "filename", "expected"),
    [
        ("garbage", "garbage.mp4", "may be corrupted"),
        ("truncated", "truncated.mp4", "may be corrupted"),
        ("webm", "actually-webm.mp4", "don't match its .mp4 extension"),
        ("small", "actually-mp4.webm", "don't match its .webm extension"),
        ("audio_only", "audio_only.mp4", "No video track"),
    ],
)
def test_invalid_contents_fail_with_a_message_the_person_can_act_on(
    user_client: TestClient, videos: dict[str, Path], video: str, filename: str, expected: str
) -> None:
    project = upload_video(user_client, videos[video], filename)
    assert project["status"] == "FAILED"
    assert expected in project["error"]
    assert project["width"] is None


def test_videos_over_the_duration_limit_are_rejected(user_client: TestClient, videos: dict[str, Path], env: pytest.MonkeyPatch) -> None:
    env.setenv("MAX_VIDEO_DURATION_SECONDS", "1")
    get_settings.cache_clear()
    project = upload_video(user_client, videos["small"])
    assert project["status"] == "FAILED"
    assert "limit is 0:01" in project["error"]


def test_videos_over_the_resolution_limit_are_rejected(user_client: TestClient, videos: dict[str, Path], env: pytest.MonkeyPatch) -> None:
    env.setenv("MAX_VIDEO_PIXELS", "1000")
    get_settings.cache_clear()
    project = upload_video(user_client, videos["small"])
    assert project["status"] == "FAILED"
    assert "320×568" in project["error"]


# --- retry, queue and storage failures ---------------------------------------------------------


def test_failed_analysis_can_be_retried_and_only_when_it_makes_sense(user_client: TestClient, videos: dict[str, Path]) -> None:
    failed = upload_video(user_client, videos["garbage"])
    retry = user_client.post(f"/api/projects/{failed['id']}/analyze")
    assert retry.status_code == 202
    assert retry.json()["status"] == "FAILED"  # same file, same outcome, but the retry ran

    good = upload_video(user_client, videos["small"])
    assert user_client.post(f"/api/projects/{good['id']}/analyze").status_code == 409


def test_queue_outage_leaves_the_upload_safe_and_retryable(
    user_client: TestClient, videos: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.api.projects as projects_api

    original = projects_api.enqueue_analysis

    def broken(_: uuid.UUID) -> None:
        raise EnqueueError("queue down")

    monkeypatch.setattr(projects_api, "enqueue_analysis", broken)
    data = videos["small"].read_bytes()
    project_id = user_client.post("/api/projects", json={"filename": "small.mp4", "size_bytes": len(data)}).json()["project"]["id"]
    parts = upload_parts(user_client, project_id, data, [1], 5 * MIB)

    response = user_client.post(f"/api/projects/{project_id}/upload/complete", json={"parts": parts})
    assert response.status_code == 503
    assert "couldn't start analyzing" in response.json()["detail"]
    assert user_client.get(f"/api/projects/{project_id}").json()["status"] == "UPLOADED"

    monkeypatch.setattr(projects_api, "enqueue_analysis", original)
    assert user_client.post(f"/api/projects/{project_id}/analyze").status_code == 202
    assert user_client.get(f"/api/projects/{project_id}").json()["status"] == "MASKING"


def test_storage_outage_gives_a_readable_error_not_a_500(user_client: TestClient) -> None:
    boto3.client("s3", region_name="us-east-1").delete_bucket(Bucket=BUCKET)
    response = user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 1000})
    assert response.status_code == 502
    assert "try again" in response.json()["detail"].lower()


def test_analysis_only_runs_for_projects_waiting_for_it(user_client: TestClient, videos: dict[str, Path], db: Session) -> None:
    project = upload_video(user_client, videos["small"])
    before = db.get(Project, uuid.UUID(project["id"]))
    assert before is not None and before.status == ProjectStatus.MASKING.value

    analyze_project(before.id)  # a duplicate delivery of the same task
    db.expire_all()
    after = db.get(Project, before.id)
    assert after is not None and after.status == ProjectStatus.MASKING.value


# --- listing and deleting ----------------------------------------------------------------------


def test_list_is_newest_first_and_only_shows_your_projects(
    user_client: TestClient, other_client: TestClient, videos: dict[str, Path]
) -> None:
    first = upload_video(user_client, videos["small"], "first.mp4")
    second = upload_video(user_client, videos["webm"], "second.webm")
    upload_video(other_client, videos["small"], "theirs.mp4")

    listed = user_client.get("/api/projects").json()
    assert [p["id"] for p in listed] == [second["id"], first["id"]]
    assert len(other_client.get("/api/projects").json()) == 1


def test_delete_removes_the_stored_file_and_the_project(user_client: TestClient, videos: dict[str, Path], db: Session) -> None:
    project = upload_video(user_client, videos["small"])
    key = db.get(Project, uuid.UUID(project["id"])).original_file_key  # type: ignore[union-attr]
    s3 = boto3.client("s3", region_name="us-east-1")
    assert s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] > 0

    assert user_client.delete(f"/api/projects/{project['id']}").status_code == 204
    with pytest.raises(ClientError):
        s3.head_object(Bucket=BUCKET, Key=key)
    assert user_client.get(f"/api/projects/{project['id']}").status_code == 404


def test_deleting_an_unfinished_upload_aborts_it(user_client: TestClient) -> None:
    project_id = user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 12 * MIB}).json()["project"]["id"]
    s3 = boto3.client("s3", region_name="us-east-1")
    assert len(s3.list_multipart_uploads(Bucket=BUCKET).get("Uploads", [])) == 1

    assert user_client.delete(f"/api/projects/{project_id}").status_code == 204
    assert s3.list_multipart_uploads(Bucket=BUCKET).get("Uploads", []) == []


# --- access control ----------------------------------------------------------------------------


def test_other_peoples_projects_look_like_they_do_not_exist(
    user_client: TestClient, other_client: TestClient, videos: dict[str, Path]
) -> None:
    mine = upload_video(user_client, videos["small"])
    pid = mine["id"]
    attempts = [
        other_client.get(f"/api/projects/{pid}"),
        other_client.get(f"/api/projects/{pid}/upload"),
        other_client.post(f"/api/projects/{pid}/upload/urls", json={"part_numbers": [1]}),
        other_client.post(f"/api/projects/{pid}/upload/complete", json={"parts": [{"part_number": 1, "etag": "x"}]}),
        other_client.post(f"/api/projects/{pid}/analyze"),
        other_client.delete(f"/api/projects/{pid}"),
    ]
    assert [a.status_code for a in attempts] == [404] * 6
    assert len({a.text for a in attempts}) == 1  # identical to a project that never existed
    assert user_client.get(f"/api/projects/{pid}").status_code == 200  # and nothing was deleted

    missing = other_client.get(f"/api/projects/{uuid.uuid4()}")
    assert missing.text == attempts[0].text


def test_every_projects_endpoint_requires_a_session(client: TestClient) -> None:
    pid = uuid.uuid4()
    calls = [
        client.get("/api/projects"),
        client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 1}),
        client.get(f"/api/projects/{pid}"),
        client.get(f"/api/projects/{pid}/upload"),
        client.post(f"/api/projects/{pid}/upload/urls", json={"part_numbers": [1]}),
        client.post(f"/api/projects/{pid}/upload/complete", json={"parts": [{"part_number": 1, "etag": "x"}]}),
        client.post(f"/api/projects/{pid}/analyze"),
        client.delete(f"/api/projects/{pid}"),
    ]
    assert [c.status_code for c in calls] == [401] * 8


def test_storage_never_sees_a_part_from_someone_elses_project(
    user_client: TestClient, other_client: TestClient
) -> None:
    pid = user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 1000}).json()["project"]["id"]
    assert other_client.post(f"/api/projects/{pid}/upload/urls", json={"part_numbers": [1]}).status_code == 404
    # Sanity check that the owner can still sign and upload.
    url = user_client.post(f"/api/projects/{pid}/upload/urls", json={"part_numbers": [1]}).json()["urls"][0]["url"]
    assert put_part(url, b"x" * 1000)
