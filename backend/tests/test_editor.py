import subprocess
import uuid
from pathlib import Path
from typing import Any

import boto3
import pytest
import requests
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Mask, Project
from app.video.probe import run_ffprobe
from tests.conftest import BUCKET, upload_video


def rect_region(frame: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "id": "region_1",
        "name": "Top-left logo",
        "type": "rectangle",
        "keyframes": [{"frame": frame, "rect": {"x": 0.05, "y": 0.05, "w": 0.2, "h": 0.1}}],
        **extra,
    }


def polygon_region() -> dict[str, Any]:
    return {
        "id": "region_2",
        "name": "Center text",
        "type": "polygon",
        "keyframes": [
            {"frame": 0, "points": [[0.3, 0.4], [0.7, 0.4], [0.5, 0.6]]},
            {"frame": 30, "points": [[0.35, 0.45], [0.75, 0.45], [0.55, 0.65]]},
        ],
    }


def brush_region() -> dict[str, Any]:
    return {
        "id": "region_3",
        "name": "Username",
        "type": "brush",
        "keyframes": [
            {
                "frame": 10,
                "strokes": [{"radius": 0.02, "points": [[0.6, 0.9], [0.7, 0.9]]}],
                "erase": [{"radius": 0.01, "points": [[0.65, 0.9]]}],
            }
        ],
    }


def doc(*regions: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "regions": list(regions)}


@pytest.fixture
def project(user_client: TestClient, videos: dict[str, Path]) -> dict[str, Any]:
    result = upload_video(user_client, videos["small"])  # 320x568, 30 fps, 60 frames
    assert result["status"] == "MASKING"
    return result


# --- preview media -----------------------------------------------------------------------------


def test_a_variable_frame_rate_video_gets_the_previews_timeline(user_client: TestClient, videos: dict[str, Path], tmp_path: Path) -> None:
    """The source stores 90 frames at uneven times; the preview is resampled to a constant rate. Masks and
    tracking count frames on the preview, so the project's frame count and rate must be the preview's."""
    result = upload_video(user_client, videos["vfr"])
    assert result["status"] == "MASKING"
    media = user_client.get(f"/api/projects/{result['id']}/media").json()
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(requests.get(media["proxy_url"], timeout=30).content)
    video = next(s for s in run_ffprobe(proxy)["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 100  # not the 90 frames the source stores
    assert result["frame_count"] == 100 and media["frame_count"] == 100
    assert result["fps"] == pytest.approx(30) and media["fps"] == pytest.approx(30)


def test_media_links_serve_a_real_proxy_and_thumbnail(user_client: TestClient, project: dict[str, Any], tmp_path: Path) -> None:
    response = user_client.get(f"/api/projects/{project['id']}/media")
    assert response.status_code == 200
    media = response.json()
    assert media["expires_in"] == 3600
    assert (media["width"], media["height"]) == (320, 568)  # smaller than the cap, so never upscaled
    assert media["fps"] == pytest.approx(30) and media["frame_count"] == 60

    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(requests.get(media["proxy_url"], timeout=30).content)
    probe = run_ffprobe(proxy)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p"
    assert not any(s["codec_type"] == "audio" for s in probe["streams"])  # the editor is silent by design
    assert int(video["nb_frames"]) == 60  # frame N in the proxy is frame N in the source

    thumbnail = requests.get(media["thumbnail_url"], timeout=30)
    assert thumbnail.content[:3] == b"\xff\xd8\xff"
    assert thumbnail.headers["Content-Type"] == "image/jpeg"


def _make_video(path: Path, *filters: str, size: str, extra_input: list[str] | None = None) -> Path:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *(extra_input or []), "-f", "lavfi", "-i",
               f"testsrc2=size={size}:rate=24:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, timeout=60)
    return path


def test_large_videos_are_scaled_down_to_the_cap(user_client: TestClient, tmp_path: Path) -> None:
    result = upload_video(user_client, _make_video(tmp_path / "wide.mp4", size="1600x1200"))
    media = user_client.get(f"/api/projects/{result['id']}/media").json()
    assert (media["width"], media["height"]) == (960, 720)


def test_odd_dimensions_become_even_so_the_proxy_always_encodes(user_client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "odd.mp4"
    # Encode odd dimensions with 4:4:4 (4:2:0 can't hold them), so the *source* is odd-sized.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=321x241:rate=24:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv444p", str(path)],
                   check=True, timeout=60)
    result = upload_video(user_client, path)
    assert result["status"] == "MASKING"
    media = user_client.get(f"/api/projects/{result['id']}/media").json()
    assert media["width"] % 2 == 0 and media["height"] % 2 == 0
    assert (media["width"], media["height"]) == (320, 240)


def test_rotated_phone_video_gets_an_upright_proxy(user_client: TestClient, tmp_path: Path) -> None:
    flat = _make_video(tmp_path / "flat.mp4", size="640x360")
    rotated = tmp_path / "rotated.mp4"
    done = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90", "-i",
                           str(flat), "-c", "copy", str(rotated)], capture_output=True, timeout=60, check=False)
    if done.returncode != 0:
        pytest.skip("this ffmpeg can't write rotation metadata")
    result = upload_video(user_client, rotated)
    assert (result["width"], result["height"]) == (360, 640)  # as displayed
    media = user_client.get(f"/api/projects/{result['id']}/media").json()
    assert (media["width"], media["height"]) == (360, 640)


def test_media_is_not_available_until_the_preview_exists(user_client: TestClient) -> None:
    pid = user_client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 1000}).json()["project"]["id"]
    response = user_client.get(f"/api/projects/{pid}/media")
    assert response.status_code == 409
    assert "isn't ready" in response.json()["detail"]


def test_rejected_videos_never_get_a_preview(user_client: TestClient, videos: dict[str, Path]) -> None:
    failed = upload_video(user_client, videos["garbage"])
    assert failed["status"] == "FAILED"
    assert user_client.get(f"/api/projects/{failed['id']}/media").status_code == 409


def test_deleting_a_project_removes_its_preview_and_masks(
    user_client: TestClient, project: dict[str, Any], db: Session
) -> None:
    assert user_client.put(f"/api/projects/{project['id']}/mask", json=doc(rect_region())).status_code == 200
    row = db.get(Project, uuid.UUID(project["id"]))
    assert row is not None and row.proxy_file_key and row.thumbnail_file_key
    keys = [row.original_file_key, row.proxy_file_key, row.thumbnail_file_key]
    s3 = boto3.client("s3", region_name="us-east-1")
    assert all(s3.head_object(Bucket=BUCKET, Key=k) for k in keys)

    assert user_client.delete(f"/api/projects/{project['id']}").status_code == 204

    for key in keys:
        with pytest.raises(ClientError):
            s3.head_object(Bucket=BUCKET, Key=key)
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Mask)) == 0


# --- saving and loading masks ------------------------------------------------------------------


def test_a_project_with_no_mask_returns_an_empty_document(user_client: TestClient, project: dict[str, Any]) -> None:
    body = user_client.get(f"/api/projects/{project['id']}/mask").json()
    assert body == {"mask_data": {"version": 1, "regions": []}, "tracking_status": "none", "updated_at": None}


def test_masks_round_trip_exactly_and_replace_on_save(user_client: TestClient, project: dict[str, Any], db: Session) -> None:
    url = f"/api/projects/{project['id']}/mask"
    first = doc(rect_region(), polygon_region(), brush_region())

    saved = user_client.put(url, json=first)
    assert saved.status_code == 200
    assert saved.json()["mask_data"] == first  # nothing dropped, reordered or rounded
    assert saved.json()["updated_at"] is not None
    assert user_client.get(url).json()["mask_data"] == first

    row = db.scalar(select(Mask))
    assert row is not None and (row.frame_start, row.frame_end) == (0, 30)

    second = doc(rect_region(frame=5))
    assert user_client.put(url, json=second).json()["mask_data"] == second
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Mask)) == 1  # replaced, not appended
    assert db.scalar(select(Mask)).frame_start == 5  # type: ignore[union-attr]


def test_clearing_all_regions_is_a_valid_save(user_client: TestClient, project: dict[str, Any]) -> None:
    url = f"/api/projects/{project['id']}/mask"
    user_client.put(url, json=doc(rect_region()))
    assert user_client.put(url, json=doc()).json()["mask_data"]["regions"] == []


def test_a_keyframe_past_the_last_frame_is_rejected(user_client: TestClient, project: dict[str, Any]) -> None:
    url = f"/api/projects/{project['id']}/mask"
    assert user_client.put(url, json=doc(rect_region(frame=59))).status_code == 200  # the last of 60
    response = user_client.put(url, json=doc(rect_region(frame=60)))
    assert response.status_code == 422
    assert "has 60 frames" in response.json()["detail"]


def test_masks_can_only_be_saved_while_the_project_is_ready_for_masking(
    user_client: TestClient, videos: dict[str, Path]
) -> None:
    failed = upload_video(user_client, videos["garbage"])
    response = user_client.put(f"/api/projects/{failed['id']}/mask", json=doc(rect_region()))
    assert response.status_code == 409


def _bad(**override: Any) -> dict[str, Any]:
    region = rect_region()
    region.update(override)
    return doc(region)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(doc({**rect_region(), "keyframes": [{"frame": 0, "rect": {"x": 0.9, "y": 0.1, "w": 0.3, "h": 0.1}}]}), id="rect-leaves-frame"),
        pytest.param(doc({**rect_region(), "keyframes": [{"frame": 0, "rect": {"x": 0.1, "y": 0.1, "w": 0, "h": 0.1}}]}), id="rect-zero-width"),
        pytest.param(doc({**rect_region(), "keyframes": [{"frame": 0, "rect": {"x": -0.1, "y": 0.1, "w": 0.1, "h": 0.1}}]}), id="rect-negative"),
        pytest.param(doc({**polygon_region(), "keyframes": [{"frame": 0, "points": [[0.1, 0.1], [0.2, 0.2]]}]}), id="polygon-two-points"),
        pytest.param(doc({**polygon_region(), "keyframes": [{"frame": 0, "points": [[0.1, 0.1], [0.2, 1.5], [0.3, 0.3]]}]}), id="point-outside"),
        pytest.param(doc({**polygon_region(), "keyframes": [{"frame": 0, "points": [[0.1], [0.2, 0.2], [0.3, 0.3]]}]}), id="point-malformed"),
        pytest.param(_bad(type="polygon"), id="rect-shape-in-polygon-region"),
        pytest.param(_bad(type="circle"), id="unknown-type"),
        pytest.param(doc({**rect_region(), "keyframes": [{"frame": 5, "rect": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}, {"frame": 5, "rect": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}]}), id="duplicate-frame"),
        pytest.param(doc({**rect_region(), "keyframes": [{"frame": 9, "rect": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}, {"frame": 2, "rect": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}]}), id="keyframes-out-of-order"),
        pytest.param(doc({**rect_region(), "keyframes": [{"frame": -1, "rect": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}]}), id="negative-frame"),
        pytest.param(_bad(keyframes=[]), id="no-keyframes"),
        pytest.param(doc(rect_region(), rect_region()), id="duplicate-region-ids"),
        pytest.param(_bad(id="../../etc"), id="path-in-id"),
        pytest.param(_bad(name="   "), id="blank-name"),
        pytest.param(_bad(name="x" * 61), id="name-too-long"),
        pytest.param(_bad(extra="field"), id="unexpected-field"),
        pytest.param(doc(*[{**rect_region(), "id": f"r{i}"} for i in range(21)]), id="too-many-regions"),
        pytest.param({"version": 2, "regions": []}, id="unknown-version"),
        pytest.param(doc({**brush_region(), "keyframes": [{"frame": 0, "strokes": [{"radius": 0.9, "points": [[0.5, 0.5]]}]}]}), id="brush-too-big"),
        pytest.param(doc({**brush_region(), "keyframes": [{"frame": 0, "strokes": [{"radius": 0.02, "points": []}]}]}), id="stroke-no-points"),
        pytest.param(doc({**brush_region(), "keyframes": [{"frame": 0, "strokes": []}]}), id="brush-no-strokes"),
    ],
)
def test_invalid_masks_are_rejected_and_nothing_is_saved(
    user_client: TestClient, project: dict[str, Any], payload: dict[str, Any], db: Session
) -> None:
    response = user_client.put(f"/api/projects/{project['id']}/mask", json=payload)
    assert response.status_code == 422, response.text
    assert db.scalar(select(func.count()).select_from(Mask)) == 0


def test_a_mask_that_is_too_detailed_is_rejected(user_client: TestClient, project: dict[str, Any]) -> None:
    # 26 keyframes x 2000 points = 52,000 points: over the point budget, yet under the body-size cap.
    stroke = {"radius": 0.01, "points": [[0.5, 0.5]] * 2000}
    heavy = {"id": "heavy", "name": "Heavy", "type": "brush",
             "keyframes": [{"frame": f, "strokes": [stroke]} for f in range(26)]}
    response = user_client.put(f"/api/projects/{project['id']}/mask", json=doc(heavy))
    assert response.status_code == 422
    assert "too detailed" in response.text


def test_oversized_bodies_are_refused_before_they_are_parsed(user_client: TestClient, project: dict[str, Any]) -> None:
    response = user_client.put(
        f"/api/projects/{project['id']}/mask",
        content=b" " * (3 * 1024 * 1024),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_region_names_are_stripped_of_control_characters(user_client: TestClient, project: dict[str, Any]) -> None:
    response = user_client.put(f"/api/projects/{project['id']}/mask", json=_bad(name="  Logo\x00\x07 A  "))
    assert response.status_code == 200
    assert response.json()["mask_data"]["regions"][0]["name"] == "Logo A"


def test_saving_resets_tracking_status(user_client: TestClient, project: dict[str, Any], db: Session) -> None:
    url = f"/api/projects/{project['id']}/mask"
    user_client.put(url, json=doc(rect_region()))
    row = db.scalar(select(Mask))
    assert row is not None
    row.tracking_status = "done"  # as a later phase's tracking would leave it
    db.commit()

    assert user_client.put(url, json=doc(rect_region(frame=3))).json()["tracking_status"] == "none"


# --- access control ----------------------------------------------------------------------------


def test_other_peoples_masks_and_media_look_like_they_do_not_exist(
    user_client: TestClient, other_client: TestClient, project: dict[str, Any], db: Session
) -> None:
    pid = project["id"]
    user_client.put(f"/api/projects/{pid}/mask", json=doc(rect_region()))

    attempts = [
        other_client.get(f"/api/projects/{pid}/media"),
        other_client.get(f"/api/projects/{pid}/mask"),
        other_client.put(f"/api/projects/{pid}/mask", json=doc(polygon_region())),
    ]
    assert [a.status_code for a in attempts] == [404, 404, 404]
    assert len({a.text for a in attempts}) == 1
    assert user_client.get(f"/api/projects/{pid}/mask").json()["mask_data"]["regions"][0]["id"] == "region_1"


def test_editor_endpoints_require_a_session(client: TestClient) -> None:
    pid = uuid.uuid4()
    calls = [
        client.get(f"/api/projects/{pid}/media"),
        client.get(f"/api/projects/{pid}/mask"),
        client.put(f"/api/projects/{pid}/mask", json=doc()),
    ]
    assert [c.status_code for c in calls] == [401, 401, 401]
