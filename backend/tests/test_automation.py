"""Automatic detection: what it must find, what it must leave alone, and how the upload flow uses it."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.detect.build import document_for
from app.detect.service import removable, survey_and_detect
from app.models import Job, Mask, Project
from app.schemas.mask import Keyframe, MaskDocument, Region
from app.tracking.raster import rasterize
from app.tracking.shapes import shape_at
from app.video.source import inspect_source
from tests.conftest import upload_video
from tests.synthetic_video import SyntheticVideo, make_video

FRAMES = 60


def pan(frame: int) -> tuple[float, float]:
    """A camera that drifts, so the picture changes while a fixed overlay stays put."""
    return (frame * 1.6, 12.0 * np.sin(frame / 9.0))


def still_logo(frame: int) -> tuple[float, float, float, float]:
    return (380.0, 50.0, 1.0, 0.0)


@pytest.fixture(scope="module")
def logo_clip(tmp_path_factory: pytest.TempPathFactory) -> SyntheticVideo:
    root = tmp_path_factory.mktemp("auto_logo")
    return make_video(root / "logo.mp4", frames=FRAMES, pose=still_logo, background_shift=pan, sprite_size=(80, 44))


@pytest.fixture(scope="module")
def clean_clip(tmp_path_factory: pytest.TempPathFactory) -> SyntheticVideo:
    root = tmp_path_factory.mktemp("auto_clean")
    return make_video(root / "clean.mp4", frames=FRAMES, pose=still_logo, background_shift=pan, visible=lambda f: False)


def detect(clip: SyntheticVideo):  # noqa: ANN201
    media = inspect_source(clip.path)
    return survey_and_detect(clip.path, media, clip.frame_count, appearance=False)


def coverage(doc: MaskDocument, clip: SyntheticVideo, frame: int) -> tuple[float, float]:
    """(share of the true logo covered, share of the mask lying outside the logo)"""
    mask = np.zeros((clip.height, clip.width), np.uint8)
    for region in doc.regions:
        mask |= rasterize(shape_at(region, frame, None, clip.height / clip.width), clip.width, clip.height)
    truth = clip.outline_mask(frame) > 0
    found = mask > 0
    recall = float((truth & found).sum() / max(truth.sum(), 1))
    outside = float((found & ~truth).sum() / max(found.sum(), 1))
    return recall, outside


# --- the detector -------------------------------------------------------------------------------------


def test_a_fixed_logo_over_a_moving_picture_is_found_on_every_frame(logo_clip: SyntheticVideo) -> None:
    survey, report = detect(logo_clip)
    keep = removable(report)
    assert len(keep) == 1 and keep[0].verdict == "remove"
    doc = document_for(keep, survey)
    for frame in (0, FRAMES // 2, FRAMES - 1):
        recall, outside = coverage(doc, logo_clip, frame)
        assert recall > 0.9, f"frame {frame}: only {recall:.2f} of the logo is covered"
        assert outside < 0.6


def test_a_clean_video_gets_no_removal(clean_clip: SyntheticVideo) -> None:
    _, report = detect(clean_clip)
    assert removable(report) == []


def test_the_report_is_made_of_measurements(logo_clip: SyntheticVideo) -> None:
    _, report = detect(logo_clip)
    assert report.timings and all(v >= 0 for v in report.timings.values())
    assert report.samples > 0


def test_a_logo_that_comes_and_goes_is_only_masked_while_it_shows(tmp_path: Path) -> None:
    clip = make_video(
        tmp_path / "blink.mp4", frames=FRAMES, pose=still_logo, background_shift=pan, sprite_size=(80, 44),
        visible=lambda f: f >= 20,
    )  # fmt: skip
    survey, report = detect(clip)
    keep = removable(report)
    assert keep, "a logo shown for two thirds of the clip should be found"
    doc = document_for(keep, survey)
    early, _ = coverage_mask_share(doc, clip, 5)
    late, _ = coverage_mask_share(doc, clip, 45)
    assert early < 0.02, "nothing should be masked before the logo appears"
    assert late > 0.0005


def coverage_mask_share(doc: MaskDocument, clip: SyntheticVideo, frame: int) -> tuple[float, float]:
    mask = np.zeros((clip.height, clip.width), np.uint8)
    for region in doc.regions:
        mask |= rasterize(shape_at(region, frame, None, clip.height / clip.width), clip.width, clip.height)
    return float((mask > 0).mean()), 0.0


# --- hidden keyframes ------------------------------------------------------------------------------------


def _region(*frames: tuple[int, bool]) -> Region:
    return Region.model_validate(
        {
            "id": "r", "name": "R", "type": "rectangle",
            "keyframes": [
                {"frame": f, "rect": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}, **({"hidden": True} if h else {})}
                for f, h in frames
            ],
        }
    )  # fmt: skip


def test_hidden_is_left_out_of_saved_json_unless_set() -> None:
    assert "hidden" not in Keyframe.model_validate({"frame": 0, "rect": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}).model_dump(mode="json")
    hidden = Keyframe.model_validate({"frame": 0, "rect": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}, "hidden": True})
    assert hidden.model_dump(mode="json")["hidden"] is True


def test_a_hidden_keyframe_masks_nothing_until_the_next_one() -> None:
    region = _region((0, True), (10, False), (20, True), (30, False))
    sizes = {f: int(rasterize(shape_at(region, f, None, 1.0), 100, 100).sum()) for f in (0, 5, 10, 15, 20, 25, 30)}
    assert sizes[0] == 0 and sizes[5] == 0
    assert sizes[10] > 0 and sizes[15] > 0
    assert sizes[20] == 0 and sizes[25] == 0
    assert sizes[30] > 0


def test_a_shape_does_not_grow_out_of_nothing() -> None:
    """Between a hidden and a visible keyframe the shape is never blended (no half-size ghost mask)."""
    region = _region((0, False), (10, True))
    sizes = [int(rasterize(shape_at(region, f, None, 1.0), 100, 100).sum()) for f in range(0, 10)]
    assert len(set(sizes)) == 1, sizes


# --- the upload flow ---------------------------------------------------------------------------------------


@pytest.fixture
def automatic(env: pytest.MonkeyPatch) -> None:
    env.setenv("AUTO_PROCESS", "true")
    env.setenv("AUTO_APPEARANCE", "false")
    get_settings.cache_clear()


def test_uploading_a_video_with_a_logo_marks_and_renders_it_with_nobody_clicking(
    automatic: None, user_client: TestClient, db: Session, logo_clip: SyntheticVideo
) -> None:
    project = upload_video(user_client, logo_clip.path)
    assert project["status"] == "COMPLETED", project
    assert project["automation"]["removing"] == 1
    assert project["automation"]["state"] == "rendering"
    mask = db.scalar(select(Mask).where(Mask.project_id == uuid.UUID(project["id"])))
    assert mask is not None and mask.source == "auto"
    job = db.scalar(select(Job).where(Job.project_id == uuid.UUID(project["id"])))
    assert job is not None and job.status == "completed"
    assert job.settings["keep_audio"] is True
    status = user_client.get(f"/api/projects/{project['id']}/render").json()
    assert status["job"]["status"] == "completed"


def test_a_video_with_nothing_to_remove_is_not_pretended_processed(
    automatic: None, user_client: TestClient, db: Session, clean_clip: SyntheticVideo
) -> None:
    project = upload_video(user_client, clean_clip.path)
    assert project["status"] == "MASKING"
    assert project["automation"]["state"] == "nothing_found"
    assert db.scalar(select(Mask).where(Mask.project_id == uuid.UUID(project["id"]))) is None
    assert db.scalar(select(Job).where(Job.project_id == uuid.UUID(project["id"]))) is None


def test_with_automation_off_nothing_starts(
    env: pytest.MonkeyPatch, user_client: TestClient, db: Session, logo_clip: SyntheticVideo
) -> None:
    project = upload_video(user_client, logo_clip.path)
    assert project["status"] == "MASKING"
    assert project["automation"] is None
    assert db.scalar(select(Job).where(Job.project_id == uuid.UUID(project["id"]))) is None


def test_a_persons_own_marks_are_never_replaced_by_a_later_detection(db: Session, user_client: TestClient, logo_clip: SyntheticVideo) -> None:
    from app.services.automation import apply_detection

    project = upload_video(user_client, logo_clip.path)
    row = db.get(Project, uuid.UUID(project["id"]))
    assert row is not None
    manual = {"version": 1, "regions": [{"id": "mine", "name": "Mine", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}]}]}
    assert user_client.put(f"/api/projects/{project['id']}/mask", json=manual).status_code == 200
    survey, report = detect(logo_clip)
    doc = document_for(removable(report), survey)
    assert apply_detection(db, row, doc, {"found": 1}) is False
    saved = db.scalar(select(Mask).where(Mask.project_id == row.id))
    assert saved is not None and saved.source == "manual"
    assert saved.mask_data["regions"][0]["id"] == "mine"


def test_the_finished_render_carries_a_quality_report(
    automatic: None, user_client: TestClient, logo_clip: SyntheticVideo
) -> None:
    project = upload_video(user_client, logo_clip.path)
    result = user_client.get(f"/api/projects/{project['id']}/render").json()["job"]["result"]
    assert result["automatic"] is True
    quality = result["quality"]
    assert quality["regions"], "every removed region is checked against the original"
    for region in quality["regions"]:
        assert 0.0 <= region["residual"] or region["residual"] is None


def test_the_logo_area_of_the_result_really_changed(
    automatic: None, user_client: TestClient, logo_clip: SyntheticVideo, tmp_path: Path
) -> None:
    """A render that handed back the input would fail here: the stored result differs from the original in the logo."""
    from app.storage.s3 import get_storage
    from app.video.frames import FrameReader

    upload_video(user_client, logo_clip.path)
    storage = get_storage()
    listing = storage._client.list_objects_v2(Bucket=storage._bucket, Prefix="outputs/")  # noqa: SLF001
    [item] = listing["Contents"]
    result = tmp_path / "result.mp4"
    storage.download_to(item["Key"], result)
    frame = FRAMES // 2
    box = logo_clip.outline_mask(frame) > 0
    with FrameReader(logo_clip.path, inspect_source(logo_clip.path)) as before, FrameReader(result, inspect_source(result)) as after:
        original = list(before.frames())[frame].astype(np.float32)
        restored = list(after.frames())[frame].astype(np.float32)
    inside = float(np.abs(original - restored)[box].mean())
    outside = float(np.abs(original - restored)[~cv2.dilate(box.astype(np.uint8), np.ones((15, 15))).astype(bool)].mean())
    assert inside > 3 * max(outside, 1.0), (inside, outside)


def test_a_title_in_the_middle_of_the_picture_is_left_alone(tmp_path: Path) -> None:
    """A steady line of text over moving footage is a caption: removing it would damage the video."""
    clip = make_video(
        tmp_path / "title.mp4", frames=FRAMES, pose=lambda f: (240.0, 135.0, 1.0, 0.0), background_shift=pan, sprite_size=(150, 36)
    )
    _, report = detect(clip)
    assert removable(report) == []
    assert any("title or caption" in o.reason for o in report.overlays)
