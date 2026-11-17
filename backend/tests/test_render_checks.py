"""The mask for each frame at full resolution, and the checks run on a finished video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.tracking.tracker import track_document
from app.video.audio import mux_audio
from app.video.masks import MaskSequence
from app.video.proxy import build_proxy
from app.video.source import inspect_source
from app.video.verify import (
    ArtifactMonitor,
    Expectation,
    VerificationFailed,
    verify_output,
)
from tests.render_media import ffmpeg, make_source
from tests.synthetic_video import make_video
from tests.test_render_pipeline import box_document
from tests.test_tracker import doc_of, region_json

# --- masks -----------------------------------------------------------------------------------------


def test_a_rectangle_covers_the_same_share_of_the_frame_at_any_resolution() -> None:
    document = box_document(0.25, 0.1, 0.5, 0.4)
    shares = []
    for width, height in ((320, 180), (1280, 720), (1920, 1080)):
        mask = MaskSequence(document, None, width=width, height=height, frame_count=10).mask(0)
        assert mask.shape == (height, width) and mask.dtype == np.uint8
        shares.append(np.count_nonzero(mask) / mask.size)
    assert shares == pytest.approx([0.2] * 3, abs=0.005)


def test_a_still_mark_shares_one_mask_between_frames() -> None:
    sequence = MaskSequence(box_document(0.1, 0.1, 0.2, 0.2), None, width=320, height=180, frame_count=50)
    assert sequence.mask(0) is sequence.mask(49)
    assert len(list(sequence)) == 50


def test_all_marked_regions_are_removed_together() -> None:
    first = {"id": "a", "name": "A", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": 0.05, "y": 0.05, "w": 0.2, "h": 0.2}}]}
    second = {"id": "b", "name": "B", "type": "polygon", "keyframes": [{"frame": 0, "points": [[0.6, 0.6], [0.9, 0.6], [0.75, 0.9]]}]}
    mask = MaskSequence(doc_of(first, second), None, width=400, height=200, frame_count=5).mask(0)
    assert mask[20, 40] == 255 and mask[150, 300] == 255 and mask[100, 200] == 0


def test_brush_strokes_and_erasing_are_honoured() -> None:
    document = doc_of(
        {
            "id": "b", "name": "B", "type": "brush",
            "keyframes": [{
                "frame": 0,
                "strokes": [{"radius": 0.05, "points": [[0.2, 0.5], [0.8, 0.5]]}],
                "erase": [{"radius": 0.03, "points": [[0.5, 0.5]]}],
            }],
        }
    )  # fmt: skip
    mask = MaskSequence(document, None, width=400, height=200, frame_count=3).mask(0)
    assert mask[100, 120] == 255 and mask[100, 280] == 255
    assert mask[100, 200] == 0  # erased in the middle
    assert mask[20, 200] == 0


def test_tracked_motion_moves_the_mask_with_the_logo(tmp_path: Path) -> None:
    video = make_video(tmp_path / "v.mp4", width=480, height=270, frames=30, pose=lambda f: (100 + 9 * f, 90, 1.0, 0.0))
    document = doc_of(region_json(video, 0))
    proxy = tmp_path / "p.mp4"
    info = build_proxy(video.path, proxy)
    tracking = track_document(document, proxy, width=info.width, height=info.height, frame_count=info.frame_count)
    sequence = MaskSequence(document, tracking, width=480, height=270, frame_count=30)
    assert sequence.uses_tracking
    for frame in (0, 15, 29):
        truth = video.outline_mask(frame) > 0
        mask = sequence.mask(frame) > 0
        assert np.count_nonzero(truth & mask) / np.count_nonzero(truth | mask) > 0.8
    # at full resolution too: the same tracking applies to a render at four times the pixels
    large = MaskSequence(document, tracking, width=1920, height=1080, frame_count=30)
    small_mask = cv2.resize(large.mask(29), (480, 270), interpolation=cv2.INTER_AREA) > 127
    truth = video.outline_mask(29) > 0
    assert np.count_nonzero(truth & small_mask) / np.count_nonzero(truth | small_mask) > 0.8


def test_a_sequence_needs_real_dimensions() -> None:
    with pytest.raises(ValueError):
        MaskSequence(box_document(0, 0, 0.5, 0.5), None, width=1, height=100, frame_count=3)
    with pytest.raises(ValueError):
        MaskSequence(box_document(0, 0, 0.5, 0.5), None, width=100, height=100, frame_count=0)


# --- verification ----------------------------------------------------------------------------------


def _expect(**changes: object) -> Expectation:
    values: dict[str, object] = {"width": 128, "height": 72, "frames": 24, "fps": 30.0, "expect_audio": False, "source_audio_gap": None}
    values.update(changes)
    return Expectation(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def good(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_source(tmp_path_factory.mktemp("good") / "good.mp4", size="128x72", frames=24)


def test_a_correct_video_passes(good: Path) -> None:
    report = verify_output(good, _expect())
    assert (report.frames, report.width, report.height) == (24, 128, 72)
    assert report.fps == 30.0 and report.warnings == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frames": 25}, "missing frames"),
        ({"width": 130}, "should be 130"),
        ({"fps": 25.0}, "frame rate"),
        ({"expect_audio": True, "source_audio_gap": 0.0}, "audio was lost"),
    ],
)
def test_a_wrong_video_is_refused_with_a_plain_reason(good: Path, changes: dict[str, object], message: str) -> None:
    with pytest.raises(VerificationFailed, match=message):
        verify_output(good, _expect(**changes))


def test_audio_that_should_not_be_there_is_refused(tmp_path: Path) -> None:
    with_sound = make_source(tmp_path / "s.mp4", size="128x72", frames=24, audio="aac")
    with pytest.raises(VerificationFailed, match="shouldn't be there"):
        verify_output(with_sound, _expect(expect_audio=False))


def test_audio_that_drifted_from_the_picture_is_refused(tmp_path: Path) -> None:
    sound = make_source(tmp_path / "s.mp4", size="128x72", frames=24, audio="aac")
    # a picture 1 second longer than the sound it should match
    long_picture = tmp_path / "long.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x72:rate=30:duration=1.8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(long_picture))
    joined = tmp_path / "joined.mp4"
    mux_audio(long_picture, sound, joined, inspect_source(sound), container="mp4", keep_audio=True)
    with pytest.raises(VerificationFailed, match="lines up"):
        verify_output(joined, _expect(frames=54, expect_audio=True, source_audio_gap=0.0))
    # the same file is fine if the original had that same gap
    verify_output(joined, _expect(frames=54, expect_audio=True, source_audio_gap=-1.0))


def test_a_damaged_file_is_refused(good: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    data = bytearray(good.read_bytes())
    for position in range(len(data) // 2, len(data) // 2 + 400):
        data[position] ^= 0xFF
    broken.write_bytes(bytes(data))
    with pytest.raises(VerificationFailed):
        verify_output(broken, _expect())


def test_garbage_is_refused(videos: dict[str, Path]) -> None:
    with pytest.raises(VerificationFailed, match="couldn't be read back"):
        verify_output(videos["garbage"], _expect())


# --- shimmer monitor -------------------------------------------------------------------------------


def _scene_frames(noise: float, count: int = 12) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(1)
    base = (np.indices((90, 160)).sum(axis=0) % 64 * 2 + 60).astype(np.uint8)
    mask = np.zeros((90, 160), np.uint8)
    mask[30:60, 60:110] = 255
    frames = []
    for _ in range(count):
        frame = np.stack([base] * 3, axis=2).copy()
        if noise:
            patch = frame[mask > 0].astype(np.float32) + rng.normal(0, noise, frame[mask > 0].shape)
            frame[mask > 0] = np.clip(patch, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames, mask


def test_a_steady_patch_is_not_flagged() -> None:
    frames, mask = _scene_frames(0)
    monitor = ArtifactMonitor()
    for frame in frames:
        monitor.update(frame, mask)
    assert monitor.evaluated == len(frames) - 1 and monitor.fraction == 0.0


def test_a_shimmering_patch_is_flagged_on_every_pair() -> None:
    frames, mask = _scene_frames(30)
    monitor = ArtifactMonitor()
    for frame in frames:
        monitor.update(frame, mask)
    assert monitor.fraction == 1.0


def test_a_mark_that_moves_is_not_judged_on_pixels_it_did_not_cover_before() -> None:
    frames, mask = _scene_frames(0, 4)
    monitor = ArtifactMonitor()
    monitor.update(frames[0], mask)
    moved = np.roll(mask, 90, axis=1)  # no overlap with the previous mask
    monitor.update(frames[1], moved)
    assert monitor.evaluated == 0
