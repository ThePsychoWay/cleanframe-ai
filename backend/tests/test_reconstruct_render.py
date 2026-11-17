"""The reconstruction engine through the real render pipeline: MP4 in, FFmpeg decode and encode, audio, verification."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.ai.base import InpaintingError
from app.ai.dev import DevInpaintingEngine
from app.ai.temporal import TemporalReconstructionEngine
from app.video.audio import AudioAction
from app.video.frames import FrameReader
from app.video.pipeline import RenderSettings, render_restored_video
from app.video.source import inspect_source
from tests.test_render_pipeline import box_document
from tests.test_temporal_engine import (
    MARK,
    WINDOW_H,
    WINDOW_W,
    make_scene,
    psnr,
    watermark,
)

N = 30


def encode(frames: list[np.ndarray], path: Path, *, audio_seconds: float = 0.0) -> Path:
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WINDOW_W}x{WINDOW_H}", "-r", "30", "-i", "-",
    ]  # fmt: skip
    if audio_seconds:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_seconds:.3f}", "-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True, timeout=120)
    return path


def decode(path: Path) -> list[np.ndarray]:
    with FrameReader(path, inspect_source(path)) as reader:
        return list(reader.frames())


def clip(tmp_path: Path, *, pan: int):
    scene = make_scene()
    truth_frames, marked_frames = [], []
    for i in range(N):
        window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 + pan * i : 100 + pan * i + WINDOW_W])
        truth_frames.append(window)
        marked_frames.append(watermark(window))
    truth = encode(truth_frames, tmp_path / "truth.mp4")
    marked = encode(marked_frames, tmp_path / "marked.mp4", audio_seconds=N / 30)
    return truth, marked


def mark_region(frame_shape) -> np.ndarray:
    region = np.zeros(frame_shape[:2], bool)
    region[MARK[1] : MARK[3], MARK[0] : MARK[2]] = True
    return region


def render(source: Path, tmp_path: Path, engine, name: str = "out"):
    work = tmp_path / f"work-{name}"
    work.mkdir()
    x0, y0, x1, y1 = MARK
    document = box_document(x0 / WINDOW_W, y0 / WINDOW_H, (x1 - x0) / WINDOW_W, (y1 - y0) / WINDOW_H)
    return render_restored_video(
        source, tmp_path / name / "restored.mp4", document=document, tracking=None, engine=engine, work_dir=work,
        settings=RenderSettings(quality="balanced"), expected_frames=N,
    )  # fmt: skip


def test_a_watermark_is_removed_for_good_from_a_real_encoded_video(tmp_path: Path) -> None:
    truth, marked = clip(tmp_path, pan=5)
    engine = TemporalReconstructionEngine()
    result = render(marked, tmp_path, engine)

    assert result.engine == "reconstruct" and not result.development_only
    assert result.audio is AudioAction.COPIED
    assert any("rebuilt from other frames" in w for w in result.warnings)
    assert not any("blurs" in w or "development" in w for w in result.warnings)

    restored, clean, original = decode(result.path), decode(truth), decode(marked)
    region = mark_region(clean[0].shape)
    scores = [psnr(r[region], c[region]) for r, c in zip(restored, clean, strict=True)]
    baseline = render(marked, tmp_path, DevInpaintingEngine(), name="dev")
    dev_scores = [psnr(r[region], c[region]) for r, c in zip(decode(baseline.path), clean, strict=True)]
    # Both are measured after a lossy encode, so the ceiling is the codec, not the engine.
    assert min(scores[2:-2]) > 33, scores
    assert np.mean(scores) > np.mean(dev_scores) + 10
    # The logo itself is gone: inside the marked area every restored frame is close to the clean one.
    for index in (0, 15, N - 1):
        assert np.abs(restored[index][region].astype(int) - clean[index][region].astype(int)).mean() < 5
    assert len(restored) == len(original) == N


def test_a_fixed_mark_on_a_still_camera_stops_the_render_with_a_clear_message_and_leaves_nothing_behind(tmp_path: Path) -> None:
    _, marked = clip(tmp_path, pan=0)
    with pytest.raises(InpaintingError, match="hidden in every nearby frame"):
        render(marked, tmp_path, TemporalReconstructionEngine())
    assert not (tmp_path / "out" / "restored.mp4").exists()
    assert not list((tmp_path / "work-out").glob("*"))  # no half-written picture


def test_a_semi_transparent_logo_on_a_still_camera_is_taken_off_exactly_in_a_real_encoded_video(tmp_path: Path) -> None:
    from app.ai.auto import AutoRemovalEngine
    from tests.test_overlay_engine import LOGO_BOX, H, W, logo_coverage, make_clip

    truth_frames, shown, _ = make_clip(count=48, opacity=0.4)

    def write(frames, path, audio=False):
        command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1.6", "-c:a", "aac", "-shortest"]
        command += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True, timeout=120)
        return path

    truth = write(truth_frames, tmp_path / "truth.mp4")
    source = write(shown, tmp_path / "shown.mp4", audio=True)
    work = tmp_path / "work"
    work.mkdir()
    x0, y0, x1, y1 = LOGO_BOX
    result = render_restored_video(
        source, tmp_path / "out" / "restored.mp4", document=box_document(x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H),
        tracking=None, engine=AutoRemovalEngine(), work_dir=work, settings=RenderSettings(quality="balanced"), expected_frames=48,
    )  # fmt: skip
    assert result.audio is AudioAction.COPIED and not result.development_only
    assert any("inverted" in w for w in result.warnings), result.warnings

    clean, restored, original = decode(truth), decode(result.path), decode(source)
    inside = logo_coverage() > 0.02
    after = np.mean([psnr(r[inside], c[inside]) for r, c in zip(restored, clean, strict=True)])
    before = np.mean([psnr(o[inside], c[inside]) for o, c in zip(original, clean, strict=True)])
    assert after > 30 and after > before + 12, (before, after)


def test_a_clip_that_opens_on_a_still_card_is_still_measured_from_the_whole_clip(tmp_path: Path) -> None:
    """The first frames alone can't separate the logo from the picture (nothing moves there); frames from across the clip can."""
    from app.ai.auto import AutoRemovalEngine
    from app.ai.base import InpaintingOptions
    from app.ai.overlay import measure
    from tests.test_overlay_engine import LOGO_BOX, H, W, logo_coverage, make_clip

    still = 40
    truth_frames, shown, masks = make_clip(count=48, opacity=0.4)
    truth_frames = [truth_frames[0]] * still + list(truth_frames)
    shown = [shown[0]] * still + list(shown)
    masks = [masks[0]] * still + list(masks)

    with pytest.raises(InpaintingError):  # measured from the opening frames only, as before: refused
        measure([(i, shown[i], masks[i]) for i in range(60)], InpaintingOptions())

    def write(frames, path):
        command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-"]
        command += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True, timeout=120)
        return path

    truth, source = write(truth_frames, tmp_path / "truth.mp4"), write(shown, tmp_path / "shown.mp4")
    work = tmp_path / "work"
    work.mkdir()
    x0, y0, x1, y1 = LOGO_BOX
    result = render_restored_video(
        source, tmp_path / "out" / "restored.mp4", document=box_document(x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H),
        tracking=None, engine=AutoRemovalEngine(), work_dir=work, settings=RenderSettings(quality="balanced"), expected_frames=still + 48,
    )  # fmt: skip
    assert any("inverted" in w for w in result.warnings), result.warnings
    clean, restored, original = decode(truth), decode(result.path), decode(source)
    inside = logo_coverage() > 0.02
    after = np.mean([psnr(r[inside], c[inside]) for r, c in zip(restored, clean, strict=True)])
    before = np.mean([psnr(o[inside], c[inside]) for o, c in zip(original, clean, strict=True)])
    assert after > 30 and after > before + 12, (before, after)


def test_cancelling_during_the_survey_stops_the_render_before_anything_is_written(tmp_path: Path) -> None:
    from app.ai.auto import AutoRemovalEngine
    from app.video.ffmpeg import ProcessingCancelled
    from tests.test_overlay_engine import LOGO_BOX, H, W, make_clip

    _, shown, _ = make_clip(count=48, opacity=0.4)
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-"]
    source = tmp_path / "shown.mp4"
    subprocess.run([*command, "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(source)], input=b"".join(f.tobytes() for f in shown), check=True, timeout=120)
    work = tmp_path / "work"
    work.mkdir()
    x0, y0, x1, y1 = LOGO_BOX
    with pytest.raises(ProcessingCancelled):
        render_restored_video(
            source, tmp_path / "out" / "restored.mp4", document=box_document(x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H),
            tracking=None, engine=AutoRemovalEngine(), work_dir=work, expected_frames=48, progress=lambda stage, done, total: False,
        )  # fmt: skip
    assert not list(work.glob("*")) and not (tmp_path / "out" / "restored.mp4").exists()


def test_a_mark_that_changes_only_at_the_very_end_is_found_by_the_survey(tmp_path: Path) -> None:
    from app.video.pipeline import _survey
    from app.video.source import inspect_source as inspect
    from tests.test_overlay_engine import H, W, make_clip
    from tests.test_render_pipeline import doc_of

    _, shown, _ = make_clip(count=88, opacity=0.4)
    source = tmp_path / "shown.mp4"
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-"]
    subprocess.run([*command, "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(source)], input=b"".join(f.tobytes() for f in shown), check=True, timeout=120)
    rect = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5}
    document = doc_of({"id": "r", "name": "Mark", "type": "rectangle", "keyframes": [
        {"frame": 0, "rect": rect}, {"frame": 84, "rect": rect}, {"frame": 85, "rect": {**rect, "x": 0.2}},
    ]})  # fmt: skip
    kept = _survey(source, inspect(source), document, None, 88, 20, lambda *_: True)
    assert any(index >= 85 for index, _, _ in kept), [index for index, _, _ in kept]
    assert 20 <= len(kept) <= 21


def test_a_logo_that_fades_is_taken_off_in_a_real_encoded_video(tmp_path: Path) -> None:
    from app.ai.auto import AutoRemovalEngine
    from tests.test_overlay_engine import (
        LOGO_BOX,
        H,
        W,
        fade_gains,
        logo_coverage,
        make_clip,
    )

    count = 72
    truth_frames, shown, _ = make_clip(count=count, opacity=0.4, gains=fade_gains(count, fade=12))
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-"]

    def write(frames, path):
        subprocess.run([*command, "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)], input=b"".join(f.tobytes() for f in frames), check=True, timeout=120)
        return path

    truth, source = write(truth_frames, tmp_path / "truth.mp4"), write(shown, tmp_path / "shown.mp4")
    work = tmp_path / "work"
    work.mkdir()
    x0, y0, x1, y1 = LOGO_BOX
    result = render_restored_video(
        source, tmp_path / "out" / "restored.mp4", document=box_document(x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H),
        tracking=None, engine=AutoRemovalEngine(), work_dir=work, settings=RenderSettings(quality="balanced"), expected_frames=count,
    )  # fmt: skip
    assert any("strength changed" in w for w in result.warnings), result.warnings
    clean, restored, original = decode(truth), decode(result.path), decode(source)
    inside = logo_coverage() > 0.02
    after = np.mean([psnr(r[inside], c[inside]) for r, c in zip(restored, clean, strict=True)])
    before = np.mean([psnr(o[inside], c[inside]) for o, c in zip(original, clean, strict=True)])
    assert after > 28 and after > before + 6, (before, after)


def test_the_survey_is_reported_as_its_own_stage_before_the_rebuild(tmp_path: Path) -> None:
    from app.ai.auto import AutoRemovalEngine
    from tests.test_overlay_engine import LOGO_BOX, H, W, make_clip

    _, shown, _ = make_clip(count=40, opacity=0.4)
    source = tmp_path / "shown.mp4"
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-"]
    subprocess.run([*command, "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(source)], input=b"".join(f.tobytes() for f in shown), check=True, timeout=120)
    x0, y0, x1, y1 = LOGO_BOX
    document = box_document(x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H)

    def stages_for(engine, name: str) -> list[str]:
        seen: list[str] = []
        work = tmp_path / name
        work.mkdir()
        render_restored_video(
            source, work / "out" / "r.mp4", document=document, tracking=None, engine=engine, work_dir=work, expected_frames=40,
            progress=lambda stage, done, total: (seen.append(stage) if not seen or seen[-1] != stage else None, True)[1],
        )  # fmt: skip
        return seen

    auto = stages_for(AutoRemovalEngine(), "auto")
    assert auto[:2] == ["measuring", "reconstructing"], auto
    assert "measuring" not in stages_for(DevInpaintingEngine(), "dev")
