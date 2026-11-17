"""The whole render, on real files: decode, remove the marked area, encode, put the sound back, verify."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.ai.base import (
    EngineUnavailableError,
    Frame,
    InpaintingEngine,
    InpaintingError,
    InpaintingOptions,
    Mask,
)
from app.ai.dev import DevInpaintingEngine
from app.schemas.mask import MaskDocument
from app.tracking.tracker import track_document
from app.video.audio import AudioAction
from app.video.ffmpeg import ProcessingCancelled, ProcessingError
from app.video.frames import FrameReader
from app.video.pipeline import RenderSettings, render_restored_video
from app.video.proxy import build_proxy
from app.video.source import inspect_source
from app.video.verify import VerificationFailed
from tests.render_media import child_processes, make_source, pcm, stream_of
from tests.synthetic_video import SyntheticVideo, make_video
from tests.test_inpaint_consistency import GenerativeStandIn
from tests.test_tracker import doc_of, region_json

N = 24


def read(path: Path) -> list[np.ndarray]:
    with FrameReader(path, inspect_source(path)) as reader:
        return list(reader.frames())


def error_in_logo(candidate: list[np.ndarray], truth: list[np.ndarray], video: SyntheticVideo) -> float:
    errors = []
    for index, (a, b) in enumerate(zip(candidate, truth, strict=True)):
        region = cv2.dilate(video.outline_mask(index), np.ones((5, 5), np.uint8)) > 0
        if region.any():
            errors.append(np.abs(a[region].astype(float) - b[region].astype(float)).mean())
    return float(np.mean(errors))


def with_audio(silent: Path, destination: Path, seconds: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-f", "lavfi", "-i", f"sine=frequency=523:duration={seconds:.3f}",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(destination)],
        check=True, timeout=60,
    )  # fmt: skip
    return destination


class Scene:
    """A logo that stays put over a still background, with the ground truth (no logo) next to it."""

    def __init__(self, root: Path) -> None:
        kw: dict[str, Any] = {"width": 320, "height": 180, "frames": N, "pose": lambda f: (230, 60, 1.0, 0.0)}
        self.video = make_video(root / "silent.mp4", **kw)
        self.clean = make_video(root / "clean.mp4", **kw, visible=lambda f: False)
        self.source = with_audio(self.video.path, root / "source.mp4", N / 30)
        self.document = doc_of(region_json(self.video, 0))
        self.truth = read(self.clean.path)
        self.original = read(self.source)


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> Scene:
    return Scene(tmp_path_factory.mktemp("scene"))


def box_document(x: float, y: float, w: float, h: float) -> MaskDocument:
    return doc_of(
        {"id": "r", "name": "Mark", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": x, "y": y, "w": w, "h": h}}]}
    )


def render(scene: Scene, tmp_path: Path, **kwargs: Any):  # noqa: ANN202
    kwargs.setdefault("document", scene.document)
    kwargs.setdefault("tracking", None)
    kwargs.setdefault("engine", DevInpaintingEngine())
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    source = kwargs.pop("source", scene.source)
    return render_restored_video(source, tmp_path / "out" / "restored.mp4", work_dir=work, **kwargs)


# --- the result ------------------------------------------------------------------------------------


def test_the_marked_logo_is_gone_and_everything_else_is_kept(scene: Scene, tmp_path: Path) -> None:
    result = render(scene, tmp_path, expected_frames=N)
    out = read(result.path)
    assert len(out) == N and out[0].shape == scene.original[0].shape
    assert (result.frames, result.width, result.height) == (N, 320, 180)
    assert result.fps == pytest.approx(30)

    before = error_in_logo(scene.original, scene.truth, scene.video)
    after = error_in_logo(out, scene.truth, scene.video)
    assert after < before / 3, (before, after)  # the logo is replaced by something close to the background

    away = np.ones((180, 320), bool)
    away[:100, 150:] = False  # everything except a generous box around the logo
    outside = np.mean([np.abs(a[away].astype(float) - b[away].astype(float)).mean() for a, b in zip(out, scene.original, strict=True)])
    assert outside < 3.0  # only encoder noise away from the mark


def test_the_audio_comes_back_unchanged_and_the_result_says_so(scene: Scene, tmp_path: Path) -> None:
    result = render(scene, tmp_path, expected_frames=N)
    assert result.audio is AudioAction.COPIED and result.audio_codec == "aac"
    assert pcm(result.path) == pcm(scene.source)
    assert stream_of(result.path, "audio")["codec_name"] == "aac"


def test_the_result_is_honest_about_the_development_engine(scene: Scene, tmp_path: Path) -> None:
    result = render(scene, tmp_path)
    assert result.development_only and result.engine == "dev"
    assert any("development engine" in w and "not an AI" in w for w in result.warnings)
    assert result.verification.artifact_fraction == 0.0
    assert not result.tracking_used


def test_progress_is_reported_from_real_work_and_is_in_order(scene: Scene, tmp_path: Path) -> None:
    seen: list[tuple[str, int, int]] = []
    render(scene, tmp_path, expected_frames=N, progress=lambda s, d, t: seen.append((s, d, t)) or True)
    stages = [s for s, _, _ in seen]
    assert stages[0] == "reconstructing" and stages[-1] == "verifying"
    assert stages.index("audio") > max(i for i, s in enumerate(stages) if s == "reconstructing")
    done = [d for s, d, _ in seen if s == "reconstructing"]
    assert done == list(range(1, N + 1))  # one report per frame actually finished, never ahead of the work
    assert all(d <= t for _, d, t in seen)


def test_no_audio_option_removes_it(scene: Scene, tmp_path: Path) -> None:
    result = render(scene, tmp_path, settings=RenderSettings(keep_audio=False))
    assert result.audio is AudioAction.REMOVED
    with pytest.raises(StopIteration):
        stream_of(result.path, "audio")


def test_a_silent_original_gives_a_silent_result(scene: Scene, tmp_path: Path) -> None:
    result = render(scene, tmp_path, source=scene.video.path)
    assert result.audio is AudioAction.NONE


def test_webm_output(scene: Scene, tmp_path: Path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    result = render_restored_video(
        scene.source, tmp_path / "r.webm", document=scene.document, tracking=None, engine=DevInpaintingEngine(),
        work_dir=work, settings=RenderSettings(container="webm", quality="fast"), expected_frames=N,
    )  # fmt: skip
    assert stream_of(result.path, "video")["codec_name"] == "vp9"
    assert result.audio is AudioAction.CONVERTED and stream_of(result.path, "audio")["codec_name"] == "opus"
    assert len(read(result.path)) == N


def test_work_files_and_processes_are_cleaned_up_after_success(scene: Scene, tmp_path: Path) -> None:
    before = set(child_processes())
    render(scene, tmp_path)
    assert list((tmp_path / "work").iterdir()) == []
    assert set(child_processes()) <= before


# --- sizes and timing ------------------------------------------------------------------------------


def test_the_output_height_cap_scales_a_large_video_down(tmp_path: Path) -> None:
    video = make_video(tmp_path / "hd.mp4", width=1920, height=1080, frames=6, pose=lambda f: (1500, 250, 1.0, 0.0), sprite_size=(300, 160))
    document = doc_of(region_json(video, 0))
    work = tmp_path / "w"
    work.mkdir()
    result = render_restored_video(
        video.path, tmp_path / "o.mp4", document=document, tracking=None, engine=DevInpaintingEngine(), work_dir=work,
        settings=RenderSettings(output_height=720, quality="fast"),
    )  # fmt: skip
    assert (result.width, result.height) == (1280, 720)
    assert (stream_of(result.path, "video")["width"], stream_of(result.path, "video")["height"]) == (1280, 720)
    # the mask was applied at 1080p, then the result scaled: the logo's place in the small frame is clean
    frame = read(result.path)[3]
    logo_area = frame[int(250 / 1.5) - 40 : int(250 / 1.5) + 40, int(1500 / 1.5) - 80 : int(1500 / 1.5) + 80]
    assert not ((logo_area[..., 2] > 180) & (logo_area[..., 0] < 80)).any()  # no pure logo-red left


def test_ntsc_frame_rate_is_exact(tmp_path: Path) -> None:
    source = make_source(tmp_path / "n.mp4", size="192x108", rate="30000/1001", frames=24, audio="aac")
    document = box_document(0.6, 0.1, 0.3, 0.2)
    work = tmp_path / "w"
    work.mkdir()
    result = render_restored_video(source, tmp_path / "o.mp4", document=document, tracking=None, engine=DevInpaintingEngine(), work_dir=work, expected_frames=24)
    assert stream_of(result.path, "video")["r_frame_rate"] == "30000/1001"
    assert result.frames == 24


def test_variable_frame_rate_follows_the_previews_timeline(videos: dict[str, Path], tmp_path: Path) -> None:
    source = videos["vfr"]
    proxy = build_proxy(source, tmp_path / "proxy.mp4")
    document = box_document(0.1, 0.1, 0.2, 0.2)
    work = tmp_path / "w"
    work.mkdir()
    result = render_restored_video(
        source, tmp_path / "o.mp4", document=document, tracking=None, engine=DevInpaintingEngine(), work_dir=work,
        expected_frames=proxy.frame_count,
    )  # fmt: skip
    assert result.frames == proxy.frame_count == 100
    assert result.fps == pytest.approx(30)


def test_a_timeline_that_does_not_match_the_video_stops_the_render(videos: dict[str, Path], tmp_path: Path) -> None:
    document = box_document(0.1, 0.1, 0.2, 0.2)
    work = tmp_path / "w"
    work.mkdir()
    with pytest.raises(ProcessingError, match="timeline"):
        render_restored_video(
            videos["vfr"], tmp_path / "o.mp4", document=document, tracking=None, engine=DevInpaintingEngine(), work_dir=work, expected_frames=90
        )  # the stored frame count of the file, not what the editor showed
    assert not (tmp_path / "o.mp4").exists()


# --- tracking --------------------------------------------------------------------------------------


def test_a_tracked_moving_logo_is_removed_where_it_really_is(tmp_path: Path) -> None:
    kw: dict[str, Any] = {"width": 480, "height": 270, "frames": 30, "pose": lambda f: (120 + 8 * f, 80 + 2 * f, 1.0, 0.0)}
    video = make_video(tmp_path / "silent.mp4", **kw, background_shift=lambda f: (f * 2.0, f * 1.0))
    clean = make_video(tmp_path / "clean.mp4", **kw, background_shift=lambda f: (f * 2.0, f * 1.0), visible=lambda f: False)
    source = with_audio(video.path, tmp_path / "source.mp4", 1.0)
    document = doc_of(region_json(video, 0))
    proxy = tmp_path / "proxy.mp4"
    info = build_proxy(source, proxy)
    tracking = track_document(document, proxy, width=info.width, height=info.height, frame_count=info.frame_count)

    work = tmp_path / "w"
    work.mkdir()
    result = render_restored_video(
        source, tmp_path / "o.mp4", document=document, tracking=tracking, engine=DevInpaintingEngine(), work_dir=work,
        expected_frames=info.frame_count,
    )  # fmt: skip
    assert result.tracking_used
    out, original, truth = read(result.path), read(source), read(clean.path)
    before, after = error_in_logo(original, truth, video), error_in_logo(out, truth, video)
    assert after < before / 3, (before, after)

    # Without tracking the same single keyframe covers only where the logo started, so the logo is mostly left.
    work2 = tmp_path / "w2"
    work2.mkdir()
    untracked = render_restored_video(
        source, tmp_path / "u.mp4", document=document, tracking=None, engine=DevInpaintingEngine(), work_dir=work2, expected_frames=info.frame_count
    )  # fmt: skip
    assert error_in_logo(read(untracked.path), truth, video) > 2 * after


# --- failures --------------------------------------------------------------------------------------


class Wrapped(InpaintingEngine):
    provider = "dev"
    name = "test engine"

    def __init__(self) -> None:
        self.inner = DevInpaintingEngine()

    def is_available(self) -> bool:
        return True

    def process(self, frames: Iterable[Frame], masks: Iterable[Mask], options: InpaintingOptions) -> Iterator[Frame]:
        return self.inner.process(frames, masks, options)


class FailsAfter(Wrapped):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    def process(self, frames, masks, options):  # type: ignore[no-untyped-def]
        for index, frame in enumerate(super().process(frames, masks, options)):
            if index == self.count:
                raise InpaintingError("The reconstruction stopped unexpectedly.")
            yield frame


class DropsLastFrame(Wrapped):
    def process(self, frames, masks, options):  # type: ignore[no-untyped-def]
        produced = list(super().process(frames, masks, options))
        yield from produced[:-1]


class Down(Wrapped):
    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return "The model file is missing."


def test_cancelling_stops_the_work_and_leaves_nothing_behind(scene: Scene, tmp_path: Path) -> None:
    before = set(child_processes())
    with pytest.raises(ProcessingCancelled):
        render(scene, tmp_path, progress=lambda stage, done, total: done < 5)
    assert not (tmp_path / "out" / "restored.mp4").exists()
    assert list((tmp_path / "work").iterdir()) == []
    assert set(child_processes()) <= before


def test_cancelling_during_the_audio_step_leaves_nothing_behind(scene: Scene, tmp_path: Path) -> None:
    with pytest.raises(ProcessingCancelled):
        render(scene, tmp_path, progress=lambda stage, done, total: stage != "audio")
    assert not (tmp_path / "out" / "restored.mp4").exists()
    assert list((tmp_path / "work").iterdir()) == []


def test_an_engine_that_fails_halfway_fails_the_render_cleanly(scene: Scene, tmp_path: Path) -> None:
    before = set(child_processes())
    with pytest.raises(InpaintingError, match="stopped unexpectedly"):
        render(scene, tmp_path, engine=FailsAfter(5))
    assert not (tmp_path / "out" / "restored.mp4").exists()
    assert list((tmp_path / "work").iterdir()) == []
    assert set(child_processes()) <= before


def test_an_engine_that_returns_too_few_frames_is_caught(scene: Scene, tmp_path: Path) -> None:
    with pytest.raises(ProcessingError, match="different number of frames"):
        render(scene, tmp_path, engine=DropsLastFrame())
    assert not (tmp_path / "out" / "restored.mp4").exists()


def test_an_unavailable_engine_fails_before_any_video_is_read(scene: Scene, tmp_path: Path) -> None:
    with pytest.raises(EngineUnavailableError, match="model file is missing"):
        render(scene, tmp_path, engine=Down())
    assert list((tmp_path / "work").iterdir()) == []


def test_nothing_marked_is_an_error_not_a_copy(scene: Scene, tmp_path: Path) -> None:
    empty = MaskDocument.model_validate({"version": 1, "regions": []})
    with pytest.raises(ProcessingError, match="nothing marked"):
        render(scene, tmp_path, document=empty)


def test_a_file_that_is_not_a_video_is_refused_plainly(videos: dict[str, Path], scene: Scene, tmp_path: Path) -> None:
    with pytest.raises(ProcessingError) as raised:
        render(scene, tmp_path, source=videos["garbage"])
    assert str(tmp_path) not in raised.value.args[0]


def test_a_destination_is_only_written_when_the_checks_pass(scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise VerificationFailed("The restored video is missing frames (3 instead of 24).")

    monkeypatch.setattr("app.video.pipeline.verify_output", fail)
    with pytest.raises(VerificationFailed):
        render(scene, tmp_path)
    assert not (tmp_path / "out" / "restored.mp4").exists()


# --- shimmer check ---------------------------------------------------------------------------------


def test_a_reconstruction_that_flickers_is_not_delivered(scene: Scene, tmp_path: Path) -> None:
    noisy = GenerativeStandIn(sigma=30.0)
    with pytest.raises(VerificationFailed, match="flickers"):
        render(scene, tmp_path, engine=noisy, settings=RenderSettings(temporal_smoothing=False))
    assert not (tmp_path / "out" / "restored.mp4").exists()


def test_the_same_engine_with_smoothing_passes(scene: Scene, tmp_path: Path) -> None:
    result = render(scene, tmp_path, engine=GenerativeStandIn(sigma=12.0), settings=RenderSettings(temporal_smoothing=True))
    assert result.verification.artifact_fraction <= 0.5
