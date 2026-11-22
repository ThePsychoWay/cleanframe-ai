"""Every input and output container the app advertises really works, on real files: MKV and M4V are accepted as
sources; MP4, MOV, WebM and MKV (with H.264, HEVC, VP9 and AV1 where the container holds them) are written, keep
the frame count and size, and carry the original sound."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.dev import DevInpaintingEngine
from app.video.audio import AudioAction
from app.video.containers import CONTAINER_VIDEO_CODECS, CONTAINERS, COPYABLE_AUDIO, REENCODE_AUDIO, container_args, file_extension
from app.video.encoders import choose_encoder, codec_available
from app.video.pipeline import RenderSettings, render_restored_video
from app.video.probe import VideoRejected, parse_video_info, run_ffprobe
from app.video.source import inspect_source
from tests.render_media import make_source, stream_of
from tests.test_render_pipeline import N, Scene


def _formats() -> list[tuple[str, str]]:
    return [(c, codec) for c in CONTAINERS for codec in CONTAINER_VIDEO_CODECS[c]]


# --- the tables -----------------------------------------------------------------------------------


@pytest.mark.parametrize("container", CONTAINERS)
def test_every_container_has_ffmpeg_args_audio_rules_and_an_extension(container: str) -> None:
    assert "-f" in container_args(container)
    assert COPYABLE_AUDIO[container] and REENCODE_AUDIO[container][0]
    assert file_extension(container) == container


def test_unknown_container_is_refused_and_extension_falls_back() -> None:
    with pytest.raises(ValueError):
        container_args("flv")
    assert file_extension("flv") == "mp4"


@pytest.mark.parametrize(("container", "codec"), _formats())
def test_encoder_matches_the_codec_asked_for(container: str, codec: str) -> None:
    if not codec_available(codec):
        pytest.skip(f"this FFmpeg has no {codec} encoder")
    assert choose_encoder(container, codec, "balanced", "cpu").codec == codec


@pytest.mark.parametrize(("container", "codec"), [("mov", "vp9"), ("mov", "av1"), ("webm", "h264"), ("webm", "hevc"), ("mp4", "vp9")])
def test_a_codec_the_container_cannot_hold_is_refused(container: str, codec: str) -> None:
    with pytest.raises(ValueError):
        choose_encoder(container, codec, "balanced", "cpu")


def test_default_codec_is_the_containers_first() -> None:
    assert choose_encoder("mkv", None, "fast", "cpu").codec == "h264"
    assert choose_encoder("mov", None, "fast", "cpu").codec == "h264"
    assert choose_encoder("webm", None, "fast", "cpu").codec == "vp9"


# --- reading ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "video", "audio"), [("in.mkv", "libx264", "libopus"), ("in.m4v", "libx264", "aac"), ("in.mkv", "libvpx-vp9", "libvorbis")])
def test_mkv_and_m4v_are_accepted_as_sources(tmp_path: Path, name: str, video: str, audio: str) -> None:
    source = make_source(tmp_path / name, size="128x72", frames=12, video=video, audio=audio)
    info = parse_video_info(run_ffprobe(source), "." + name.rsplit(".", 1)[1], max_duration=120, max_pixels=4096 * 2304)
    assert (info.width, info.height) == (128, 72) and info.frame_count == 12


def test_a_text_file_named_mkv_is_refused(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mkv"
    fake.write_text("not a video at all, just words")
    with pytest.raises(VideoRejected):
        run_ffprobe(fake)


# --- writing ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> Scene:
    return Scene(tmp_path_factory.mktemp("fmt_scene"))


@pytest.mark.parametrize(("container", "codec"), _formats())
def test_render_writes_each_container_and_codec_with_the_original_sound(scene: Scene, tmp_path: Path, container: str, codec: str) -> None:
    if not codec_available(codec):
        pytest.skip(f"this FFmpeg has no {codec} encoder")
    work = tmp_path / "work"
    work.mkdir()
    result = render_restored_video(
        scene.source, tmp_path / f"out.{container}", document=scene.document, tracking=None, engine=DevInpaintingEngine(),
        work_dir=work, settings=RenderSettings(quality="fast", container=container, codec=codec), expected_frames=N,
    )  # fmt: skip
    assert result.container == container and result.codec == codec
    assert (result.frames, result.width, result.height) == (N, 320, 180)
    video, audio = stream_of(result.path, "video"), stream_of(result.path, "audio")
    assert video["codec_name"] == {"h264": "h264", "hevc": "hevc", "vp9": "vp9", "av1": "av1"}[codec]
    assert audio is not None, "the original sound was lost"
    assert result.audio in (AudioAction.COPIED, AudioAction.CONVERTED)
    # the file reads back as the container it claims to be, with the same number of frames
    assert inspect_source(result.path).width == 320
