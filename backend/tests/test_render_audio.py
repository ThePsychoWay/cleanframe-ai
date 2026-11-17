"""The original sound goes back onto the rebuilt picture: copied untouched when the new file can hold it,
converted once when not, never lost, and never out of step with the picture."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.video.audio import AudioAction, mux_audio, plan_audio
from app.video.ffmpeg import ProcessingError
from app.video.source import inspect_source
from tests.render_media import (
    add_audio_start,
    ffmpeg,
    first_packet_time,
    make_source,
    pcm,
    stream_of,
)

SMALL = {"size": "128x72", "frames": 24}


def _picture(path: Path, container: str = "mp4", frames: int = 24) -> Path:
    """A video-only file standing in for the rebuilt picture."""
    codec = ["-c:v", "libvpx-vp9", "-b:v", "1M"] if container == "webm" else ["-c:v", "libx264"]
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=128x72:rate=30:duration={frames / 30:.4f}", "-pix_fmt", "yuv420p", *codec, str(path))
    return path


@pytest.mark.parametrize("codec", ["aac", "mp3", "libopus"])
def test_audio_the_mp4_can_hold_is_copied_bit_for_bit(tmp_path: Path, codec: str) -> None:
    source = make_source(tmp_path / f"in_{codec}.mp4" if codec != "libopus" else tmp_path / "in.mkv", audio=codec, **SMALL)
    media = inspect_source(source)
    out = tmp_path / "out.mp4"
    result = mux_audio(_picture(tmp_path / "pic.mp4"), source, out, media, container="mp4", keep_audio=True)
    assert result.action is AudioAction.COPIED and result.codec == media.audio[0].codec
    assert stream_of(out, "audio")["codec_name"] == media.audio[0].codec
    restored, original = pcm(out), pcm(source)
    # Decoded samples are identical, so nothing was re-encoded. Opus carried in MP4 can have a few milliseconds
    # more tail than the same stream in Matroska (MP4 has no exact end trim), which is the only allowed difference.
    shared = min(len(restored), len(original))
    assert restored[:shared] == original[:shared]
    assert abs(len(restored) - len(original)) <= 48_000 * 2 * 0.05


def test_audio_the_container_cannot_hold_is_converted_once(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in.mov", audio="pcm_s16le", **SMALL)
    media = inspect_source(source)
    out = tmp_path / "out.mp4"
    result = mux_audio(_picture(tmp_path / "pic.mp4"), source, out, media, container="mp4", keep_audio=True)
    assert result.action is AudioAction.CONVERTED and result.codec == "aac"
    assert stream_of(out, "audio")["codec_name"] == "aac"


def test_webm_keeps_opus_and_converts_aac(tmp_path: Path) -> None:
    opus = make_source(tmp_path / "opus.webm", audio="libopus", video="libvpx-vp9", **SMALL)
    out = tmp_path / "a.webm"
    kept = mux_audio(_picture(tmp_path / "p1.webm", "webm"), opus, out, inspect_source(opus), container="webm", keep_audio=True)
    assert kept.action is AudioAction.COPIED and pcm(out) == pcm(opus)

    aac = make_source(tmp_path / "aac.mp4", audio="aac", **SMALL)
    out2 = tmp_path / "b.webm"
    converted = mux_audio(_picture(tmp_path / "p2.webm", "webm"), aac, out2, inspect_source(aac), container="webm", keep_audio=True)
    assert converted.action is AudioAction.CONVERTED and stream_of(out2, "audio")["codec_name"] == "opus"


def test_a_silent_original_stays_silent(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in.mp4", **SMALL)
    out = tmp_path / "out.mp4"
    result = mux_audio(_picture(tmp_path / "pic.mp4"), source, out, inspect_source(source), container="mp4", keep_audio=True)
    assert result.action is AudioAction.NONE and result.codec is None
    with pytest.raises(StopIteration):
        stream_of(out, "audio")


def test_audio_is_dropped_only_when_the_person_asked(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in.mp4", audio="aac", **SMALL)
    out = tmp_path / "out.mp4"
    result = mux_audio(_picture(tmp_path / "pic.mp4"), source, out, inspect_source(source), container="mp4", keep_audio=False)
    assert result.action is AudioAction.REMOVED
    with pytest.raises(StopIteration):
        stream_of(out, "audio")


def test_the_plan_matches_what_is_done(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in.mp4", audio="aac", **SMALL)
    media = inspect_source(source)
    assert plan_audio(media, "mp4", True)[0] is AudioAction.COPIED
    assert plan_audio(media, "webm", True) == (AudioAction.CONVERTED, "opus")
    assert plan_audio(media, "mp4", False)[0] is AudioAction.REMOVED


@pytest.mark.parametrize("late_by", [0.0, 0.5])
def test_the_sound_keeps_its_place_relative_to_the_picture(tmp_path: Path, late_by: float) -> None:
    """If the original's sound starts later than its picture, it must still start later by the same amount."""
    base = make_source(tmp_path / "base.mp4", audio="aac", **SMALL)
    source = add_audio_start(base, tmp_path / "late.mp4", late_by) if late_by else base
    media = inspect_source(source)
    out = tmp_path / "out.mp4"
    mux_audio(_picture(tmp_path / "pic.mp4"), source, out, media, container="mp4", keep_audio=True)

    original_gap = first_packet_time(source, "audio") - first_packet_time(source, "video")
    restored_gap = first_packet_time(out, "audio") - first_packet_time(out, "video")
    assert restored_gap == pytest.approx(original_gap, abs=0.005)


def test_a_failed_mux_is_reported_plainly(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in.mp4", audio="aac", **SMALL)
    media = inspect_source(source)
    with pytest.raises(ProcessingError) as raised:
        mux_audio(tmp_path / "missing.mp4", source, tmp_path / "out.mp4", media, container="mp4", keep_audio=True)
    assert "audio" in raised.value.args[0].lower() or "couldn't" in raised.value.args[0].lower()
    assert str(tmp_path) not in raised.value.args[0]  # no server paths in messages shown to people
