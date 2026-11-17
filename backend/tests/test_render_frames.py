"""Reading frames out of the original and writing them back: the part where size, colour, timing and
rotation can silently go wrong."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.video.ffmpeg import ProcessingError
from app.video.frames import EncodeSettings, FrameReader, FrameWriter, output_size
from app.video.proxy import build_proxy
from app.video.source import inspect_source
from tests.render_media import child_processes, ffmpeg, make_source, probe, stream_of

TAGS_709 = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv")
TAGS_601 = ("-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m", "-color_range", "tv")


def read_all(path: Path) -> list[np.ndarray]:
    media = inspect_source(path)
    with FrameReader(path, media) as reader:
        return list(reader.frames())


def round_trip(source: Path, destination: Path, **settings: object) -> list[np.ndarray]:
    media = inspect_source(source)
    with FrameReader(source, media) as reader:
        frames = list(reader.frames())
        width, height = reader.width, reader.height
    with FrameWriter(
        destination, width=width, height=height, frame_rate=media.frame_rate, color=media.color,
        settings=EncodeSettings(quality="high", **settings),  # type: ignore[arg-type]
    ) as writer:  # fmt: skip
        for frame in frames:
            writer.write(frame)
        writer.finish()
    return frames


# --- reading ---------------------------------------------------------------------------------------


def test_frames_come_out_as_bgr_in_the_videos_size_and_count(tmp_path: Path) -> None:
    path = make_source(tmp_path / "a.mp4", size="192x108", frames=30)
    frames = read_all(path)
    assert len(frames) == 30
    assert all(f.shape == (108, 192, 3) and f.dtype == np.uint8 and f.flags.writeable for f in frames)
    assert frames[0].std() > 10  # a real picture, not blank


def test_frame_n_is_the_same_frame_the_editor_shows(tmp_path: Path, videos: dict[str, Path]) -> None:
    """The preview is what people mark up; the render must count frames exactly as it does, including for
    variable-frame-rate originals (90 stored frames over 3.3 s that the preview plays as 100)."""
    for name in ("small", "vfr"):
        source = videos[name]
        proxy = build_proxy(source, tmp_path / f"{name}_proxy.mp4")
        assert len(read_all(source)) == proxy.frame_count
    assert proxy.frame_count == 100  # the vfr clip


def test_rotation_is_applied_so_frames_are_upright(tmp_path: Path) -> None:
    flat = make_source(tmp_path / "flat.mp4", size="160x90", frames=10)
    turned = tmp_path / "turned.mp4"
    ffmpeg("-display_rotation", "90", "-i", str(flat), "-c", "copy", str(turned))
    media = inspect_source(turned)
    assert (media.width, media.height) == (90, 160)
    frames = read_all(turned)
    assert frames[0].shape == (160, 90, 3)


def test_an_odd_edge_is_trimmed_by_one_pixel_and_the_rest_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "odd.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=0.5", "-vf", "scale=161:91,format=yuv444p", "-c:v", "libx264", "-pix_fmt", "yuv444p", str(path))
    media = inspect_source(path)
    assert (media.width, media.height) == (161, 91)
    frames = read_all(path)
    assert frames[0].shape == (90, 160, 3)
    # same pixels as an uncropped decode of the same stream
    raw = np.frombuffer(
        __import__("subprocess").run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-vf", "scale=in_range=tv:out_range=pc,format=bgr24", "-f", "rawvideo", "-"],
            capture_output=True, check=True,
        ).stdout, np.uint8,
    ).reshape(91, 161, 3)  # fmt: skip
    assert np.abs(raw[:90, :160].astype(int) - frames[0].astype(int)).mean() < 3.0


def test_ten_bit_video_is_read_and_flagged(tmp_path: Path) -> None:
    path = make_source(tmp_path / "ten.mp4", video="libx265", pix_fmt="yuv420p10le", extra_video=("-tag:v", "hvc1"))
    media = inspect_source(path)
    assert media.bit_depth == 10
    assert len(read_all(path)) == 30


def test_hdr_is_refused_instead_of_being_washed_out(tmp_path: Path) -> None:
    path = make_source(
        tmp_path / "hdr.mp4", video="libx265", pix_fmt="yuv420p10le",
        extra_video=("-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-tag:v", "hvc1"),
    )  # fmt: skip
    with pytest.raises(ProcessingError, match="HDR"):
        inspect_source(path)


def test_a_cut_off_file_is_noticed(tmp_path: Path) -> None:
    whole = make_source(tmp_path / "whole.mp4", frames=60, extra_video=("-g", "10"))
    # an MP4 keeps its index at the end and will not open when cut, so use a layout that is readable as it arrives
    streamable = tmp_path / "stream.mp4"
    ffmpeg("-i", str(whole), "-c", "copy", "-movflags", "frag_keyframe+empty_moov", str(streamable))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(streamable.read_bytes()[: int(streamable.stat().st_size * 0.6)])
    media = inspect_source(broken)
    with FrameReader(broken, media) as reader:
        frames = list(reader.frames())
    assert 0 < len(frames) < 60  # fewer frames than the original, which the pipeline compares with the timeline


# --- colour ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "size", "tags", "expected_matrix"),
    [
        ("hd_tagged_709", "320x180", TAGS_709, "bt709"),
        ("sd_tagged_601", "320x240", TAGS_601, "smpte170m"),
        ("hd_untagged", "1280x720", (), "bt709"),
        ("sd_untagged", "320x240", (), "smpte170m"),
    ],
)
def test_colour_survives_reading_and_writing(tmp_path: Path, name: str, size: str, tags: tuple[str, ...], expected_matrix: str) -> None:
    source = make_source(tmp_path / f"{name}.mp4", size=size, frames=8, extra_video=tags)
    frames = round_trip(source, tmp_path / "out.mp4")
    media = inspect_source(source)
    assert media.color.matrix == expected_matrix
    back = read_all(tmp_path / "out.mp4")
    assert len(back) == len(frames)
    difference = np.stack([a.astype(np.int16) - b.astype(np.int16) for a, b in zip(frames, back, strict=True)])
    assert np.abs(difference).mean() < 2.5  # codec noise only
    assert abs(difference.mean()) < 1.5  # and no overall brightening/darkening/tint
    # Per channel too. A wrong colour matrix moves a channel by 5-15 levels on this saturated pattern; the ~1-1.6
    # measured here is chroma subsampling at its hard edges plus one level lost to 8-bit rounding.
    assert np.abs(difference.mean(axis=(0, 1, 2))).max() < 2.0
    out = stream_of(tmp_path / "out.mp4", "video")
    assert out["color_space"] == media.color.matrix
    assert out["color_range"] == media.color.range


def test_full_range_sources_stay_full_range(tmp_path: Path) -> None:
    source = make_source(tmp_path / "pc.mp4", size="320x180", frames=8, pix_fmt="yuvj420p", extra_video=("-color_range", "pc", "-colorspace", "bt709"))
    media = inspect_source(source)
    assert media.color.range == "pc"
    frames = round_trip(source, tmp_path / "out.mp4")
    back = read_all(tmp_path / "out.mp4")
    assert stream_of(tmp_path / "out.mp4", "video")["color_range"] == "pc"
    assert np.abs(np.stack(frames).astype(int) - np.stack(back).astype(int)).mean() < 2.5


# --- writing ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "cap", "expected"),
    [
        ((1920, 1080), None, (1920, 1080)),
        ((1920, 1080), 720, (1280, 720)),
        ((1280, 720), 1080, (1280, 720)),  # never enlarged
        ((640, 360), 720, (640, 360)),
        ((1080, 1920), 720, (720, 1280)),  # portrait: the shorter side names the resolution, so this is 720p
        ((3840, 2160), 1440, (2560, 1440)),
        ((3840, 2160), 1080, (1920, 1080)),
        ((2160, 3840), 1080, (1080, 1920)),
        ((2560, 1080), 720, (1706, 720)),  # ultrawide: 21:9 stays 21:9
        ((161, 91), None, (160, 90)),
    ],
)
def test_output_size_never_enlarges_and_stays_even(size: tuple[int, int], cap: int | None, expected: tuple[int, int]) -> None:
    assert output_size(*size, cap) == expected


def test_the_height_cap_scales_down_with_the_same_picture(tmp_path: Path) -> None:
    source = make_source(tmp_path / "big.mp4", size="640x360", frames=6)
    media = inspect_source(source)
    with FrameReader(source, media) as reader:
        frames = list(reader.frames())
    out = tmp_path / "small.mp4"
    with FrameWriter(out, width=640, height=360, frame_rate=media.frame_rate, color=media.color, settings=EncodeSettings(output_height=720)) as w:
        for frame in frames:
            w.write(frame)
        w.finish()
    assert stream_of(out, "video")["width"] == 640  # a 360p source is not enlarged to 720p

    with FrameWriter(tmp_path / "s2.mp4", width=640, height=360, frame_rate=media.frame_rate, color=media.color, settings=EncodeSettings()) as w:
        assert (w.out_width, w.out_height) == (640, 360)
        w.abort()


def test_webm_output_is_vp9(tmp_path: Path) -> None:
    source = make_source(tmp_path / "a.mp4", frames=12)
    frames = round_trip(source, tmp_path / "out.webm", container="webm")
    info = stream_of(tmp_path / "out.webm", "video")
    assert info["codec_name"] == "vp9"
    assert len(read_all(tmp_path / "out.webm")) == len(frames)


def test_better_quality_makes_bigger_files(tmp_path: Path) -> None:
    source = make_source(tmp_path / "a.mp4", size="320x180", frames=20)
    media = inspect_source(source)
    with FrameReader(source, media) as reader:
        frames = list(reader.frames())
    sizes = {}
    for quality in ("fast", "balanced", "high"):
        out = tmp_path / f"{quality}.mp4"
        with FrameWriter(out, width=320, height=180, frame_rate=media.frame_rate, color=media.color, settings=EncodeSettings(quality=quality)) as w:
            for frame in frames:
                w.write(frame)
            w.finish()
        sizes[quality] = out.stat().st_size
    assert sizes["fast"] < sizes["balanced"] < sizes["high"]


def test_the_frame_rate_is_exact_for_ntsc_rates(tmp_path: Path) -> None:
    source = make_source(tmp_path / "ntsc.mp4", rate="30000/1001", frames=30)
    media = inspect_source(source)
    assert media.frame_rate == "30000/1001"
    round_trip(source, tmp_path / "out.mp4")
    assert stream_of(tmp_path / "out.mp4", "video")["r_frame_rate"] == "30000/1001"
    info = probe(tmp_path / "out.mp4", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames")
    assert int(info["streams"][0]["nb_read_frames"]) == 30


def test_a_frame_of_the_wrong_size_is_refused_and_leaves_no_process_behind(tmp_path: Path) -> None:
    media = inspect_source(make_source(tmp_path / "a.mp4", size="128x72", frames=4))
    before = set(child_processes())
    with FrameWriter(tmp_path / "out.mp4", width=128, height=72, frame_rate=media.frame_rate, color=media.color, settings=EncodeSettings()) as writer:
        with pytest.raises(ProcessingError, match="wrong size"):
            writer.write(np.zeros((70, 128, 3), np.uint8))
        with pytest.raises(ProcessingError, match="wrong size"):
            writer.write(np.zeros((72, 128, 3), np.float32))
    assert set(child_processes()) <= before


def test_a_dead_encoder_is_reported_not_hung(tmp_path: Path) -> None:
    media = inspect_source(make_source(tmp_path / "a.mp4", size="128x72", frames=4))
    writer = FrameWriter(tmp_path / "out.mp4", width=128, height=72, frame_rate=media.frame_rate, color=media.color, settings=EncodeSettings())
    try:
        writer._ffmpeg.process.kill()  # noqa: SLF001
        writer._ffmpeg.process.wait()  # noqa: SLF001
        with pytest.raises(ProcessingError, match="couldn't be encoded"):
            for _ in range(200):  # the pipe buffer takes a few frames to notice
                writer.write(np.zeros((72, 128, 3), np.uint8))
            writer.finish()
    finally:
        writer.close()


def test_closing_the_reader_early_stops_the_decoder(tmp_path: Path) -> None:
    path = make_source(tmp_path / "long.mp4", size="320x180", frames=120)
    media = inspect_source(path)
    before = set(child_processes())
    with FrameReader(path, media) as reader:
        iterator = reader.frames()
        next(iterator)
    assert set(child_processes()) <= before
