import re
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.storage.s3 import S3Storage
from app.video.probe import VideoRejected, parse_video_info, run_ffprobe

LIMITS = {"max_duration": 120.0, "max_pixels": 4096 * 2304}


def probe(**video: Any) -> dict[str, Any]:
    stream = {
        "codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920,
        "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "duration": "10.0", "nb_frames": "300",
    }
    stream.update(video)
    return {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.0", "bit_rate": "4000000"},
        "streams": [stream, {"codec_type": "audio", "codec_name": "aac"}],
    }


def parse(data: dict[str, Any], ext: str = ".mp4", **limits: float) -> Any:
    return parse_video_info(data, ext, **{**LIMITS, **limits})


def test_reads_the_basics() -> None:
    info = parse(probe())
    assert (info.width, info.height, info.fps, info.duration, info.frame_count) == (1080, 1920, 30.0, 10.0, 300)
    assert (info.codec, info.audio_codec, info.bitrate) == ("h264", "aac", 4_000_000)


def test_rotation_metadata_swaps_the_displayed_dimensions() -> None:
    # A phone filmed upright but stored sideways: 1920x1080 with a 90 degree display matrix.
    sideways = parse(probe(width=1920, height=1080, side_data_list=[{"rotation": -90}]))
    assert (sideways.width, sideways.height) == (1080, 1920)
    legacy_tag = parse(probe(width=1920, height=1080, tags={"rotate": "90"}))
    assert (legacy_tag.width, legacy_tag.height) == (1080, 1920)
    upside_down = parse(probe(width=1920, height=1080, side_data_list=[{"rotation": 180}]))
    assert (upside_down.width, upside_down.height) == (1920, 1080)


def test_ntsc_frame_rate_and_missing_stream_duration() -> None:
    data = probe(avg_frame_rate="30000/1001", r_frame_rate="30000/1001")
    del data["streams"][0]["duration"]
    del data["streams"][0]["nb_frames"]
    info = parse(data)
    assert info.fps == pytest.approx(29.97, abs=0.001)
    assert info.duration == 10.0  # falls back to the container's duration
    assert info.frame_count == round(10.0 * 29.97002997)


def test_cover_art_is_not_mistaken_for_the_video_track() -> None:
    data = probe()
    cover = {"codec_type": "video", "codec_name": "mjpeg", "width": 500, "height": 500, "disposition": {"attached_pic": 1}}
    data["streams"].insert(0, cover)
    assert parse(data).codec == "h264"


@pytest.mark.parametrize(
    ("data", "ext", "limits", "expected"),
    [
        (probe(), ".flv", {}, "Unsupported video format"),
        (probe(), ".mkv", {}, "don't match its .mkv extension"),
        (probe(), ".webm", {}, "don't match its .webm extension"),
        (probe(codec_name="gif"), ".mp4", {}, "uses the gif codec"),
        (probe(width=0), ".mp4", {}, "couldn't read this video"),
        (probe(avg_frame_rate="0/0", r_frame_rate="0/0"), ".mp4", {}, "frame rate couldn't be read"),
        (probe(avg_frame_rate="240/1"), ".mp4", {}, "240 fps"),
        (probe(duration="0"), ".mp4", {"max_duration": 120}, "length couldn't be read"),
        (probe(duration="130"), ".mp4", {}, "2:10 long"),
        (probe(width=7680, height=4320), ".mp4", {}, "7680×4320"),
    ],
)
def test_rejections_explain_what_to_do(data: dict[str, Any], ext: str, limits: dict[str, float], expected: str) -> None:
    if "duration" in data["streams"][0] and data["streams"][0]["duration"] == "0":
        del data["format"]["duration"]
    with pytest.raises(VideoRejected, match=re.escape(expected)):
        parse(data, ext, **limits)


def test_codec_names_from_the_file_are_sanitised() -> None:
    with pytest.raises(VideoRejected) as caught:
        parse(probe(codec_name="h264; rm -rf /"))
    assert ";" not in caught.value.message and "/" not in caught.value.message


def test_real_ffprobe_reads_generated_video_and_rejects_garbage(videos: dict[str, Path]) -> None:
    info = parse(run_ffprobe(videos["small"]))
    assert (info.width, info.height, info.codec, info.audio_codec) == (320, 568, "h264", "aac")
    with pytest.raises(VideoRejected, match="corrupted"):
        run_ffprobe(videos["garbage"])


def test_ffprobe_will_not_follow_urls_or_other_protocols(tmp_path: Path) -> None:
    # A playlist pointing at a remote URL: with only the file protocol allowed it must not be fetched.
    playlist = tmp_path / "evil.mp4"
    playlist.write_text("#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1,\nhttp://169.254.169.254/latest/meta-data/\n")
    with pytest.raises(VideoRejected):
        run_ffprobe(playlist)


def test_presigned_urls_use_the_address_the_browser_can_reach() -> None:
    settings = Settings(
        storage_endpoint="http://minio:9000",
        storage_public_endpoint="http://localhost:9000",
        storage_access_key="key",
        storage_secret_key="secret",
        storage_bucket="cleanframe",
    )
    url = S3Storage(settings).presign_part("uploads/u/p/original.mp4", "upload-1", 3)
    assert url.startswith("http://localhost:9000/cleanframe/uploads/u/p/original.mp4?")
    assert "partNumber=3" in url and "uploadId=upload-1" in url


def test_avi_is_recognised_from_its_bytes_and_accepted() -> None:
    from app.video.probe import sniff_head

    assert sniff_head(b"RIFF\x00\x00\x00\x00AVI ") == "avi"
    assert sniff_head(b"RIFF\x00\x00\x00\x00WAVE") is None  # audio, not video
    data = probe()
    data["format"]["format_name"] = "avi"
    assert parse(data, ext=".avi").container == "avi"


@pytest.mark.parametrize(("width", "height"), [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (4096, 2160)])
def test_every_resolution_up_to_4k_is_accepted_and_reported_as_it_is(width: int, height: int) -> None:
    info = parse(probe(width=width, height=height))
    assert (info.width, info.height) == (width, height)
