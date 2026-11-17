"""Tracker accuracy and failure behaviour on synthetic H.264 videos with known ground truth.

Scores are the overlap (intersection over union) between the tracked mask and the logo's true
outline on each frame, so "good" has a number attached.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from tests.synthetic_video import Pose, SyntheticVideo, make_video

from app.schemas.mask import MaskDocument
from app.tracking.raster import rasterize
from app.tracking.shapes import segments_from_json, shape_at
from app.tracking.tracker import (
    LEFT_FRAME,
    NO_DETAIL,
    VIDEO_ENDED,
    TrackingCancelled,
    track_document,
)


def region_json(video: SyntheticVideo, frame: int, kind: str = "rectangle", region_id: str = "r") -> dict:
    outline = video.normalised_outline(frame)
    xs, ys = [p[0] for p in outline], [p[1] for p in outline]
    if kind == "rectangle":
        shape: dict = {"rect": {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}}
    elif kind == "polygon":
        shape = {"points": outline}
    else:  # brush: a few strokes through the logo
        cx, cy = sum(xs) / 4, sum(ys) / 4
        half = (max(xs) - min(xs)) * 0.3
        shape = {
            "strokes": [
                {"radius": 0.02, "points": [[cx - half, cy], [cx + half, cy]]},
                {"radius": 0.02, "points": [[cx, cy - half * 0.5], [cx, cy + half * 0.5]]},
            ]
        }
    return {"id": region_id, "name": region_id.upper(), "type": kind, "keyframes": [{"frame": frame, **shape}]}


def doc_of(*regions: dict) -> MaskDocument:
    return MaskDocument.model_validate({"version": 1, "regions": list(regions)})


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero((a > 0) | (b > 0))
    return np.count_nonzero((a > 0) & (b > 0)) / union if union else 1.0


def scores(video: SyntheticVideo, doc: MaskDocument, data: dict, region_id: str = "r") -> list[float]:
    region = next(r for r in doc.regions if r.id == region_id)
    segments = segments_from_json(data)[region_id]
    result = []
    for frame in range(video.frame_count):
        if video.outlines[frame] is None:
            continue
        shape = shape_at(region, frame, segments, data["aspect"])
        result.append(overlap(rasterize(shape, video.width, video.height), video.outline_mask(frame)))
    return result


def track(video: SyntheticVideo, doc: MaskDocument, **kwargs) -> dict:
    return track_document(doc, video.path, width=video.width, height=video.height, frame_count=video.frame_count, **kwargs)


@pytest.fixture
def make(tmp_path: Path):
    def build(name: str, **kwargs) -> SyntheticVideo:
        return make_video(tmp_path / f"{name}.mp4", **kwargs)

    return build


def linear(frame: int) -> Pose:
    return (110 + 3 * frame, 100 + 0.8 * frame, 1.0, 0.0)


class TestAccuracy:
    def test_a_logo_moving_across_the_frame(self, make) -> None:
        video = make("move", pose=linear, frames=90)
        doc = doc_of(region_json(video, 0))
        s = scores(video, doc, track(video, doc))
        assert np.mean(s) > 0.97 and min(s) > 0.9

    def test_a_logo_that_grows(self, make) -> None:
        video = make("grow", pose=lambda f: (160 + 1.5 * f, 130, 1 + 0.006 * f, 0.0), frames=90)
        doc = doc_of(region_json(video, 0))
        s = scores(video, doc, track(video, doc))
        assert np.mean(s) > 0.95 and min(s) > 0.9

    def test_a_fixed_logo_stays_put_while_the_background_pans(self, make) -> None:
        video = make("static", pose=lambda f: (240, 135, 1.0, 0.0), background_shift=lambda f: (2.5 * f, f), frames=90)
        doc = doc_of(region_json(video, 0))
        s = scores(video, doc, track(video, doc))
        assert min(s) > 0.97, "the tracker followed the background instead of the logo"

    def test_a_half_transparent_logo(self, make) -> None:
        video = make("ghost", pose=linear, opacity=0.5, frames=90)
        doc = doc_of(region_json(video, 0))
        assert np.mean(scores(video, doc, track(video, doc))) > 0.95

    def test_a_translucent_logo_over_moving_video_for_a_long_clip(self, make) -> None:
        # The background behind a see-through logo keeps changing, so frame-to-frame tracking alone
        # slips; matching the keyframe's own corners keeps it locked on (0.775 worst frame without).
        video = make(
            "ghost-long",
            pose=lambda f: (100 + 1.2 * f, 135, 1.0, 0.0),
            background_shift=lambda f: (1.5 * f, 0.0),
            opacity=0.5,
            frames=300,
        )
        doc = doc_of(region_json(video, 0))
        assert min(scores(video, doc, track(video, doc))) > 0.95

    def test_a_long_curved_path_does_not_drift(self, make) -> None:
        video = make(
            "curve",
            pose=lambda f: (240 + 120 * math.sin(f / 25), 135 + 60 * math.cos(f / 18), 1.0, 0.0),
            background_shift=lambda f: (1.2 * f, 0.0),
            frames=240,
        )
        doc = doc_of(region_json(video, 0))
        s = scores(video, doc, track(video, doc))
        assert np.mean(s) > 0.97 and min(s) > 0.9
        assert np.mean(s[-60:]) > 0.95, "overlap should not fall off late in a long clip"

    def test_a_rotating_polygon(self, make) -> None:
        video = make("spin", pose=lambda f: (240 + 0.5 * f, 135, 1.0, 0.006 * f), frames=90)
        doc = doc_of(region_json(video, 0, "polygon"))
        s = scores(video, doc, track(video, doc))
        assert np.mean(s) > 0.93 and min(s) > 0.88

    def test_a_brush_region(self, make) -> None:
        video = make("brush", pose=linear, frames=60)
        doc = doc_of(region_json(video, 0, "brush"))
        data = track(video, doc)
        region = doc.regions[0]
        # Strokes cover only part of the logo, so compare where the stroke centre ends up instead of overlap.
        shape = shape_at(region, 59, segments_from_json(data)["r"], data["aspect"])
        assert shape.strokes is not None
        xs = [p[0] * video.width for s in shape.strokes for p in s.points]
        true_centre_x = float(np.mean(video.outlines[59][:, 0]))  # type: ignore[index]
        assert abs(np.mean(xs) - true_centre_x) < 4

    def test_tracking_backwards_from_a_keyframe_in_the_middle(self, make) -> None:
        video = make("middle", pose=linear, frames=90)
        doc = doc_of(region_json(video, 60))
        data = track(video, doc)
        segments = data["regions"]["r"]["segments"]
        assert {(s["anchor"], s["step"], s["until"]) for s in segments} == {(60, 1, 89), (60, -1, 0)}
        s = scores(video, doc, data)
        assert len(s) == 90 and np.mean(s) > 0.97 and min(s) > 0.9

    def test_two_regions_are_followed_independently(self, make) -> None:
        # A second logo would need a second sprite; use two different marks on the same logo instead.
        video = make("two", pose=linear, frames=60)
        outline = video.normalised_outline(0)
        xs, ys = [p[0] for p in outline], [p[1] for p in outline]
        left = {"id": "left", "name": "L", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {
            "x": min(xs), "y": min(ys), "w": (max(xs) - min(xs)) / 2, "h": max(ys) - min(ys)}}]}
        right = {"id": "right", "name": "R", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {
            "x": (min(xs) + max(xs)) / 2, "y": min(ys), "w": (max(xs) - min(xs)) / 2, "h": max(ys) - min(ys)}}]}
        data = track(video, doc_of(left, right))
        dx = {rid: data["regions"][rid]["segments"][0]["transforms"][-1][0] * video.width for rid in ("left", "right")}
        true_dx = 3 * 59
        assert abs(dx["left"] - true_dx) < 4 and abs(dx["right"] - true_dx) < 4


class TestKeyframesAndSegments:
    def test_each_segment_stops_before_the_next_keyframe(self, make) -> None:
        video = make("two-kf", pose=linear, frames=90)
        region = region_json(video, 0)
        region["keyframes"].append(region_json(video, 40)["keyframes"][0])
        data = track(video, doc_of(region))
        segments = {(s["anchor"], s["until"]): s for s in data["regions"]["r"]["segments"]}
        assert set(segments) == {(0, 39), (40, 89)}
        assert len(segments[(0, 39)]["transforms"]) == 39 and len(segments[(40, 89)]["transforms"]) == 49

    def test_a_keyframe_on_the_last_frame_has_nothing_to_track_after_it(self, make) -> None:
        video = make("end", pose=linear, frames=30)
        region = region_json(video, 29)
        data = track(video, doc_of(region))
        assert [(s["anchor"], s["step"]) for s in data["regions"]["r"]["segments"]] == [(29, -1)]


class TestLosingTheLogo:
    def test_a_logo_that_vanishes_is_reported_on_the_frame_it_goes(self, make) -> None:
        video = make("vanish", pose=linear, visible=lambda f: f < 50, frames=90)
        doc = doc_of(region_json(video, 0))
        data = track(video, doc)
        segment = data["regions"]["r"]["segments"][0]
        assert segment["lost_at"] in (50, 51), segment
        assert segment["reason"]
        assert len(segment["transforms"]) == segment["lost_at"] - 1
        # Tracking stopped instead of inventing positions; the mask holds where it was.
        shape = shape_at(doc.regions[0], 80, segments_from_json(data)["r"], data["aspect"])
        last = shape_at(doc.regions[0], segment["lost_at"] - 1, segments_from_json(data)["r"], data["aspect"])
        assert shape == last

    def test_a_logo_that_leaves_the_frame(self, make) -> None:
        video = make("exit", pose=lambda f: (300 + 6 * f, 135, 1.0, 0.0), frames=80)
        data = track(video, doc_of(region_json(video, 0)))
        segment = data["regions"]["r"]["segments"][0]
        assert segment["lost_at"] is not None and segment["lost_at"] < 80
        assert segment["reason"] in {LEFT_FRAME} or "detail" in segment["reason"]

    def test_a_hard_cut_ends_tracking(self, make, tmp_path: Path) -> None:
        first = make("cut-a", pose=linear, frames=40, seed=1)
        # Different texture and logo position from frame 40 on, as after an edit.
        second = make("cut-b", pose=lambda f: (400 - 2 * f, 200, 1.0, 0.0), frames=40, seed=99)
        joined = tmp_path / "cut.mp4"
        listing = tmp_path / "list.txt"
        listing.write_text(f"file '{first.path}'\nfile '{second.path}'\n")
        import subprocess

        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)],
            check=True,
        )
        doc = doc_of(region_json(first, 0))
        data = track_document(doc, joined, width=first.width, height=first.height, frame_count=80)
        lost = data["regions"]["r"]["segments"][0]["lost_at"]
        assert lost is not None and 38 <= lost <= 42, lost

    def test_a_blank_area_says_there_is_not_enough_detail(self, make) -> None:
        video = make("blank", pose=linear, frames=30)
        flat = {"id": "r", "name": "R", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": 0.0, "y": 0.0, "w": 0.02, "h": 0.02}}]}
        # A tiny corner of an otherwise textured frame still has some texture, so use a truly flat video.
        import subprocess

        flat_path = Path(video.path).with_name("flat.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=480x270:r=30:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(flat_path)],
            check=True,
        )
        data = track_document(doc_of(flat), flat_path, width=480, height=270, frame_count=30)
        segment = data["regions"]["r"]["segments"][0]
        assert segment["reason"] == NO_DETAIL and segment["lost_at"] == 1 and segment["transforms"] == []

    def test_a_video_shorter_than_its_frame_count_is_reported(self, make) -> None:
        video = make("short", pose=linear, frames=30)
        data = track_document(
            doc_of(region_json(video, 0)), video.path, width=video.width, height=video.height, frame_count=45
        )
        segment = data["regions"]["r"]["segments"][0]
        assert segment["reason"] == VIDEO_ENDED and segment["lost_at"] == 30


class TestRunControl:
    def test_progress_is_reported_and_reaches_the_total(self, make) -> None:
        video = make("progress", pose=linear, frames=60)
        calls: list[tuple[int, int]] = []
        track(video, doc_of(region_json(video, 0)), progress=lambda done, total: calls.append((done, total)) or True)
        assert calls and calls[-1][0] == calls[-1][1] == 60
        assert [c[0] for c in calls] == sorted(c[0] for c in calls)

    def test_returning_false_from_progress_stops_the_run(self, make) -> None:
        video = make("cancel", pose=linear, frames=90)
        seen: list[int] = []

        def stop_after_twenty(done: int, total: int) -> bool:
            seen.append(done)
            return done < 20

        with pytest.raises(TrackingCancelled):
            track(video, doc_of(region_json(video, 0)), progress=stop_after_twenty)
        assert max(seen) < 40

    def test_the_stored_document_is_plain_json(self, make) -> None:
        import json

        video = make("json", pose=linear, frames=20)
        data = track(video, doc_of(region_json(video, 0)))
        assert json.loads(json.dumps(data)) == data
        assert data["aspect"] == pytest.approx(270 / 480) and data["frame_count"] == 20
