"""Geometry of keyframes and tracked motion, checked against values worked out by hand."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.schemas.mask import Region
from app.tracking.raster import rasterize
from app.tracking.shapes import (
    IDENTITY,
    Segment,
    Shape,
    Stroke,
    blended_shape,
    segments_from_json,
    shape_at,
    shape_center,
    transform_shape,
)

FIXTURES = Path(__file__).parent / "fixtures" / "shape_cases.json"
SQUARE = 1.0  # aspect (height / width) of a square frame
TALL = 16 / 9  # a 9:16 phone video: height / width


def rect_region(*keyframes: tuple[int, tuple[float, float, float, float]]) -> Region:
    return Region.model_validate(
        {
            "id": "r",
            "name": "R",
            "type": "rectangle",
            "keyframes": [{"frame": f, "rect": dict(zip("xywh", r, strict=True))} for f, r in keyframes],
        }
    )


def poly_region(points: list[list[float]], frame: int = 0) -> Region:
    return Region.model_validate(
        {"id": "p", "name": "P", "type": "polygon", "keyframes": [{"frame": frame, "points": points}]}
    )


def close(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-9) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


class TestTransforms:
    def test_identity_returns_the_same_shape(self) -> None:
        shape = Shape(rect=(0.1, 0.2, 0.3, 0.4))
        assert transform_shape(shape, IDENTITY, TALL) is shape

    def test_rectangle_translation_in_frame_width_units(self) -> None:
        # dy is a distance in frame *widths*: on a 16:9 portrait frame 0.1 widths is 0.1/1.777 of the height.
        moved = transform_shape(Shape(rect=(0.2, 0.2, 0.2, 0.2)), (0.1, 0.1, 1.0, 0.0), TALL)
        assert moved.rect is not None
        assert close(moved.rect, (0.3, 0.2 + 0.1 / TALL, 0.2, 0.2))

    def test_rectangle_scales_about_its_centre(self) -> None:
        scaled = transform_shape(Shape(rect=(0.4, 0.4, 0.2, 0.1)), (0.0, 0.0, 2.0, 0.0), SQUARE)
        assert scaled.rect is not None
        assert close(scaled.rect, (0.3, 0.35, 0.4, 0.2))

    def test_rectangle_is_clipped_to_the_frame(self) -> None:
        clipped = transform_shape(Shape(rect=(0.8, 0.4, 0.2, 0.2)), (0.1, 0.0, 1.0, 0.0), SQUARE)
        assert clipped.rect is not None
        x, _, w, _ = clipped.rect
        assert x == pytest.approx(0.9) and w == pytest.approx(0.1)

    def test_rectangle_fully_off_screen_becomes_empty_not_negative(self) -> None:
        gone = transform_shape(Shape(rect=(0.8, 0.4, 0.2, 0.2)), (0.5, 0.0, 1.0, 0.0), SQUARE)
        assert gone.rect is not None and gone.rect[2] == 0.0

    def test_polygon_rotates_a_quarter_turn_about_its_bbox_centre(self) -> None:
        square = Shape(points=((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)))
        turned = transform_shape(square, (0.0, 0.0, 1.0, math.pi / 2), SQUARE)
        assert turned.points is not None
        # Rotating (x, y) by +90 degrees about (0.5, 0.5) in y-down coordinates: (dx, dy) -> (-dy, dx).
        expected = ((0.6, 0.4), (0.6, 0.6), (0.4, 0.6), (0.4, 0.4))
        for got, want in zip(turned.points, expected, strict=True):
            assert close(got, want)

    def test_rotation_is_not_stretched_on_a_tall_frame(self) -> None:
        # A shape that is 0.2 wide and 0.2*TALL... square in pixels must stay square in pixels after rotating.
        half = 0.1
        square = Shape(points=((0.5 - half, 0.5 - half / TALL), (0.5 + half, 0.5 - half / TALL),
                               (0.5 + half, 0.5 + half / TALL), (0.5 - half, 0.5 + half / TALL)))
        turned = transform_shape(square, (0.0, 0.0, 1.0, math.pi / 2), TALL)
        assert turned.points is not None
        xs = [p[0] for p in turned.points]
        ys = [p[1] * TALL for p in turned.points]  # back to frame-width units
        assert max(xs) - min(xs) == pytest.approx(2 * half)
        assert max(ys) - min(ys) == pytest.approx(2 * half)

    def test_brush_radius_scales_and_strokes_move_with_their_erase_strokes(self) -> None:
        shape = Shape(
            strokes=(Stroke(0.02, ((0.4, 0.5), (0.6, 0.5))),),
            erase=(Stroke(0.01, ((0.5, 0.5),)),),
        )
        out = transform_shape(shape, (0.1, 0.0, 2.0, 0.0), SQUARE)
        assert out.strokes is not None
        assert out.strokes[0].radius == pytest.approx(0.04)
        # Centre of the stroke bbox is (0.5, 0.5): points double their distance from it, then shift by 0.1.
        assert close(out.strokes[0].points[0], (0.6 - 0.2, 0.5))
        assert close(out.strokes[0].points[1], (0.6 + 0.2, 0.5))
        assert out.erase[0].radius == pytest.approx(0.02)
        assert close(out.erase[0].points[0], (0.6, 0.5))

    def test_center_ignores_brush_radius(self) -> None:
        assert shape_center(Shape(strokes=(Stroke(0.2, ((0.3, 0.3), (0.5, 0.7))),))) == (0.4, 0.5)


class TestShapeAt:
    def test_keyframe_frame_is_exact_even_inside_a_tracked_segment(self) -> None:
        region = rect_region((0, (0.1, 0.1, 0.1, 0.1)), (10, (0.5, 0.5, 0.1, 0.1)))
        segment = Segment(anchor=0, step=1, until=9, transforms=((0.2, 0.0, 1.0, 0.0),) * 9)
        assert shape_at(region, 10, [segment], SQUARE).rect == (0.5, 0.5, 0.1, 0.1)
        assert shape_at(region, 0, [segment], SQUARE).rect == (0.1, 0.1, 0.1, 0.1)

    def test_tracked_frames_use_the_segment_not_blending(self) -> None:
        region = rect_region((0, (0.1, 0.1, 0.1, 0.1)), (10, (0.9, 0.9, 0.1, 0.1)))
        transforms = tuple((0.01 * i, 0.0, 1.0, 0.0) for i in range(1, 10))
        segment = Segment(anchor=0, step=1, until=9, transforms=transforms)
        tracked = shape_at(region, 5, [segment], SQUARE).rect
        assert tracked is not None and tracked[0] == pytest.approx(0.1 + 0.05)
        assert blended_shape(region, 5).rect[0] == pytest.approx(0.5)  # type: ignore[index]

    def test_without_tracking_it_blends_like_the_editor(self) -> None:
        region = rect_region((0, (0.0, 0.0, 0.2, 0.2)), (10, (0.5, 0.0, 0.2, 0.2)))
        assert shape_at(region, 5, None, SQUARE).rect[0] == pytest.approx(0.25)  # type: ignore[index]
        assert shape_at(region, 99, None, SQUARE).rect[0] == pytest.approx(0.5)  # type: ignore[index]

    def test_lost_tracking_holds_the_last_good_transform_until_the_next_keyframe(self) -> None:
        region = rect_region((0, (0.1, 0.1, 0.1, 0.1)), (20, (0.1, 0.1, 0.1, 0.1)))
        segment = Segment(
            anchor=0, step=1, until=19, transforms=((0.1, 0.0, 1.0, 0.0), (0.2, 0.0, 1.0, 0.0)), lost_at=3, reason="x"
        )
        for frame, x in [(1, 0.2), (2, 0.3), (3, 0.3), (19, 0.3)]:
            assert shape_at(region, frame, [segment], SQUARE).rect[0] == pytest.approx(x)  # type: ignore[index]

    def test_lost_at_the_first_step_holds_the_keyframe_shape(self) -> None:
        region = rect_region((0, (0.1, 0.1, 0.1, 0.1)))
        segment = Segment(anchor=0, step=1, until=30, transforms=(), lost_at=1, reason="x")
        assert shape_at(region, 7, [segment], SQUARE).rect == (0.1, 0.1, 0.1, 0.1)

    def test_backward_segment_covers_frames_before_the_first_keyframe(self) -> None:
        region = rect_region((10, (0.5, 0.1, 0.1, 0.1)))
        segment = Segment(anchor=10, step=-1, until=0, transforms=tuple((-0.01 * i, 0.0, 1.0, 0.0) for i in range(1, 11)))
        assert segment.covers(9) and segment.covers(0) and not segment.covers(10) and not segment.covers(11)
        assert shape_at(region, 7, [segment], SQUARE).rect[0] == pytest.approx(0.5 - 0.03)  # type: ignore[index]

    def test_frames_outside_every_segment_fall_back_to_blending(self) -> None:
        region = rect_region((0, (0.1, 0.1, 0.1, 0.1)), (10, (0.3, 0.1, 0.1, 0.1)))
        segment = Segment(anchor=0, step=1, until=4, transforms=((0.5, 0.0, 1.0, 0.0),) * 4)
        assert shape_at(region, 8, [segment], SQUARE).rect[0] == pytest.approx(0.1 + 0.2 * 0.8)  # type: ignore[index]


def test_segments_round_trip_from_stored_json() -> None:
    stored = {
        "regions": {
            "r": {"segments": [{"anchor": 3, "step": -1, "until": 0, "transforms": [[0.1, 0.0, 1.0, 0.0]], "lost_at": 1, "reason": "gone"}]}
        }
    }
    segment = segments_from_json(stored)["r"][0]
    assert (segment.anchor, segment.step, segment.until, segment.lost_at, segment.reason) == (3, -1, 0, 1, "gone")
    assert segment.transforms == ((0.1, 0.0, 1.0, 0.0),)
    assert segments_from_json(None) == {}


class TestRasterize:
    def test_rectangle_fills_the_expected_pixels(self) -> None:
        mask = rasterize(Shape(rect=(0.25, 0.5, 0.5, 0.25)), 100, 80)
        assert mask.shape == (80, 100)
        assert mask[50, 50] == 255 and mask[39, 50] == 0 and mask[50, 24] == 0
        assert abs(int((mask > 0).sum()) - 50 * 20) <= 150  # edge pixels may differ by one

    def test_polygon_triangle_area(self) -> None:
        mask = rasterize(Shape(points=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))), 100, 100)
        assert abs(int((mask > 0).sum()) - 5000) < 300

    def test_brush_radius_is_a_fraction_of_the_width(self) -> None:
        mask = rasterize(Shape(strokes=(Stroke(0.1, ((0.5, 0.5),)),)), 200, 100)
        ys, xs = np.nonzero(mask)
        assert xs.max() - xs.min() == pytest.approx(40, abs=2) and ys.max() - ys.min() == pytest.approx(40, abs=2)

    def test_erase_cuts_a_hole(self) -> None:
        shape = Shape(strokes=(Stroke(0.05, ((0.2, 0.5), (0.8, 0.5))),), erase=(Stroke(0.03, ((0.5, 0.5),)),))
        mask = rasterize(shape, 200, 100)
        assert mask[50, 100] == 0 and mask[50, 60] == 255 and mask[50, 140] == 255


def test_python_matches_the_golden_cases() -> None:
    """The same cases run against the TypeScript mirror (frontend/lib/tracking.test.ts)."""
    cases = json.loads(FIXTURES.read_text())["cases"]
    assert len(cases) >= 6
    for case in cases:
        region = Region.model_validate(case["region"])
        segments = segments_from_json({"regions": {region.id: {"segments": case["segments"]}}}).get(region.id)
        shape = shape_at(region, case["frame"], segments, case["aspect"])
        got = {k: v for k, v in _shape_json(shape).items() if v}
        assert_json_close(got, case["expected"], case["name"])


def _shape_json(shape: Shape) -> dict[str, object]:
    def strokes(items: tuple[Stroke, ...]) -> list[dict[str, object]]:
        return [{"radius": s.radius, "points": [list(p) for p in s.points]} for s in items]

    return {
        "rect": list(shape.rect) if shape.rect else None,
        "points": [list(p) for p in shape.points] if shape.points else None,
        "strokes": strokes(shape.strokes) if shape.strokes else None,
        "erase": strokes(shape.erase),
    }


def assert_json_close(got: object, want: object, label: str, tol: float = 1e-9) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys(), label
        for key in want:
            assert_json_close(got[key], want[key], f"{label}.{key}", tol)
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), label
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            assert_json_close(g, w, f"{label}[{i}]", tol)
    else:
        assert got == pytest.approx(want, abs=tol), label
