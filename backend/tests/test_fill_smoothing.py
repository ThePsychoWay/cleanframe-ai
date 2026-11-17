"""Generated patches: steady over time, and continuous with the picture around them.

A generative fill is a new guess on every frame, so on a clip where nothing behind the mark can be borrowed from other
frames it shimmers, and its outline shows as a faint box when its tone differs a little from its surroundings. These tests
use a stand-in generator that is deliberately unsteady and a little off-tone, and measure what comes out.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from app.ai.base import InpaintingOptions
from app.ai.consistency import match_seam
from app.ai.image import ImageInpaintingEngine
from app.ai.temporal import TemporalReconstructionEngine
from tests.test_temporal_engine import (
    WINDOW_H,
    WINDOW_W,
    make_scene,
    mark_mask,
    noisy,
    watermark,
)


class _UnsteadyFill(ImageInpaintingEngine):
    """Fills the hole with a texture that is different on every call, and 14 levels darker than the picture around it."""

    provider = "unsteady-test"
    name = "Unsteady fill"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(11)

    def is_available(self) -> bool:
        return True

    def inpaint_image(self, crop, mask, options):
        out = crop.copy()
        base = crop[mask == 0].mean(axis=0) - 14.0
        texture = cv2.GaussianBlur(self.rng.normal(0, 1, crop.shape[:2]).astype(np.float32), (0, 0), 2.0) * 40.0
        out[mask > 0] = np.clip(base + texture[mask > 0][:, None], 0, 255).astype(np.uint8)
        return out


def _still_clip(count: int = 12, seed: int = 3):
    scene = make_scene()
    window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 : 100 + WINDOW_W])
    rng = np.random.default_rng(seed)
    marked = [watermark(noisy(window, rng)) for _ in range(count)]
    return marked, [mark_mask() for _ in range(count)]


def _run(engine, frames, masks, **options):
    return list(engine.process(iter(frames), iter(masks), InpaintingOptions(quality="high", **options)))


def _change_inside(frames, mask) -> float:
    inside = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    return float(np.mean([np.abs(b[inside].astype(np.float32) - a[inside].astype(np.float32)).mean() for a, b in itertools.pairwise(frames)]))


def test_a_generated_patch_changes_much_less_from_frame_to_frame_when_smoothing_is_on() -> None:
    marked, masks = _still_clip()
    off_engine = TemporalReconstructionEngine(fallback=_UnsteadyFill(), fallback_label="Stand-in")
    on_engine = TemporalReconstructionEngine(fallback=_UnsteadyFill(), fallback_label="Stand-in")
    off = _run(off_engine, marked, masks, temporal_smoothing=False)
    on = _run(on_engine, marked, masks, temporal_smoothing=True)

    assert _change_inside(on, masks[0]) < 0.6 * _change_inside(off, masks[0])
    assert on_engine._stats.smoothed_frames == len(marked) - 1  # every frame but the first had a previous patch to follow
    assert any("blended with the previous frame's" in note for note in on_engine.notes())
    assert not any("blended with the previous frame's" in note for note in off_engine.notes())


def test_smoothing_changes_nothing_outside_the_patch() -> None:
    marked, masks = _still_clip(count=6)
    on = _run(TemporalReconstructionEngine(fallback=_UnsteadyFill()), marked, masks, temporal_smoothing=True)
    far = cv2.dilate(masks[0], np.ones((31, 31), np.uint8)) == 0
    for before, after in zip(marked, on, strict=True):
        assert np.array_equal(before[far], after[far])


def test_a_cut_stops_the_previous_patch_from_being_carried_into_the_new_shot() -> None:
    first, masks = _still_clip(count=6, seed=3)
    other_scene = make_scene(seed=99)
    other = np.ascontiguousarray(other_scene[40 : 40 + WINDOW_H, 300 : 300 + WINDOW_W])
    rng = np.random.default_rng(8)
    second = [watermark(noisy(other, rng)) for _ in range(6)]
    frames, all_masks = first + second, masks + [mark_mask() for _ in range(6)]

    on_engine = TemporalReconstructionEngine(fallback=_UnsteadyFill())
    on = _run(on_engine, frames, all_masks, temporal_smoothing=True)
    off = _run(TemporalReconstructionEngine(fallback=_UnsteadyFill()), frames, all_masks, temporal_smoothing=False)

    # Same generator, same call order: the first frame of the new shot must be what it would be without smoothing.
    assert np.array_equal(on[6], off[6])
    assert on_engine._stats.smoothed_frames < len(frames) - 1


def test_a_patch_is_made_continuous_with_the_picture_around_it() -> None:
    height, width = 120, 160
    ramp = np.tile(np.linspace(60, 200, width, dtype=np.float32), (height, 1))
    source = np.dstack([ramp, ramp, ramp]).astype(np.uint8)
    grown = np.zeros((height, width), np.uint8)
    grown[40:80, 60:110] = 255
    filled = source.copy()
    filled[grown > 0] = (30, 30, 30)  # a patch much darker than the ramp it sits in

    result = match_seam(filled, source, grown)

    edge = (grown > 0) & ~(cv2.erode(grown, np.ones((3, 3), np.uint8)) > 0)
    outside = (cv2.dilate(grown, np.ones((3, 3), np.uint8)) > 0) & ~(grown > 0)
    before = abs(float(filled[edge].mean()) - float(source[outside].mean()))
    after = abs(float(result[edge].mean()) - float(source[outside].mean()))
    assert before > 60
    assert after < 12  # the edge now meets the real pixels
    assert np.array_equal(result[grown == 0], filled[grown == 0])  # only the patch is touched


def test_a_patch_at_the_edge_of_the_crop_falls_back_to_a_plain_colour_match() -> None:
    height, width = 80, 80
    source = np.full((height, width, 3), 120, np.uint8)
    grown = np.zeros((height, width), np.uint8)
    grown[0:20, 10:40] = 255  # touches the top of the crop: no real pixels above it
    filled = source.copy()
    filled[grown > 0] = (60, 60, 60)

    result = match_seam(filled, source, grown)

    assert result.shape == filled.shape and result.dtype == np.uint8
    assert np.array_equal(result[grown == 0], filled[grown == 0])


def _previous_patch(output: np.ndarray, generated: np.ndarray):
    from app.ai.roi import Roi
    from app.ai.temporal import _Previous

    return _Previous(index=0, roi=Roi(0, 0, output.shape[1], output.shape[0]), output=output.copy(), generated=generated.copy())


def test_the_previous_patch_is_not_blended_in_when_the_picture_around_it_has_changed() -> None:
    from app.ai.roi import Roi

    height, width = 90, 120
    picture = np.full((height, width, 3), 100, np.uint8)
    generated = np.zeros((height, width), np.uint8)
    generated[30:60, 40:80] = 255
    ring = (cv2.dilate(generated, np.ones((25, 25), np.uint8)) > 0) & ~(generated > 0)
    roi = Roi(0, 0, width, height)

    # Same surroundings as last frame: blended, with a weight that fades towards the patch's edge.
    current = picture.copy()
    current[generated > 0] = (180, 180, 180)
    previous = _previous_patch(np.where((generated > 0)[..., None], 120, picture).astype(np.uint8), generated)
    assert TemporalReconstructionEngine._blend_previous(current, roi, generated > 0, ring, previous, np.eye(3))
    centre = current[45, 60, 0]
    assert 120 < centre < 180  # pulled towards last frame's patch, not replaced by it
    assert current[30, 40, 0] > 170 and current[30, 40, 0] > centre  # the edge pixel stays close to the new patch

    # The picture around the patch is now much brighter than last frame's: last frame's patch no longer fits.
    brighter = np.full((height, width, 3), 170, np.uint8)
    brighter[generated > 0] = (180, 180, 180)
    untouched = brighter.copy()
    assert not TemporalReconstructionEngine._blend_previous(brighter, roi, generated > 0, ring, previous, np.eye(3))
    assert np.array_equal(brighter, untouched)
