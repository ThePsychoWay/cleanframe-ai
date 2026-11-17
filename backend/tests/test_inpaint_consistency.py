"""Colour matching and temporal stability, measured against known ground truth."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.base import InpaintingOptions
from app.ai.consistency import TemporalStabilizer, flicker, match_colour
from app.ai.dev import DevInpaintingEngine
from app.ai.image import ImageInpaintingEngine
from app.ai.roi import prepare_mask
from tests.inpaint_scene import make_scene, mean_error

ON = InpaintingOptions(temporal_smoothing=True)
OFF = InpaintingOptions(temporal_smoothing=False)


class GenerativeStandIn(ImageInpaintingEngine):
    """Stands in for a per-frame generative model: a sensible reconstruction plus a different texture every
    frame, which is what makes such models shimmer. Counting calls gives each frame its own noise."""

    provider = "dev"
    name = "generative stand-in"

    def __init__(self, sigma: float = 12.0, bias: int = 0) -> None:
        self.sigma, self.bias, self.calls = sigma, bias, 0

    def is_available(self) -> bool:
        return True

    def inpaint_image(self, crop, mask, options):
        self.calls += 1
        base = cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA).astype(np.float32) + self.bias
        noise = np.random.default_rng(self.calls).normal(0, self.sigma, crop.shape).astype(np.float32)
        out = crop.copy()
        out[mask > 0] = np.clip(base + noise, 0, 255).astype(np.uint8)[mask > 0]
        return out


# --- colour ------------------------------------------------------------------------------------------


def _ring_setup():
    source = np.full((60, 60, 3), 100, np.uint8)
    grown = np.zeros((60, 60), np.uint8)
    grown[20:40, 20:40] = 255
    return source, grown


def test_a_patch_that_is_too_bright_is_pulled_back_to_its_surroundings():
    source, grown = _ring_setup()
    filled = source.copy()
    filled[grown > 0] = 140
    fixed = match_colour(filled, source, grown)
    assert abs(float(fixed[grown > 0].mean()) - 100) < 2
    assert (fixed[grown == 0] == 100).all()  # only the patch is touched


def test_each_channel_is_corrected_separately():
    source, grown = _ring_setup()
    filled = source.copy()
    filled[grown > 0] = (100, 100, 130)  # a red cast
    fixed = match_colour(filled, source, grown)
    assert np.abs(fixed[grown > 0].astype(int) - 100).max() <= 2


def test_small_differences_are_left_alone_to_avoid_adding_flicker():
    source, grown = _ring_setup()
    filled = source.copy()
    filled[grown > 0] = 104
    assert match_colour(filled, source, grown) is filled


def test_a_huge_difference_is_corrected_by_a_bounded_amount():
    source, grown = _ring_setup()
    filled = source.copy()
    filled[grown > 0] = 250
    fixed = match_colour(filled, source, grown)
    assert 205 <= float(fixed[grown > 0].mean()) <= 215  # limited to 40 levels; not blindly forced to match


def test_a_biased_engine_comes_out_with_the_right_brightness_end_to_end():
    scene = make_scene(frames=4)
    out = list(GenerativeStandIn(sigma=0, bias=45).process(scene.marked, scene.masks, OFF))
    signed = np.mean([(f[scene.region].astype(float) - t[scene.region].astype(float)).mean() for f, t in zip(out, scene.clean, strict=True)])
    uncorrected = 45.0
    assert abs(signed) < uncorrected / 3, signed


# --- temporal ----------------------------------------------------------------------------------------


def test_smoothing_cuts_the_shimmer_of_a_per_frame_model():
    scene = make_scene(frames=24)
    plain = list(GenerativeStandIn().process(scene.marked, scene.masks, OFF))
    smooth = list(GenerativeStandIn().process(scene.marked, scene.masks, ON))
    before, after = flicker(plain, scene.region), flicker(smooth, scene.region)
    assert after < 0.6 * before, (before, after)
    # Less shimmer must not come from drifting away from the truth.
    assert mean_error(smooth, scene.clean, scene.region) <= mean_error(plain, scene.clean, scene.region)
    # And the clean footage itself doesn't flicker (up to nothing), so the smoothed result is what we aim for.
    assert flicker(scene.clean, scene.region) < 0.5


def test_smoothing_doesnt_ghost_when_the_camera_pans():
    """The picture slides 3 px per frame behind a watermark fixed to the frame. The previous patch has to be
    carried along with the motion, not smeared."""
    scene = make_scene(frames=20, pan_per_frame=3)
    plain = list(DevInpaintingEngine().process(scene.marked, scene.masks, OFF))
    smooth = list(DevInpaintingEngine().process(scene.marked, scene.masks, ON))
    assert mean_error(smooth, scene.clean, scene.region) <= 1.1 * mean_error(plain, scene.clean, scene.region)

    noisy_plain = list(GenerativeStandIn().process(scene.marked, scene.masks, OFF))
    noisy_smooth = list(GenerativeStandIn().process(scene.marked, scene.masks, ON))
    assert flicker(noisy_smooth, scene.region) < 0.75 * flicker(noisy_plain, scene.region)
    assert mean_error(noisy_smooth, scene.clean, scene.region) <= mean_error(noisy_plain, scene.clean, scene.region)


def test_a_scene_cut_does_not_leak_the_old_scene_into_the_new_one():
    scene = make_scene(frames=14, cut_at=7)
    engine = DevInpaintingEngine()
    plain = list(engine.process(scene.marked, scene.masks, OFF))
    smooth = list(engine.process(scene.marked, scene.masks, ON))
    for t in (7, 8):
        # The first frames of the new scene must match what the engine alone makes of them.
        assert mean_error([smooth[t]], [plain[t]], scene.region) < 1.0, t
    assert mean_error(smooth[7:], scene.clean[7:], scene.region) <= 1.1 * mean_error(plain[7:], scene.clean[7:], scene.region)


def test_frames_with_nothing_marked_in_between_are_fine():
    scene = make_scene(frames=9)
    masks = [m if t not in (3, 4) else np.zeros_like(m) for t, m in enumerate(scene.masks)]
    out = list(GenerativeStandIn().process(scene.marked, masks, ON))
    assert len(out) == 9
    assert np.array_equal(out[3], scene.marked[3]) and np.array_equal(out[4], scene.marked[4])


def test_a_mark_that_moves_between_frames_works_with_smoothing():
    """Tracked marks move. The patch must follow without dragging the previous location along."""
    base = make_scene(frames=16)
    clean, marked, masks = [], [], []
    for t in range(16):
        x = 40 + 8 * t
        shown = base.clean[t].copy()
        shown[60:90, x : x + 60] = (30, 30, 200)
        mask = np.zeros(shown.shape[:2], np.uint8)
        mask[58:92, x - 2 : x + 62] = 255
        clean.append(base.clean[t])
        marked.append(shown)
        masks.append(mask)
    union = np.any(np.stack(masks) > 0, axis=0)
    out = list(GenerativeStandIn(sigma=8).process(marked, masks, ON))
    plain = list(GenerativeStandIn(sigma=8).process(marked, masks, OFF))
    assert mean_error(out, clean, union) <= mean_error(plain, clean, union) * 1.05
    # Away from the mark the picture is the original footage, exactly.
    reach = ON.reach_px + 2
    for t in (0, 8, 15):
        near = cv2.dilate(masks[t], np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)) > 0
        assert np.array_equal(out[t][~near], marked[t][~near])


def test_the_stabilizer_can_be_reset_and_rejects_nonsense_strength():
    import pytest

    with pytest.raises(ValueError):
        TemporalStabilizer(strength=1.0)
    stabilizer = TemporalStabilizer()
    scene = make_scene(frames=2)
    prepared = prepare_mask(scene.masks[0], ON)
    first = scene.marked[0].copy()
    assert stabilizer.stabilize(scene.marked[0], prepared, first) is first  # nothing to blend with on the first frame
    stabilizer.reset()
    again = scene.marked[1].copy()
    assert np.array_equal(stabilizer.stabilize(scene.marked[1], prepared, again), scene.marked[1])


def test_flicker_measures_what_it_says():
    region = np.zeros((10, 10), bool)
    region[2:5, 2:5] = True
    a = np.zeros((10, 10, 3), np.uint8)
    b = a.copy()
    b[region] = 10
    assert flicker([a, b, a], region) == 10.0
    assert flicker([a], region) == 0.0
