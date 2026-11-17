"""The engine contract, region planning and the development engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ai.base import (
    EngineUnavailableError,
    InpaintingEngine,
    InpaintingError,
    InpaintingOptions,
)
from app.ai.dev import DevInpaintingEngine
from app.ai.image import ImageInpaintingEngine
from app.ai.roi import MAX_REGIONS, blend_into, plan_regions, prepare_mask
from tests.inpaint_scene import make_scene, mean_error

OPTIONS = InpaintingOptions()


class FlatFill(ImageInpaintingEngine):
    """A minimal engine written the way a plug-in would be: only the one method."""

    provider = "dev"
    name = "flat"

    def is_available(self) -> bool:
        return True

    def inpaint_image(self, crop, mask, options):
        out = crop.copy()
        out[mask > 0] = np.median(crop[mask == 0], axis=0).astype(np.uint8)  # the typical surrounding colour
        return out


# --- contract ------------------------------------------------------------------------------------------


def test_an_engine_is_swappable_by_implementing_one_method():
    scene = make_scene(frames=3)
    out = list(FlatFill().process(scene.marked, scene.masks, OPTIONS))
    assert len(out) == 3
    inside = out[0][scene.region].astype(int)
    assert inside.std(axis=0).max() < 1.0  # one flat colour...
    assert abs(int(inside[:, 2].mean()) - 200) > 30  # ...and no longer the watermark's red


def test_the_base_class_cannot_be_used_directly():
    with pytest.raises(TypeError):
        InpaintingEngine()  # type: ignore[abstract]


def test_frames_with_nothing_marked_come_back_unchanged_and_independent():
    scene = make_scene(frames=2)
    empty = [np.zeros_like(m) for m in scene.masks]
    out = list(DevInpaintingEngine().process(scene.marked, empty, OPTIONS))
    for before, after in zip(scene.marked, out, strict=True):
        assert np.array_equal(before, after)
        assert after is not before  # callers may edit the result without touching the input


def test_pixels_beyond_the_reach_of_the_mask_are_bit_identical():
    scene = make_scene(frames=4)
    out = list(DevInpaintingEngine().process(scene.marked, scene.masks, OPTIONS))
    reach = OPTIONS.reach_px + 2
    near = cv2.dilate(scene.masks[0], np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)) > 0
    for before, after in zip(scene.marked, out, strict=True):
        assert np.array_equal(before[~near], after[~near])
        assert not np.array_equal(before[scene.region], after[scene.region])


def test_output_is_one_frame_per_input_in_order_and_computed_lazily():
    scene = make_scene(frames=6)
    pulled: list[int] = []

    def source():
        for i, frame in enumerate(scene.marked):
            pulled.append(i)
            yield frame

    stream = DevInpaintingEngine().process(source(), iter(scene.masks), InpaintingOptions(temporal_smoothing=False))
    first = next(stream)
    assert first.shape == scene.marked[0].shape
    assert pulled == [0]  # nothing was read ahead
    assert len(list(stream)) == 5


@pytest.mark.parametrize("extra_frames", [True, False])
def test_unequal_numbers_of_frames_and_masks_are_an_error(extra_frames):
    scene = make_scene(frames=3)
    frames, masks = (scene.marked, scene.masks[:2]) if extra_frames else (scene.marked[:2], scene.masks)
    with pytest.raises(InpaintingError, match="more .* than"):
        list(DevInpaintingEngine().process(frames, masks, OPTIONS))


def test_malformed_input_is_rejected_with_the_frame_number():
    scene = make_scene(frames=2)
    engine = DevInpaintingEngine()
    with pytest.raises(InpaintingError, match="Mask 0 is 10x10"):
        list(engine.process(scene.marked, [np.zeros((10, 10), np.uint8)] * 2, OPTIONS))
    with pytest.raises(InpaintingError, match="Frame 0 must be"):
        list(engine.process([scene.marked[0].astype(np.float32)] * 2, scene.masks, OPTIONS))
    with pytest.raises(InpaintingError, match="Mask 1 must be"):
        list(engine.process(scene.marked, [scene.masks[0], scene.masks[1][..., None]], OPTIONS))


def test_an_engine_that_returns_the_wrong_size_is_caught():
    class Wrong(FlatFill):
        def inpaint_image(self, crop, mask, options):
            return crop[:-1]

    scene = make_scene(frames=1)
    with pytest.raises(InpaintingError, match="doesn't match its input"):
        list(Wrong().process(scene.marked, scene.masks, OPTIONS))


def test_an_unexpected_failure_inside_a_model_becomes_a_readable_error():
    class Crashes(FlatFill):
        def inpaint_image(self, crop, mask, options):
            raise RuntimeError("secret internal detail /srv/models/x.pt")

    scene = make_scene(frames=2)
    with pytest.raises(InpaintingError) as caught:
        list(Crashes().process(scene.marked, scene.masks, OPTIONS))
    assert str(caught.value) == "Reconstruction failed on frame 1."
    assert "secret" not in str(caught.value)


def test_an_engine_that_isnt_available_refuses_to_start():
    class Off(FlatFill):
        def is_available(self) -> bool:
            return False

    scene = make_scene(frames=1)
    with pytest.raises(EngineUnavailableError):
        list(Off().process(scene.marked, scene.masks, OPTIONS))


@pytest.mark.parametrize("bad", [{"quality": "ultra"}, {"feather_px": -1}, {"context_padding_px": -5}, {"edge_dilate_px": -1}])
def test_options_are_validated(bad):
    with pytest.raises(ValueError):
        InpaintingOptions(**bad)


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_every_quality_level_runs(quality):
    scene = make_scene(frames=2)
    out = list(DevInpaintingEngine().process(scene.marked, scene.masks, InpaintingOptions(quality=quality)))
    assert len(out) == 2


# --- the mask, regions and blending --------------------------------------------------------------------


def test_the_prepared_mask_grows_and_feathers_outwards_only():
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:60] = 255
    prepared = prepare_mask(mask, InpaintingOptions(edge_dilate_px=2, feather_px=4))
    assert prepared is not None
    assert (prepared.grown[mask > 0] == 255).all()
    assert (prepared.alpha[mask > 0] == 1.0).all()
    assert prepared.grown[40 - 2, 50] == 255 and prepared.grown[40 - 4, 50] == 0
    # The rim fades from the grown edge and is gone after feather_px + 1 pixels.
    column = prepared.alpha[30:41, 50]
    assert np.all(np.diff(column) >= 0)
    assert prepared.alpha[40 - 2 - 5, 50] == 0.0
    assert 0.0 < prepared.alpha[40 - 2 - 2, 50] < 1.0
    assert prepared.bounds.x0 == 40 - 2 - 4


def test_feather_zero_gives_a_hard_edge():
    mask = np.zeros((50, 50), np.uint8)
    mask[20:30, 20:30] = 255
    prepared = prepare_mask(mask, InpaintingOptions(edge_dilate_px=0, feather_px=0))
    assert set(np.unique(prepared.alpha)) == {0.0, 1.0}
    assert (prepared.grown == mask).all()


def test_an_empty_mask_has_no_regions():
    assert prepare_mask(np.zeros((20, 20), np.uint8), OPTIONS) is None


def test_marks_far_apart_get_their_own_regions_and_near_ones_are_merged():
    options = InpaintingOptions(context_padding_px=16)
    far = np.zeros((300, 600), np.uint8)
    far[10:30, 10:60] = 255
    far[250:280, 520:590] = 255
    rois = plan_regions(prepare_mask(far, options), options)
    assert len(rois) == 2
    assert sum(r.area for r in rois) < 0.25 * far.size  # far less than reconstructing a box around both

    near = np.zeros((300, 600), np.uint8)
    near[100:120, 100:150] = 255
    near[110:140, 160:200] = 255
    assert len(plan_regions(prepare_mask(near, options), options)) == 1


def test_regions_stay_inside_the_frame_and_are_aligned_when_asked():
    mask = np.zeros((100, 130), np.uint8)
    mask[0:12, 118:130] = 255  # in the corner
    options = InpaintingOptions(context_padding_px=20)
    for multiple in (1, 8, 16):
        for roi in plan_regions(prepare_mask(mask, options), options, multiple=multiple):
            assert 0 <= roi.x0 < roi.x1 <= 130 and 0 <= roi.y0 < roi.y1 <= 100
            if multiple > 1 and roi.x1 - roi.x0 < 130:
                assert roi.width % multiple == 0
            assert roi.x0 <= 118 and roi.x1 == 130 and roi.y0 == 0  # still covers the marks


def test_a_speckled_mask_does_not_explode_into_hundreds_of_regions():
    rng = np.random.default_rng(0)
    mask = np.zeros((400, 400), np.uint8)
    for _ in range(MAX_REGIONS * 4):
        x, y = rng.integers(10, 390, 2)
        mask[y : y + 3, x : x + 3] = 255
    options = InpaintingOptions(context_padding_px=4)
    rois = plan_regions(prepare_mask(mask, options), options)
    assert len(rois) <= MAX_REGIONS


def test_blending_leaves_zero_weight_pixels_untouched():
    frame = np.full((20, 20, 3), 50, np.uint8)
    filled = np.full((20, 20, 3), 250, np.uint8)
    alpha = np.zeros((20, 20), np.float32)
    alpha[5:10, 5:10] = 1.0
    alpha[10, 5:10] = 0.5
    blend_into(frame, filled, alpha, plan_regions_whole(20))
    assert (frame[5:10, 5:10] == 250).all()
    assert (frame[10, 5:10] == 150).all()
    assert (frame[0:5] == 50).all() and (frame[11:] == 50).all()


def plan_regions_whole(size: int):
    from app.ai.roi import Roi

    return Roi(0, 0, size, size)


# --- the development engine ----------------------------------------------------------------------------


def test_the_development_engine_is_labelled_and_really_removes_the_mark():
    engine = DevInpaintingEngine()
    info = engine.info()
    assert info.development_only and info.available and not info.sends_frames_to_third_party
    assert "not AI" in info.name

    scene = make_scene(frames=6)
    out = list(engine.process(scene.marked, scene.masks, InpaintingOptions(temporal_smoothing=False)))
    before = mean_error(scene.marked, scene.clean, scene.region)
    after = mean_error(out, scene.clean, scene.region)
    assert after < 0.5 * before, (before, after)
    # The solid red of the watermark must be gone from the middle of it.
    x0, y0, x1, y1 = scene.boxes[0]
    centre = out[0][(y0 + y1) // 2, (x0 + x1) // 2].astype(int)
    assert abs(int(centre[2]) - 200) > 30 or abs(int(centre[0]) - 30) > 30


def test_two_watermarks_in_different_corners_are_both_removed():
    scene = make_scene(frames=3, boxes=[(10, 10, 70, 34), (240, 140, 310, 172)])
    out = list(DevInpaintingEngine().process(scene.marked, scene.masks, OPTIONS))
    assert mean_error(out, scene.clean, scene.region) < 0.5 * mean_error(scene.marked, scene.clean, scene.region)


def test_a_mark_that_touches_the_frame_edge_works():
    scene = make_scene(frames=2, boxes=[(250, 0, 320, 30)], mask_margin=0)
    out = list(DevInpaintingEngine().process(scene.marked, scene.masks, OPTIONS))
    assert mean_error(out, scene.clean, scene.region) < 0.5 * mean_error(scene.marked, scene.clean, scene.region)


def test_a_mask_covering_the_whole_frame_still_returns_a_frame():
    scene = make_scene(frames=1)
    full = [np.full(scene.masks[0].shape, 255, np.uint8)]
    out = list(DevInpaintingEngine().process(scene.marked, full, OPTIONS))
    assert out[0].shape == scene.marked[0].shape
