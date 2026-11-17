"""Overlay inversion against ground truth: a semi-transparent logo over a picture that changes, on a camera that doesn't move."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ai.base import InpaintingError, InpaintingOptions
from app.ai.overlay import OverlayInversionEngine
from tests.test_temporal_engine import make_scene, psnr

H, W = 160, 320
LOGO_BOX = (24, 36, 282, 124)  # the marked area (covers the whole logo): x0, y0, x1, y1


def logo_coverage() -> np.ndarray:
    canvas = np.zeros((H, W), np.uint8)
    cv2.putText(canvas, "WATERMARK", (44, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.25, 255, 3, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def logo_mask() -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    mask[LOGO_BOX[1] : LOGO_BOX[3], LOGO_BOX[0] : LOGO_BOX[2]] = 255
    return mask


def make_clip(count: int = 60, *, opacity: float = 0.45, grey: float = 255.0, colour=None, seed: int = 0, content: str = "changing", gains=None):
    """Frames of a picture that changes (a different part of a big scene each frame), with the logo over it."""
    rng = np.random.default_rng(seed)
    scene = make_scene(1400, 520, seed=5)
    cov = logo_coverage()[..., None]
    paint = np.array(colour if colour is not None else (grey, grey, grey), np.float32)
    truth, shown = [], []
    fixed = scene[60:60 + H, 300:300 + W].astype(np.float32)
    for n in range(count):
        if content == "changing":
            ox, oy = int(rng.integers(0, 1000)), int(rng.integers(0, 340))
            picture = scene[oy : oy + H, ox : ox + W].astype(np.float32)
        elif content == "still":
            picture = fixed
        else:  # "mostly still": a static background with one small moving block
            picture = fixed.copy()
            x = int(rng.integers(0, W - 40))
            picture[100:140, x : x + 40] = rng.integers(0, 255, 3)
        a = opacity * (1.0 if gains is None else gains[n]) * cov
        frame = (1 - a) * picture + a * paint
        frame = np.clip(frame + rng.normal(0, 1.5, frame.shape), 0, 255)
        truth.append(np.rint(picture).astype(np.uint8))
        shown.append(np.rint(frame).astype(np.uint8))
    return truth, shown, [logo_mask() for _ in range(count)]


def run(frames, masks, **kwargs):
    engine = OverlayInversionEngine(**kwargs)
    return engine, list(engine.process(iter(frames), iter(masks), InpaintingOptions()))


def logo_psnr(result, truth) -> float:
    inside = logo_coverage() > 0.02
    return float(np.mean([psnr(r[inside], t[inside]) for r, t in zip(result, truth, strict=True)]))


@pytest.mark.parametrize(("opacity", "grey"), [(0.45, 255.0), (0.30, 255.0), (0.55, 0.0)])
def test_a_semi_transparent_logo_is_inverted_back_to_the_real_pixels(opacity: float, grey: float) -> None:
    truth, shown, masks = make_clip(opacity=opacity, grey=grey)
    engine, result = run(shown, masks)
    before = logo_psnr(shown, truth)
    after = logo_psnr(result, truth)
    assert after > 33, (before, after)
    assert after > before + 15
    assert "was measured from the video itself" in engine.notes()[0]
    assert abs(engine._model.opacity - opacity) < 0.06 and np.abs(engine._model.colour - grey).max() < 12


def test_pixels_outside_the_marked_area_are_untouched() -> None:
    _truth, shown, masks = make_clip(count=24)
    _, result = run(shown, masks)
    outside = cv2.dilate(masks[0], np.ones((11, 11), np.uint8)) == 0
    for before, after in zip(shown, result, strict=True):
        assert np.array_equal(before[outside], after[outside])


def test_frames_where_the_logo_is_not_shown_are_left_alone() -> None:
    truth, shown, masks = make_clip(count=30)
    clean_index = 7
    shown[clean_index] = truth[clean_index].copy()
    masks[clean_index] = np.zeros_like(masks[clean_index])
    _, result = run(shown, masks)
    assert np.array_equal(result[clean_index], truth[clean_index])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"content": "still"}, "hardly changes"),
        ({"content": "mostly still"}, None),  # must be refused, or restored properly: never silently wrong
        ({"opacity": 1.0}, None),
    ],
)
def test_footage_it_cannot_handle_is_refused_or_restored_correctly_never_silently_wrong(kwargs, message) -> None:
    truth, shown, masks = make_clip(**kwargs)
    try:
        _, result = run(shown, masks)
    except InpaintingError as error:
        if message:
            assert message in str(error)
        return
    assert message is None
    assert logo_psnr(result, truth) > 28, "an unsafe result was returned instead of stopping"


@pytest.mark.parametrize(("colour", "minimum"), [((30, 40, 255), 33), ((200, 120, 20), 29)])
def test_a_coloured_logo_is_measured_and_inverted_too(colour, minimum) -> None:
    truth, shown, masks = make_clip(colour=colour, opacity=0.45)
    try:
        engine, result = run(shown, masks)
    except InpaintingError:
        assert minimum < 33, "the easy colour must be restored"  # a hard one may be refused, never half-removed
        return
    assert logo_psnr(result, truth) > minimum, (logo_psnr(shown, truth), logo_psnr(result, truth))
    assert np.abs(engine._model.colour - np.array(colour)).max() < 30
    assert "coloured" in engine.notes()[0]


def test_a_logo_of_two_colours_is_refused_not_half_removed() -> None:
    truth, shown, masks = make_clip(opacity=0.45)
    cov = logo_coverage()
    left = np.zeros((H, W, 1), np.float32)
    left[:, : W // 2] = 1.0
    rng = np.random.default_rng(3)
    shown = []
    for t in truth:
        a = 0.45 * cov[..., None]
        paint = left * np.array([20, 20, 255], np.float32) + (1 - left) * np.array([255, 200, 20], np.float32)
        frame = np.clip((1 - a) * t.astype(np.float32) + a * paint + rng.normal(0, 1.5, t.shape), 0, 255)
        shown.append(np.rint(frame).astype(np.uint8))
    try:
        _, result = run(shown, masks)
    except InpaintingError:
        return
    assert logo_psnr(result, truth) > 28, "an unsafe result was returned instead of stopping"


def test_a_mark_that_moves_is_refused() -> None:
    _truth, shown, masks = make_clip(count=30)
    masks[10] = np.roll(masks[10], 5, axis=1)
    with pytest.raises(InpaintingError, match="moves or changes"):
        run(shown, masks)


def test_a_clip_that_is_too_short_is_refused() -> None:
    _truth, shown, masks = make_clip(count=8)
    with pytest.raises(InpaintingError, match="too short"):
        run(shown, masks)


def test_the_mark_changing_after_the_measured_frames_stops_the_render() -> None:
    _truth, shown, masks = make_clip(count=40)
    masks[35] = np.roll(masks[35], 5, axis=1)
    with pytest.raises(InpaintingError, match="frame 36"):
        run(shown, masks, sample_frames=20)


# -- automatic choice --------------------------------------------------------------------------------------


def test_auto_takes_a_semi_transparent_logo_off_exactly_and_says_so() -> None:
    from app.ai.auto import AutoRemovalEngine

    truth, shown, masks = make_clip()
    engine = AutoRemovalEngine()
    result = list(engine.process(iter(shown), iter(masks), InpaintingOptions()))
    assert logo_psnr(result, truth) > 33
    assert engine._chosen is engine._overlay and "inverted" in engine.notes()[0]


def test_auto_rebuilds_from_other_frames_when_the_logo_is_opaque_and_the_camera_pans() -> None:
    from app.ai.auto import AutoRemovalEngine
    from tests.test_temporal_engine import hole_psnr, mark_mask, panning_clip

    truth, marked, masks = panning_clip()
    engine = AutoRemovalEngine()
    result = list(engine.process(iter(marked), iter(masks), InpaintingOptions()))
    assert min(hole_psnr(r, t, m) for r, t, m in zip(result, truth, masks, strict=True)) > 28
    assert engine._chosen is engine._reconstruct
    notes = engine.notes()
    assert "couldn't be used" in notes[0] and "rebuilt from other frames" in notes[1]
    assert mark_mask().any()


def test_auto_stops_when_neither_method_can_work() -> None:
    from app.ai.auto import AutoRemovalEngine
    from tests.test_temporal_engine import (
        WINDOW_H,
        WINDOW_W,
        make_scene,
        mark_mask,
        noisy,
        watermark,
    )

    scene = make_scene()
    window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 : 100 + WINDOW_W])
    rng = np.random.default_rng(9)
    marked = [watermark(noisy(window, rng)) for _ in range(30)]
    masks = [mark_mask() for _ in range(30)]
    with pytest.raises(InpaintingError, match="hidden in every nearby frame"):
        list(AutoRemovalEngine().process(iter(marked), iter(masks), InpaintingOptions()))


def fade_gains(count: int, fade: int = 10) -> list[float]:
    ramp = [i / fade for i in range(fade)]
    return ramp + [1.0] * (count - 2 * fade) + ramp[::-1]


def test_a_logo_that_fades_in_and_out_is_taken_off_in_every_frame() -> None:
    gains = fade_gains(60)
    truth, shown, masks = make_clip(count=60, opacity=0.45, gains=gains)
    engine, result = run(shown, masks)
    inside = logo_coverage() > 0.02
    per_frame = [psnr(r[inside], t[inside]) for r, t in zip(result, truth, strict=True)]
    before = [psnr(r[inside], t[inside]) for r, t in zip(shown, truth, strict=True)]
    assert np.mean(per_frame) > 31, per_frame
    assert min(per_frame) > 26, per_frame
    assert np.mean(per_frame) > np.mean(before) + 8
    assert any("strength changed" in note for note in engine.notes()), engine.notes()


def test_a_logo_that_is_not_there_at_the_start_does_not_damage_those_frames() -> None:
    gains = [0.0] * 14 + [1.0] * 46
    truth, shown, masks = make_clip(count=60, opacity=0.45, gains=gains)
    engine, result = run(shown, masks)
    inside = logo_coverage() > 0.02
    absent = [psnr(r[inside], t[inside]) for r, t in zip(result[:14], truth[:14], strict=True)]
    present = [psnr(r[inside], t[inside]) for r, t in zip(result[14:], truth[14:], strict=True)]
    assert min(absent) > 38, absent  # left as they were (noise only)
    assert np.mean(present) > 33, present
    assert "not visible in 14" in " ".join(engine.notes()) or "not visible in 1" in " ".join(engine.notes()), engine.notes()


def test_a_constant_logo_reports_no_change_in_strength() -> None:
    _truth, shown, masks = make_clip(count=40)
    engine, _ = run(shown, masks)
    assert not any("strength changed" in note for note in engine.notes()), engine._gains


def test_footage_that_was_enlarged_is_still_restored() -> None:
    """Enlarged footage is smoother than the test pictures, which used to push the edge check over its limit."""
    truth, shown, masks = make_clip(count=40)
    f = 3
    resize = lambda a, mode: cv2.resize(a, None, fx=f, fy=f, interpolation=mode)
    big_truth = [resize(t, cv2.INTER_LANCZOS4) for t in truth]
    big_shown = [resize(s, cv2.INTER_LANCZOS4) for s in shown]
    big_masks = [resize(m, cv2.INTER_NEAREST) for m in masks]
    _, result = run(big_shown, big_masks)
    inside = resize(logo_coverage(), cv2.INTER_LANCZOS4) > 0.02
    after = np.mean([psnr(r[inside], t[inside]) for r, t in zip(result, big_truth, strict=True)])
    before = np.mean([psnr(s[inside], t[inside]) for s, t in zip(big_shown, big_truth, strict=True)])
    assert after > 33 and after > before + 15, (before, after)


def test_a_small_memory_budget_measures_from_fewer_frames_and_still_restores(monkeypatch) -> None:
    from app.ai import overlay

    seen: list[int] = []
    real = overlay.estimate_overlay

    def spy(crops, *args, **kwargs):
        seen.append(len(crops))
        return real(crops, *args, **kwargs)

    monkeypatch.setattr(overlay, "estimate_overlay", spy)
    truth, shown, masks = make_clip(count=60)
    _, relaxed = run(shown, masks, memory_mb=4096)
    _, tight = run(shown, masks, memory_mb=1)
    assert seen[0] > seen[1] == overlay.MIN_SAMPLE, seen
    assert logo_psnr(tight, truth) > 33 and logo_psnr(relaxed, truth) > 33
