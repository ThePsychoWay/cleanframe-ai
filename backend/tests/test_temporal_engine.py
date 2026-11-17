"""The reconstruction engine against ground truth.

The scene is a large textured picture seen through a window. The "watermark" is an opaque box pasted over the
frames; the clean window is what a perfect remover would give back. Quality is measured as PSNR inside the marked
area against that truth and compared with plain OpenCV inpainting, so these tests show whether real pixels were
recovered, not just whether something was written there.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ai.base import InpaintingError, InpaintingOptions
from app.ai.image import ImageInpaintingEngine
from app.ai.temporal import TemporalReconstructionEngine

WINDOW_W, WINDOW_H = 320, 200
MARK = (230, 140, 258, 158)  # x0, y0, x1, y1 on screen: 28 x 18 pixels


def make_scene(width: int = 1400, height: int = 260, seed: int = 7) -> np.ndarray:
    """Textured colour image with detail at several scales, like a real shot (not a flat or repeating pattern)."""
    rng = np.random.default_rng(seed)
    scene = np.zeros((height, width, 3), np.float32)
    for sigma, weight in ((2, 60), (6, 55), (18, 45)):
        noise = rng.normal(0, 1, (height, width, 3)).astype(np.float32)
        scene += cv2.GaussianBlur(noise, (0, 0), sigma) * weight * (sigma**0.5)
    scene += 128
    out = np.clip(scene, 0, 255).astype(np.uint8)
    for _ in range(90):  # edges and blobs
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(out, (x, y), int(rng.integers(5, 28)), colour, -1)
        else:
            cv2.rectangle(out, (x, y), (x + int(rng.integers(8, 50)), y + int(rng.integers(8, 40))), colour, -1)
    return out


def noisy(frame: np.ndarray, rng: np.random.Generator, sigma: float = 1.5) -> np.ndarray:
    return np.clip(frame.astype(np.float32) + rng.normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)


def mark_mask(box: tuple[int, int, int, int] = MARK) -> np.ndarray:
    mask = np.zeros((WINDOW_H, WINDOW_W), np.uint8)
    mask[box[1] : box[3], box[0] : box[2]] = 255
    return mask


def watermark(frame: np.ndarray, box: tuple[int, int, int, int] = MARK) -> np.ndarray:
    marked = frame.copy()
    marked[box[1] : box[3], box[0] : box[2]] = (255, 255, 255)
    cv2.putText(marked, "LOGO", (box[0] + 2, box[3] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return marked


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if error == 0 else 10 * np.log10(255.0**2 / error)


def hole_psnr(result: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    inside = mask > 0
    return psnr(result[inside], truth[inside])


def panning_clip(count: int = 30, step: int = 6, rng_seed: int = 1):
    scene = make_scene()
    rng = np.random.default_rng(rng_seed)
    truth, marked, masks = [], [], []
    for i in range(count):
        window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 60 + step * i : 60 + step * i + WINDOW_W])
        truth.append(window)
        marked.append(watermark(noisy(window, rng)))
        masks.append(mark_mask())
    return truth, marked, masks


def run(engine, frames, masks, **options):
    return list(engine.process(iter(frames), iter(masks), InpaintingOptions(**options)))


def test_removes_a_fixed_watermark_from_a_panning_shot_with_real_pixels() -> None:
    truth, marked, masks = panning_clip()
    engine = TemporalReconstructionEngine()
    result = run(engine, marked, masks)

    assert len(result) == len(marked)
    scores, baseline = [], []
    for i in range(len(marked)):
        scores.append(hole_psnr(result[i], truth[i], masks[i]))
        base = cv2.inpaint(marked[i], masks[i], 3, cv2.INPAINT_TELEA)
        baseline.append(hole_psnr(base, truth[i], masks[i]))
    # The rebuilt area matches the real background closely, far better than spatial inpainting can.
    assert min(scores) > 28, scores
    assert np.mean(scores) > np.mean(baseline) + 8
    # The logo is gone: its white box and black text no longer exist inside the marked area.
    for i in (0, 12, 29):
        inside = result[i][mark_mask() > 0]
        assert inside.min() > 0 or (inside == 0).all(axis=1).sum() == 0
        assert np.abs(inside.astype(int) - truth[i][mark_mask() > 0].astype(int)).mean() < 6

    notes = engine.notes()
    assert len(notes) == 1 and "rebuilt from other frames" in notes[0] and "Nothing was generated" in notes[0]


def test_pixels_outside_the_marked_area_are_untouched() -> None:
    truth, marked, masks = panning_clip(count=14)
    result = run(TemporalReconstructionEngine(), marked, masks)
    options = InpaintingOptions()
    near = cv2.dilate(masks[0], np.ones((2 * (options.reach_px + 2) + 1,) * 2, np.uint8)) > 0
    for before, after in zip(marked, result, strict=True):
        assert np.array_equal(before[~near], after[~near])


def test_frames_without_a_mark_come_back_unchanged() -> None:
    truth, marked, masks = panning_clip(count=12)
    masks = [m if i % 2 == 0 else np.zeros_like(m) for i, m in enumerate(masks)]
    result = run(TemporalReconstructionEngine(), marked, masks)
    for i in range(1, 12, 2):
        assert np.array_equal(result[i], marked[i])


def test_a_mark_that_moves_on_a_still_camera_is_rebuilt_from_the_frames_it_uncovered() -> None:
    scene = make_scene()
    window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 : 100 + WINDOW_W])
    rng = np.random.default_rng(3)
    marked, masks = [], []
    for i in range(30):  # the mark slides 6 px per frame, so every pixel is uncovered within a few frames
        box = (20 + 6 * i, 60, 20 + 6 * i + 28, 78)
        marked.append(watermark(noisy(window, rng), box))
        masks.append(mark_mask(box))
    result = run(TemporalReconstructionEngine(), marked, masks)
    scores = [hole_psnr(result[i], window, masks[i]) for i in range(len(marked))]
    assert min(scores) > 30, scores


def test_a_mark_fixed_on_a_still_camera_cannot_be_rebuilt_and_the_render_stops() -> None:
    scene = make_scene()
    window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 : 100 + WINDOW_W])
    rng = np.random.default_rng(4)
    marked = [watermark(noisy(window, rng)) for _ in range(12)]
    masks = [mark_mask() for _ in range(12)]
    with pytest.raises(InpaintingError, match="hidden in every nearby frame"):
        run(TemporalReconstructionEngine(), marked, masks)


class _FlatFill(ImageInpaintingEngine):
    """Stands in for a generative engine: fills the hole with the mean colour of the crop."""

    provider = "flat-test"
    name = "Flat fill"

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def inpaint_image(self, crop, mask, options):  # noqa: ANN001
        self.calls += 1
        out = crop.copy()
        out[mask > 0] = crop[mask == 0].mean(axis=0).astype(np.uint8)
        return out


def test_what_no_frame_can_supply_goes_to_the_fallback_and_the_notes_say_so() -> None:
    scene = make_scene()
    window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 : 100 + WINDOW_W])
    rng = np.random.default_rng(5)
    marked = [watermark(noisy(window, rng)) for _ in range(8)]
    masks = [mark_mask() for _ in range(8)]
    fallback = _FlatFill()
    engine = TemporalReconstructionEngine(fallback=fallback, fallback_label="the test fill")
    result = run(engine, marked, masks)

    assert fallback.calls == 1  # the picture around the mark only differs by sensor noise: one fill serves every frame
    assert all(not np.array_equal(r, m) for r, m in zip(result, marked, strict=True))
    note = engine.notes()[0]
    assert "0% of the marked area was rebuilt from other frames" in note
    assert "100% was hidden in every frame and was generated by the test fill" in note


def test_partly_recoverable_marks_report_both_shares() -> None:
    """A slow pan: the edges of the mark get uncovered, its middle never does within the short window."""
    truth, marked, masks = panning_clip(count=24, step=2)
    fallback = _FlatFill()
    engine = TemporalReconstructionEngine(fallback=fallback, fallback_label="the test fill", window=6)
    run(engine, marked, masks)
    note = engine.notes()[0]
    assert "rebuilt from other frames" in note and "generated by the test fill" in note
    assert not note.startswith("0%") and not note.startswith("100%")


def test_a_cut_does_not_pull_pixels_from_the_other_scene() -> None:
    scene_a, scene_b = make_scene(seed=11), make_scene(seed=99)
    rng = np.random.default_rng(6)
    truth, marked, masks = [], [], []
    for i in range(28):
        scene = scene_a if i < 14 else scene_b
        offset = 60 + 6 * (i if i < 14 else i - 14)
        window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, offset : offset + WINDOW_W])
        truth.append(window)
        marked.append(watermark(noisy(window, rng)))
        masks.append(mark_mask())
    result = run(TemporalReconstructionEngine(), marked, masks, quality="high")
    for i in range(28):
        assert hole_psnr(result[i], truth[i], masks[i]) > 25, i


def test_a_short_clip_without_enough_frames_stops_instead_of_guessing() -> None:
    truth, marked, masks = panning_clip(count=1)
    with pytest.raises(InpaintingError):
        run(TemporalReconstructionEngine(), marked, masks)


def test_the_window_bounds_memory() -> None:
    """Frames far behind the target are dropped: the buffer never holds more than 2 * window + 1 of them."""
    truth, marked, masks = panning_clip(count=40)
    engine = TemporalReconstructionEngine(fallback=_FlatFill(), window=4)
    seen = []
    original = engine._enter

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003
        entry = original(*args, **kwargs)
        seen.append(entry.index)
        return entry

    engine._enter = spy  # type: ignore[method-assign]
    produced = 0
    for _ in engine.process(iter(marked), iter(masks), InpaintingOptions()):
        produced += 1
        assert max(seen) - (produced - 1) <= 4  # never reads further ahead than its window
    assert produced == 40


def test_unavailable_when_its_fallback_is_unavailable() -> None:
    class Missing(_FlatFill):
        def is_available(self) -> bool:
            return False

        def unavailable_reason(self) -> str | None:
            return "no model"

    engine = TemporalReconstructionEngine(fallback=Missing())
    assert not engine.is_available() and engine.unavailable_reason() == "no model"
    assert TemporalReconstructionEngine().is_available()


# -- configuration and the registry -----------------------------------------------------------------------


def _settings(monkeypatch: pytest.MonkeyPatch, **values: str):
    from app.config import Settings, get_settings

    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///x.db")
    monkeypatch.setenv("JWT_SECRET", "x" * 64)
    for name in ("AI_PROVIDER", "AI_MODEL_PATH", "AI_MODEL_URL", "AI_MODEL_SHA256", "RECONSTRUCT_FALLBACK", "DEV_MODE"):
        monkeypatch.delenv(name, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key.upper(), value)
    get_settings.cache_clear()
    return Settings()


def test_the_reconstruct_provider_has_a_fallback_only_when_one_is_configured(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from app.ai.lama import LamaOnnxEngine
    from app.ai.registry import create_engine, get_engine

    plain = create_engine(_settings(monkeypatch, ai_provider="reconstruct"))
    assert isinstance(plain, TemporalReconstructionEngine) and plain._fallback is None
    assert get_engine(_settings(monkeypatch, ai_provider="reconstruct")).is_available()
    assert "can't be rebuilt" in plain.expectation()

    model = tmp_path / "lama.onnx"
    with_lama = create_engine(_settings(monkeypatch, ai_provider="reconstruct", ai_model_path=str(model)))
    assert isinstance(with_lama._fallback, LamaOnnxEngine)
    assert not with_lama.is_available()  # the model file isn't there: an error, not a quiet switch
    none = create_engine(_settings(monkeypatch, ai_provider="reconstruct", ai_model_path=str(model), reconstruct_fallback="none"))
    assert none._fallback is None
    assert isinstance(create_engine(_settings(monkeypatch, ai_provider="lama")), LamaOnnxEngine)

    from app.ai.auto import AutoRemovalEngine
    from app.ai.overlay import OverlayInversionEngine

    auto = create_engine(_settings(monkeypatch, ai_provider="auto"))
    assert isinstance(auto, AutoRemovalEngine) and auto.is_available()
    assert isinstance(create_engine(_settings(monkeypatch, ai_provider="overlay")), OverlayInversionEngine)
    assert not create_engine(_settings(monkeypatch, ai_provider="auto", ai_model_path=str(model))).is_available()


def test_production_configuration_rules_for_the_new_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import ConfigError

    base = {"dev_mode": "false", "rate_limit_enabled": "true", "storage_bucket": "b", "cors_origins": "https://x.test"}
    _settings(monkeypatch, ai_provider="reconstruct", **base).validate_runtime()
    with pytest.raises(ConfigError, match="AI_MODEL_PATH"):
        _settings(monkeypatch, ai_provider="reconstruct", reconstruct_fallback="lama", **base).validate_runtime()
    with pytest.raises(ConfigError, match="SHA256"):
        _settings(monkeypatch, ai_provider="lama", ai_model_path="/m/l.onnx", ai_model_url="https://x.test/l.onnx", **base).validate_runtime()
    with pytest.raises(ConfigError, match="HTTPS"):
        _settings(
            monkeypatch, ai_provider="lama", ai_model_path="/m/l.onnx", ai_model_url="http://x.test/l.onnx", ai_model_sha256="a" * 64, **base
        ).validate_runtime()
    with pytest.raises(ConfigError, match="AI_MODEL_PATH"):
        _settings(monkeypatch, ai_provider="lama", ai_model_url="https://x.test/l.onnx", ai_model_sha256="a" * 64, **base).validate_runtime()
    _settings(
        monkeypatch, ai_provider="lama", ai_model_path="/m/l.onnx", ai_model_url="https://x.test/l.onnx", ai_model_sha256="a" * 64, **base
    ).validate_runtime()


# -- speed-ups that must not change what comes out ----------------------------------------------------------------


def _still_clip(count: int, sigma: float, seed: int = 5):
    scene = make_scene()
    window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 : 100 + WINDOW_W])
    rng = np.random.default_rng(seed)
    base = window if sigma == 0 else None
    marked = [watermark(window if base is not None else noisy(window, rng)) for _ in range(count)]
    return marked, [mark_mask() for _ in range(count)]


def test_a_generated_fill_is_reused_while_the_picture_around_the_mark_is_unchanged() -> None:
    marked, masks = _still_clip(8, sigma=0)
    fallback = _FlatFill()
    engine = TemporalReconstructionEngine(fallback=fallback, fallback_label="the test fill")
    result = run(engine, marked, masks)

    assert fallback.calls == 1  # made once, then reused for the other seven frames
    assert all(np.array_equal(result[0], r) for r in result)  # the same patch every time: nothing to flicker
    assert any("reused 7 times" in note for note in engine.notes())


def test_a_changed_picture_around_the_mark_gets_a_fresh_fill() -> None:
    marked, masks = _still_clip(6, sigma=0)
    for i in range(3, 6):  # the scene changes (a brightness jump) half way through
        marked[i] = np.clip(marked[i].astype(np.int16) + 40, 0, 255).astype(np.uint8)
        marked[i][MARK[1] : MARK[3], MARK[0] : MARK[2]] = 255
    fallback = _FlatFill()
    run(TemporalReconstructionEngine(fallback=fallback), marked, masks)
    assert fallback.calls == 2


def test_high_quality_only_reuses_an_identical_context() -> None:
    marked, masks = _still_clip(6, sigma=0.4)  # a little noise: not identical
    fallback = _FlatFill()
    engine = TemporalReconstructionEngine(fallback=fallback)
    options = InpaintingOptions(quality="high", extra={})
    list(engine.process(iter(marked), iter(masks), options))
    assert fallback.calls == 6


def test_skipping_the_search_for_footage_that_cannot_exist_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same picture position and same mark in the neighbour: it hides exactly what we are missing. Skipping it must give
    the same frames as checking it (checked by turning the skip off)."""
    import app.ai.temporal as temporal

    truth, marked, masks = panning_clip(count=14, step=0)  # a still camera with a fixed mark and a moving-free scene
    fast = run(TemporalReconstructionEngine(fallback=_FlatFill()), marked, masks)
    monkeypatch.setattr(temporal, "_IDENTITY", np.full((3, 3), np.nan))  # nothing equals this: no neighbour is skipped
    slow = run(TemporalReconstructionEngine(fallback=_FlatFill()), marked, masks)
    for a, b in zip(fast, slow, strict=True):
        assert np.array_equal(a, b)


def test_the_search_window_shrinks_to_the_memory_budget_and_the_notes_say_so() -> None:
    truth, marked, masks = panning_clip(count=24, step=3)
    big = TemporalReconstructionEngine(fallback=_FlatFill(), memory_mb=1)  # far too little for the default window
    result = run(big, marked, masks)
    assert len(result) == len(marked)
    assert any("were limited to 3 each side" in note and "memory budget" in note for note in big.notes())

    roomy = TemporalReconstructionEngine(fallback=_FlatFill(), memory_mb=1536)
    run(roomy, marked, masks)
    assert not any("memory budget" in note for note in roomy.notes())
