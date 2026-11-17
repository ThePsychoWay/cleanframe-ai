"""Large marked areas are filled in overlapping pieces at full resolution (no shrinking), without seams, and a model
that runs out of memory gets smaller pieces instead of crashing the render.

The stand-in model fills the hole with OpenCV's inpainting: it depends on the surroundings (unlike a flat colour), so
a wrong overlap, a missed pixel or a seam shows up as an error against the known picture.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ai.base import DeviceInfo, InpaintingError, InpaintingOptions, ModelMemoryExhausted
from app.ai.lama import TILE_OVERLAP, LamaOnnxEngine, OutOfModelMemory, plan_axis


class FakeRuntime:
    fixed_size = None

    def __init__(self, *, memory_limit_side: int | None = None, start_side: int = 1024, kind: str = "cpu") -> None:
        self.kind = kind
        self.shapes: list[tuple[int, int]] = []
        self.memory_limit_side = memory_limit_side
        self.start_side = start_side

    def problem(self):  # noqa: ANN201
        return None

    def load(self) -> None: ...

    def device_info(self) -> DeviceInfo:
        return DeviceInfo(self.kind, "GPU" if self.kind == "cuda" else "CPU", "")

    def suggested_tile_side(self) -> int:
        return self.start_side

    def run(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.shapes.append(rgb.shape[:2])
        if self.memory_limit_side is not None and max(rgb.shape[:2]) > self.memory_limit_side:
            raise OutOfModelMemory("out of memory")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, (mask > 127).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def smooth_scene(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    b = 120 + 60 * np.sin(x / 90.0) + 30 * np.cos(y / 70.0)
    g = 110 + 50 * np.cos(x / 130.0 + y / 160.0)
    r = 100 + 70 * np.sin(y / 110.0 + x / 200.0)
    return np.clip(np.stack([b, g, r], -1), 0, 255).astype(np.uint8)


def hole_in(height: int, width: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if error == 0 else 10 * float(np.log10(255.0**2 / error))


@pytest.mark.parametrize("length", [1, 500, 1024, 1025, 1500, 3000, 3840, 4097])
@pytest.mark.parametrize("side", [384, 512, 1024])
def test_pieces_cover_the_whole_axis_exactly_once_and_stay_within_the_side(length: int, side: int) -> None:
    pieces = plan_axis(length, side, TILE_OVERLAP)
    covered = np.zeros(length, int)
    for start, stop, commit_start, commit_stop in pieces:
        assert 0 <= start < stop <= length and stop - start <= side
        assert start <= commit_start < commit_stop <= stop
        covered[commit_start:commit_stop] += 1
    assert (covered == 1).all()


def test_a_wide_hole_is_filled_in_pieces_at_full_size_and_matches_the_truth_without_seams() -> None:
    truth = smooth_scene(300, 3000)  # a caption across a 4K-wide frame
    mask = hole_in(300, 3000, 100, 100, 2900, 200)
    image = truth.copy()
    image[mask > 0] = (255, 255, 255)
    runtime = FakeRuntime()
    engine = LamaOnnxEngine(runtime)  # type: ignore[arg-type]
    result = engine.inpaint_image(image, mask, InpaintingOptions())

    assert len(runtime.shapes) >= 3 and all(max(shape) <= 1024 for shape in runtime.shapes)  # pieces, none shrunk
    assert np.array_equal(result[mask == 0], image[mask == 0])  # outside the hole: untouched
    assert (result[mask > 0] != 255).any(axis=-1).all()  # every hole pixel was replaced
    whole = FakeRuntime().run(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask)  # the same model on the whole picture at once
    whole = cv2.cvtColor(whole, cv2.COLOR_RGB2BGR)
    assert psnr(result[mask > 0], truth[mask > 0]) >= psnr(whole[mask > 0], truth[mask > 0]) - 1.5  # no worse than unsplit
    # No jump where one piece hands over to the next: look at the column-to-column step across the hole.
    step = np.abs(np.diff(result[100:200].astype(np.int16), axis=1)).mean(axis=(0, 2))
    assert step.max() < 3 * np.median(step) + 2
    assert any("overlapping pieces" in note and "nothing was shrunk" in note for note in engine.notes())


def test_a_hole_that_fits_in_one_piece_is_one_call_and_says_nothing_about_pieces() -> None:
    truth = smooth_scene(300, 600)
    mask = hole_in(300, 600, 200, 100, 400, 200)
    runtime = FakeRuntime()
    engine = LamaOnnxEngine(runtime)  # type: ignore[arg-type]
    engine.inpaint_image(truth, mask, InpaintingOptions())
    assert len(runtime.shapes) == 1
    assert not any("pieces" in note for note in engine.notes())


def test_a_model_that_runs_out_of_memory_is_given_smaller_pieces() -> None:
    truth = smooth_scene(400, 2400)
    mask = hole_in(400, 2400, 100, 100, 2300, 300)
    runtime = FakeRuntime(memory_limit_side=520)
    engine = LamaOnnxEngine(runtime)  # type: ignore[arg-type]
    result = engine.inpaint_image(truth, mask, InpaintingOptions())
    assert max(runtime.shapes[-1]) <= 512
    assert (result[mask > 0] != 255).any(axis=-1).all() and np.array_equal(result[mask == 0], truth[mask == 0])
    assert any("up to 512 px" in note for note in engine.notes())


def test_when_even_the_smallest_piece_does_not_fit_the_error_says_what_to_do() -> None:
    truth = smooth_scene(400, 2400)
    mask = hole_in(400, 2400, 100, 100, 2300, 300)
    engine = LamaOnnxEngine(FakeRuntime(memory_limit_side=100))  # type: ignore[arg-type]
    with pytest.raises(InpaintingError, match="doesn't have enough memory to rebuild the marked area"):
        engine.inpaint_image(truth, mask, InpaintingOptions())


def test_the_out_of_memory_message_names_the_video_size_and_says_gpu_memory_on_a_gpu() -> None:
    truth = smooth_scene(400, 2400)
    mask = hole_in(400, 2400, 100, 100, 2300, 300)
    engine = LamaOnnxEngine(FakeRuntime(memory_limit_side=100, kind="cuda"))  # type: ignore[arg-type]
    with pytest.raises(ModelMemoryExhausted) as caught:
        engine.inpaint_image(truth, mask, InpaintingOptions())
    assert "2400 × 400" in str(caught.value) and "GPU memory (VRAM)" in str(caught.value)


def test_a_render_failing_that_way_shows_that_message_not_the_generic_out_of_memory_one() -> None:
    from app.errors import RESOURCES_EXHAUSTED
    from app.services.rendering import _message_for

    try:
        try:
            raise MemoryError  # what the model's allocator raised underneath
        except MemoryError:
            raise ModelMemoryExhausted("Your video is 3840 × 2160, but the processing worker doesn't have enough memory.") from None
    except ModelMemoryExhausted as error:
        assert _message_for(error) == str(error)
        assert _message_for(error) != RESOURCES_EXHAUSTED


def test_a_small_gpu_starts_with_smaller_pieces() -> None:
    truth = smooth_scene(300, 2000)
    mask = hole_in(300, 2000, 50, 100, 1950, 200)
    runtime = FakeRuntime(start_side=512)
    LamaOnnxEngine(runtime).inpaint_image(truth, mask, InpaintingOptions())  # type: ignore[arg-type]
    assert all(max(shape) <= 512 for shape in runtime.shapes)
