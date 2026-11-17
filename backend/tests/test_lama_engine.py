"""LaMa plumbing, tested with a small stand-in ONNX model (the real weights are large and aren't shipped).

The stand-in has the same inputs and outputs as the public LaMa exports (image [1,3,H,W] 0..1 RGB, mask [1,1,H,W],
one image out) and paints the hole a flat colour, so these tests show that sizes, colour order, scaling, padding,
compositing and weight handling are right, not that any model is good. Model quality has to be judged with the real
file (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from app.ai.base import EngineUnavailableError, InpaintingOptions
from app.ai.lama import LamaOnnxEngine, LamaOnnxRuntime
from app.ai.temporal import TemporalReconstructionEngine
from app.ai.weights import ensure_weights, sha256_of


def build_model(path: Path, *, size: int | None, scale_255: bool, colour=(0.2, 0.5, 0.8)) -> Path:
    """out = image * (1 - mask) + colour * mask   (RGB), optionally in 0..255."""
    dim = [1, 3, size or "h", size or "w"]
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, dim)
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, size or "h", size or "w"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, dim)
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    paint = helper.make_tensor("paint", TensorProto.FLOAT, [1, 3, 1, 1], list(colour))
    factor = helper.make_tensor("factor", TensorProto.FLOAT, [1], [255.0 if scale_255 else 1.0])
    nodes = [
        helper.make_node("Sub", ["one", "mask"], ["keep"]),
        helper.make_node("Mul", ["image", "keep"], ["kept"]),
        helper.make_node("Mul", ["paint", "mask"], ["painted"]),
        helper.make_node("Add", ["kept", "painted"], ["mixed"]),
        helper.make_node("Mul", ["mixed", "factor"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "lama-stand-in", [image, mask], [out], initializer=[one, paint, factor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())
    return path


def crop_and_mask(height: int, width: int):
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), np.uint8)
    mask[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 255
    return crop, mask


@pytest.mark.parametrize("size", [None, 512])
@pytest.mark.parametrize("scale_255", [False, True])
@pytest.mark.parametrize("shape", [(120, 90), (700, 900)])
def test_fills_only_the_hole_and_keeps_colour_order(tmp_path: Path, size, scale_255, shape) -> None:
    model = build_model(tmp_path / "lama.onnx", size=size, scale_255=scale_255)
    engine = LamaOnnxEngine(LamaOnnxRuntime(str(model)))
    crop, mask = crop_and_mask(*shape)
    result = engine.inpaint_image(crop, mask, InpaintingOptions())
    assert result.shape == crop.shape and result.dtype == np.uint8
    assert np.array_equal(result[mask == 0], crop[mask == 0])  # outside the hole: bit-identical
    # Away from the hole's edge (a scaled-down crop is scaled back up, which softens the edge itself).
    core = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    inside = result[core].astype(int)
    expected_bgr = np.array([0.8, 0.5, 0.2]) * 255  # the model paints RGB (0.2, 0.5, 0.8)
    assert np.abs(inside - expected_bgr).max() <= 2
    assert "LaMa" in engine.notes()[0]


def test_reports_missing_runtime_file_and_non_lama_models(tmp_path: Path) -> None:
    assert "wasn't found" in (LamaOnnxEngine(LamaOnnxRuntime(str(tmp_path / "none.onnx"))).unavailable_reason() or "")
    assert "No LaMa model" in (LamaOnnxEngine(LamaOnnxRuntime("")).unavailable_reason() or "")
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(EngineUnavailableError):
        LamaOnnxEngine(LamaOnnxRuntime(str(bad))).prepare()


def test_a_temporal_engine_with_lama_generates_only_what_no_frame_shows(tmp_path: Path) -> None:
    from tests.test_temporal_engine import (
        WINDOW_H,
        WINDOW_W,
        hole_psnr,
        make_scene,
        mark_mask,
        noisy,
        panning_clip,
        run,
        watermark,
    )

    model = build_model(tmp_path / "lama.onnx", size=512, scale_255=False)
    fallback = LamaOnnxEngine(LamaOnnxRuntime(str(model)))
    # Still camera: nothing to borrow, so everything is generated (and reported as such).
    scene = make_scene()
    window = np.ascontiguousarray(scene[30 : 30 + WINDOW_H, 100 : 100 + WINDOW_W])
    rng = np.random.default_rng(2)
    marked = [watermark(noisy(window, rng)) for _ in range(6)]
    masks = [mark_mask() for _ in range(6)]
    engine = TemporalReconstructionEngine(fallback=fallback, fallback_label="LaMa")
    result = run(engine, marked, masks)
    assert "100% was hidden in every frame and was generated by LaMa" in engine.notes()[0]
    core = cv2.erode(masks[0], np.ones((9, 9), np.uint8)) > 0
    for frame in result:  # the watermark is gone; what replaced it is the model's output
        assert frame[core].min() < 250  # no white logo box left
        # The stand-in model paints one flat colour. The patch is then made continuous with the real pixels at its edge, which
        # tilts that colour smoothly (a few levels per pixel at most), so what must hold is that the fill has no texture or edge of its own.
        grey = frame[..., 0].astype(int)
        both = core[:, 1:] & core[:, :-1]
        assert np.abs(np.diff(grey, axis=1))[both].max() <= 8  # a smooth ramp; a logo edge or texture would step by far more
        both = core[1:, :] & core[:-1, :]
        assert np.abs(np.diff(grey, axis=0))[both].max() <= 8  # a smooth ramp; a logo edge or texture would step by far more
    # Panning camera: the real pixels are used, LaMa is not needed for the middle of the clip.
    truth, marked, masks = panning_clip()
    engine = TemporalReconstructionEngine(fallback=fallback, fallback_label="LaMa")
    result = run(engine, marked, masks)
    assert min(hole_psnr(r, t, m) for r, t, m in zip(result[8:20], truth[8:20], masks[8:20], strict=True)) > 40
    assert "Nothing was generated" in engine.notes()[0] or "generated by LaMa" in engine.notes()[0]


# -- weights --------------------------------------------------------------------------------------------------


def test_a_present_file_is_checked_against_its_checksum(tmp_path: Path) -> None:
    path = tmp_path / "m.onnx"
    path.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    assert ensure_weights(str(path), sha256=good) == path
    assert sha256_of(path) == good
    with pytest.raises(EngineUnavailableError, match="checksum"):
        ensure_weights(str(path), sha256="0" * 64)


def test_downloads_verify_before_the_file_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    payload = b"model-bytes" * 1000
    good = hashlib.sha256(payload).hexdigest()

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    real_stream = httpx.stream

    def fake_stream(method, url, **kwargs):  # noqa: ANN001, ANN003
        client = httpx.Client(transport=httpx.MockTransport(transport))
        return client.stream(method, url)

    monkeypatch.setattr(httpx, "stream", fake_stream)
    target = tmp_path / "models" / "lama.onnx"
    assert ensure_weights(str(target), "https://example.test/lama.onnx", good) == target
    assert target.read_bytes() == payload
    assert not list(target.parent.glob("*.part"))

    bad_target = tmp_path / "models" / "other.onnx"
    with pytest.raises(EngineUnavailableError, match="checksum"):
        ensure_weights(str(bad_target), "https://example.test/lama.onnx", "1" * 64)
    assert not bad_target.exists() and not list(bad_target.parent.glob("*.part"))
    monkeypatch.setattr(httpx, "stream", real_stream)


def test_download_configuration_is_strict(tmp_path: Path) -> None:
    with pytest.raises(EngineUnavailableError, match="SHA256"):
        ensure_weights(str(tmp_path / "a.onnx"), "https://example.test/a.onnx", "")
    with pytest.raises(EngineUnavailableError, match="HTTPS"):
        ensure_weights(str(tmp_path / "a.onnx"), "http://example.test/a.onnx", "a" * 64)


def test_a_torchscript_model_in_a_pt_file_is_used_at_any_size_multiple_of_8_and_is_composited_like_the_onnx_one(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Paint(torch.nn.Module):
        def forward(self, image, mask):
            colour = torch.tensor([0.2, 0.5, 0.8]).view(1, 3, 1, 1)
            return image * (1 - mask) + colour * mask

    path = tmp_path / "big-lama.pt"
    torch.jit.script(Paint()).save(str(path))
    engine = LamaOnnxEngine(LamaOnnxRuntime(str(path)))
    assert engine.is_available() and engine.device().kind == "cpu"
    crop = np.full((70, 90, 3), 40, np.uint8)
    mask = np.zeros((70, 90), np.uint8)
    mask[20:40, 30:60] = 255
    out = engine.inpaint_image(crop, mask, InpaintingOptions())
    assert out.shape == crop.shape
    assert (out[mask == 0] == 40).all()  # outside the hole nothing changes
    assert abs(int(out[30, 45, 0]) - 204) <= 2 and abs(int(out[30, 45, 2]) - 51) <= 2  # BGR of RGB (0.2, 0.5, 0.8)
    assert "LaMa" in " ".join(engine.notes())


def test_a_pt_model_without_pytorch_says_how_to_get_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ai import lama

    path = tmp_path / "big-lama.pt"
    path.write_bytes(b"x")
    monkeypatch.setattr(lama, "torch_module", lambda: None)
    problem = LamaOnnxRuntime(str(path)).problem()
    assert problem is not None and "generative-fill" in problem


def test_the_generative_fill_gets_a_small_box_per_mark_not_the_whole_frame() -> None:
    from app.ai.temporal import GENERATION_CONTEXT_PX, _generation_boxes

    todo = np.zeros((1280, 720), np.uint8)
    todo[540:560, 520:590] = 255  # a small mark high up
    todo[1150:1210, 270:440] = 255  # a bigger one far below
    boxes = _generation_boxes(todo, InpaintingOptions())
    assert len(boxes) == 2
    assert sum(b.area for b in boxes) < 0.15 * todo.size
    for box in boxes:
        assert (todo[box.slices] > 0).any()
    assert boxes[0].y0 <= 540 - GENERATION_CONTEXT_PX + 1 and boxes[0].x1 >= 590 + GENERATION_CONTEXT_PX - 1
    todo[600:610, 520:590] = 255  # near the first: shares its box
    assert len(_generation_boxes(todo, InpaintingOptions())) == 2
