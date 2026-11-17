"""LaMa (Apache-2.0) through ONNX Runtime: a generative fill for areas no other frame can supply.

LaMa looks at the picture around a hole and makes up what plausibly belongs there. It is good at textures, edges and
repeating structure and is the commercially licensable option here (ProPainter, E2FGVI and MiniMax-Remover are
non-commercial). What it cannot do is recover what was really behind a mark: the result is invented detail, so every
render that used it says so.

Model contract (the public LaMa ONNX exports): inputs `image` [1,3,H,W] float32 RGB 0..1 and `mask` [1,1,H,W] float32
(1 = fill); one output [1,3,H,W], RGB, either 0..1 or 0..255 (detected). Some exports are fixed at 512x512, others
accept any size that is a multiple of 8; both work. The file comes from server configuration (AI_MODEL_PATH, or
AI_MODEL_URL with AI_MODEL_SHA256 for a verified download).

A `.pt` file is also accepted: the public `big-lama.pt` TorchScript model (same inputs and output, any size that is a multiple
of 8). It needs PyTorch (CPU is enough) instead of ONNX Runtime; `python run_local.py --generative-fill` sets both up.
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ai.base import (
    DeviceInfo,
    EngineUnavailableError,
    Frame,
    InpaintingError,
    InpaintingOptions,
    ModelMemoryExhausted,
    Mask,
)
from app.ai.image import ImageInpaintingEngine, check_result
from app.ai.weights import config_problem, ensure_weights

logger = logging.getLogger(__name__)

MAX_DYNAMIC_SIDE = 1024  # the largest piece (tile) given to the model at once
TILE_SIDES = (1024, 768, 512, 384)  # tried in turn when the model runs out of memory
TILE_OVERLAP = 64  # pixels two neighbouring tiles share, so the model always has context across the join
PAD_MULTIPLE = 8
MIB = 1024 * 1024


class OutOfModelMemory(InpaintingError):
    """The model ran out of (GPU or system) memory on a piece of this size; a smaller piece may fit."""


def _memory_message(width: int, height: int, device_kind: str) -> str:
    where = "GPU memory" if device_kind == "cuda" else "memory"
    return (
        f"Your video is {width} × {height}, but the processing worker doesn't have enough {where} to rebuild the marked area, "
        "even in small pieces. Mark a smaller area, or retry on a processing worker with more "
        f"{'GPU memory (VRAM)' if device_kind == 'cuda' else 'memory or a GPU'}. Nothing was lowered in quality to get around this."
    )


def plan_axis(length: int, side: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Cut [0, length) into pieces of at most `side` that overlap by `overlap`: (start, stop, commit_start, commit_stop).

    Each piece only *keeps* what it generated inside its commit range: the model is least reliable at the edge of what
    it was shown, so each handover happens in the middle of the overlap, where both pieces see the area well inside their view.
    The first piece keeps its leading edge and the last its trailing edge; the commit ranges tile [0, length) exactly."""
    if length <= side:
        return [(0, length, 0, length)]
    stride = side - overlap
    starts = list(range(0, length - side, stride)) + [length - side]
    # Where one piece hands over to the next: the middle of what they both see.
    handovers = [(starts[i + 1] + starts[i] + side) // 2 for i in range(len(starts) - 1)]
    pieces = []
    for index, start in enumerate(starts):
        commit_start = 0 if index == 0 else handovers[index - 1]
        commit_stop = length if index == len(starts) - 1 else handovers[index]
        pieces.append((start, start + side, commit_start, commit_stop))
    return pieces


def torch_module() -> Any | None:
    try:
        import torch
    except Exception:  # noqa: BLE001  ImportError, or OSError from a broken install
        return None
    return torch


def onnxruntime_module() -> Any | None:
    try:
        import onnxruntime
    except Exception:  # ImportError, or OSError from a broken runtime
        return None
    return onnxruntime


class LamaOnnxRuntime:
    """Loads the ONNX model once and runs it. Shared by anything that wants LaMa."""

    def __init__(self, model_path: str, url: str = "", sha256: str = "", cuda_setting: str = "auto", *, allow_insecure_loopback: bool = False) -> None:
        self.model_path = model_path
        self.url = url
        self.sha256 = sha256
        self.cuda_setting = cuda_setting
        self.allow_insecure_loopback = allow_insecure_loopback
        self._session: Any = None
        self._input_names: tuple[str, str] = ("image", "mask")
        self._fixed: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()  # one inference at a time: two at once only fight over the same cores / GPU memory
        self._warmed = False

    @property
    def uses_torch(self) -> bool:
        return self.model_path.lower().endswith(".pt")

    def problem(self) -> str | None:
        if self.uses_torch:
            if torch_module() is None:
                return "PyTorch isn't installed, so the .pt LaMa model can't be used (run `python run_local.py --generative-fill`)."
        elif onnxruntime_module() is None:
            return "ONNX Runtime isn't installed on this server (pip install onnxruntime)."
        if not self.model_path:
            return "No LaMa model is configured (AI_MODEL_PATH)."
        if Path(self.model_path).is_file():
            return None
        if self.url:
            return config_problem(self.url, self.sha256, allow_insecure_loopback=self.allow_insecure_loopback)
        return "The configured LaMa model file wasn't found."

    def _torch_device(self) -> str:
        torch = torch_module()
        if torch is not None and self.cuda_setting != "false" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def providers(self) -> list[str]:
        if self.uses_torch:
            return ["CUDAExecutionProvider" if self._torch_device() == "cuda" else "CPUExecutionProvider"]
        runtime = onnxruntime_module()
        available = runtime.get_available_providers() if runtime is not None else []
        if self.cuda_setting != "false" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def device_info(self) -> DeviceInfo:
        if self.providers()[0] == "CUDAExecutionProvider":
            return DeviceInfo("cuda", "GPU", "Running on the GPU.")
        return DeviceInfo("cpu", "CPU", "Running on the CPU: generating a fill takes a moment per frame.")

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            if self.uses_torch:
                self._load_torch()
                return
            runtime = onnxruntime_module()
            if runtime is None:
                raise EngineUnavailableError(self.problem() or "ONNX Runtime isn't available.")
            path = ensure_weights(self.model_path, self.url, self.sha256, allow_insecure_loopback=self.allow_insecure_loopback)
            try:
                session = runtime.InferenceSession(str(path), providers=self.providers())
            except Exception as error:
                logger.exception("couldn't load the LaMa model")
                raise EngineUnavailableError("The LaMa model couldn't be loaded. Check the model file.") from error
            inputs = session.get_inputs()
            if len(inputs) != 2 or len(session.get_outputs()) != 1:
                raise EngineUnavailableError("This isn't a LaMa model: it should take an image and a mask and return one image.")
            names = [i.name for i in inputs]
            image_input = next((i for i in inputs if len(i.shape) == 4 and i.shape[1] == 3), None)
            mask_input = next((i for i in inputs if len(i.shape) == 4 and i.shape[1] == 1), None)
            if image_input is None or mask_input is None:
                raise EngineUnavailableError("This isn't a LaMa model: it should take an image and a mask and return one image.")
            self._input_names = (image_input.name, mask_input.name)
            height, width = image_input.shape[2], image_input.shape[3]
            self._fixed = (height, width) if isinstance(height, int) and isinstance(width, int) else None
            logger.info("LaMa model loaded", extra={"inputs": names, "fixed": self._fixed, "provider": self.providers()[0]})
            self._session = session

    def _load_torch(self) -> None:
        torch = torch_module()
        if torch is None:
            raise EngineUnavailableError(self.problem() or "PyTorch isn't available.")
        path = ensure_weights(self.model_path, self.url, self.sha256, allow_insecure_loopback=self.allow_insecure_loopback)
        device = self._torch_device()
        try:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # torch.jit.load is deprecated but is how this public model is shipped
                model = torch.jit.load(str(path), map_location=device).eval()
        except Exception as error:
            logger.exception("couldn't load the LaMa model")
            raise EngineUnavailableError("The LaMa model couldn't be loaded. Check the model file.") from error
        self._fixed = None
        logger.info("LaMa model loaded", extra={"runtime": "torch", "provider": device})
        self._session = model

    def _run_torch(self, image: np.ndarray, hole: np.ndarray) -> np.ndarray:
        torch = torch_module()
        device = self._torch_device()
        try:
            with torch.no_grad():
                output = self._session(torch.from_numpy(image).to(device), torch.from_numpy(hole).to(device))
        except Exception as error:
            if "memory" in str(error).lower():
                self.release_cached_memory()
                raise OutOfModelMemory("The computer ran out of memory filling the area.") from error
            raise
        return output.detach().cpu().numpy()

    def release_cached_memory(self) -> None:
        torch = torch_module()
        if torch is not None and self.uses_torch and self._torch_device() == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001  a broken CUDA state must not hide the real error
                pass

    def free_gpu_mb(self) -> int | None:
        """Free GPU memory right now, or None when the model isn't running on a GPU (or it can't be read)."""
        torch = torch_module()
        if torch is None or not self.uses_torch or self._torch_device() != "cuda":
            return None
        try:
            return int(torch.cuda.mem_get_info()[0] // MIB)
        except Exception:  # noqa: BLE001
            return None

    def suggested_tile_side(self) -> int:
        """How big a piece to give the model at once. On a GPU it follows the free memory (about 2 GB per 768-px piece
        with headroom); on the CPU the largest, since a slower run still fits in system memory."""
        free = self.free_gpu_mb()
        if free is None:
            return TILE_SIDES[0]
        if free >= 8000:
            return 1024
        if free >= 4000:
            return 768
        if free >= 2000:
            return 512
        return 384

    def warm_up(self) -> float | None:
        """Run the model once on a small dummy picture so the first real frame doesn't pay for one-time set-up (kernel
        selection, memory allocation). Seconds it took, or None when it was already warm. Never raises: a failed
        warm-up only means the first frame is slower."""
        if self._warmed:
            return None
        import time

        started = time.perf_counter()
        try:
            self.load()
            size = self._fixed or (256, 256)
            picture = np.full((size[0], size[1], 3), 127, np.uint8)
            hole = np.zeros(size, np.uint8)
            hole[size[0] // 2 - 8 : size[0] // 2 + 8, size[1] // 2 - 8 : size[1] // 2 + 8] = 255
            self.run(picture, hole)
        except Exception:
            logger.warning("LaMa warm-up failed; the first frame will be slower", exc_info=True)
            return None
        self._warmed = True
        elapsed = time.perf_counter() - started
        logger.info("LaMa warmed up", extra={"seconds": round(elapsed, 2), "provider": self.providers()[0]})
        return elapsed

    @property
    def fixed_size(self) -> tuple[int, int] | None:
        return self._fixed

    def run(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image_rgb [H,W,3] uint8, mask [H,W] uint8 (255 = fill) -> [H,W,3] uint8 RGB, at the size given."""
        self.load()
        image = image_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        hole = (mask > 127).astype(np.float32)[None, None]
        try:
            with self._run_lock:
                if self.uses_torch:
                    output = self._run_torch(image, hole)
                else:
                    output = self._session.run(None, {self._input_names[0]: image, self._input_names[1]: hole})[0]
        except Exception as error:
            if "memory" in str(error).lower() or isinstance(error, MemoryError):
                self.release_cached_memory()
                raise OutOfModelMemory("The GPU ran out of memory filling the area.") from error
            raise
        result = np.asarray(output)[0].transpose(1, 2, 0)
        if float(result.max()) <= 2.0:  # 0..1 export
            result = result * 255.0
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)


class LamaOnnxEngine(ImageInpaintingEngine):
    provider = "lama"
    name = "LaMa (generative fill)"

    def __init__(self, runtime: LamaOnnxRuntime) -> None:
        self._runtime = runtime
        self._used = False
        self._tiled_fills = 0  # marked areas too big for one piece, filled in several at full resolution
        self._tile_count = 0
        self._tile_side = 0  # the smallest piece size that was used
        self._shrunk_fills = 0  # areas that had to be scaled down (only a fixed-size model does this)

    def is_available(self) -> bool:
        return self._runtime.problem() is None

    def unavailable_reason(self) -> str | None:
        return self._runtime.problem()

    def device(self) -> DeviceInfo:
        return self._runtime.device_info()

    def expectation(self) -> str:
        return "The marked area is filled in with plausible detail generated from its surroundings. " + self._runtime.device_info().note

    def prepare(self) -> None:
        self._runtime.load()

    def notes(self) -> list[str]:
        if not self._used:
            return []
        notes = ["The marked area was filled in by LaMa, which generates plausible detail; it can't recover what was really behind the mark."]
        if self._tiled_fills:
            notes.append(
                f"{self._tiled_fills} large marked areas were too big for the model to see at once, so they were filled in "
                f"{self._tile_count} overlapping pieces of up to {self._tile_side} px at the video's full resolution "
                "(nothing was shrunk). Each piece sees the already filled neighbour as context."
            )
        if self._shrunk_fills:
            notes.append(
                f"The model only accepts a fixed picture size, so {self._shrunk_fills} large areas were shrunk to fit it and "
                "scaled back up: the filled area is softer than the rest of the picture."
            )
        return notes

    def inpaint_image(self, crop: Frame, mask: Mask, options: InpaintingOptions) -> Frame:
        self._used = True
        height, width = crop.shape[:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        self._runtime.load()
        fixed = self._runtime.fixed_size
        if fixed is not None:
            filled = self._run_fixed(rgb, mask, fixed)
        else:
            filled = self._fill_in_pieces(rgb, mask)
        out = crop.copy()
        generated = cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)
        out[mask > 127] = generated[mask > 127]  # everything outside the hole stays exactly as it was
        return check_result(out, (height, width, 3))

    # -- sizes -----------------------------------------------------------------------------------------------

    def _run_fixed(self, rgb: np.ndarray, mask: Mask, size: tuple[int, int]) -> np.ndarray:
        target_h, target_w = size
        height, width = rgb.shape[:2]
        scale = min(1.0, target_h / height, target_w / width)  # shrink to fit, never enlarge
        if scale < 1.0:
            self._shrunk_fills += 1
            new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
            small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            small_mask = (cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
        else:
            small, small_mask = rgb, mask
        h, w = small.shape[:2]
        padded = cv2.copyMakeBorder(small, 0, target_h - h, 0, target_w - w, cv2.BORDER_REPLICATE)
        padded_mask = cv2.copyMakeBorder(small_mask, 0, target_h - h, 0, target_w - w, cv2.BORDER_CONSTANT, value=0)
        result = self._runtime.run(padded, padded_mask)[:h, :w]
        if scale < 1.0:
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC)
        return result

    def _fill_in_pieces(self, rgb: np.ndarray, mask: Mask) -> np.ndarray:
        """Fill with a model that accepts any size. Whole if it fits in one piece, otherwise in overlapping tiles at the
        picture's own resolution. If the model runs out of memory the piece size is reduced and the fill restarted; only
        when even the smallest piece doesn't fit does the job fail (with a message saying why)."""
        start = self._runtime.suggested_tile_side()
        sides = [side for side in TILE_SIDES if side <= start] or [TILE_SIDES[-1]]
        for position, side in enumerate(sides):
            try:
                return self._fill_with_side(rgb, mask, side)
            except OutOfModelMemory:
                logger.warning("LaMa ran out of memory with %d px pieces", side)
                if position == len(sides) - 1:
                    raise ModelMemoryExhausted(_memory_message(rgb.shape[1], rgb.shape[0], self._runtime.device_info().kind)) from None
        raise AssertionError("unreachable")  # pragma: no cover

    def _fill_with_side(self, rgb: np.ndarray, mask: Mask, side: int) -> np.ndarray:
        height, width = rgb.shape[:2]
        if max(height, width) <= side:
            return self._run_piece(rgb, mask)
        columns = plan_axis(width, side, TILE_OVERLAP)
        rows = plan_axis(height, side, TILE_OVERLAP)
        work = rgb.copy()
        remaining = mask > 127
        tiles = 0
        for y0, y1, cy0, cy1 in rows:
            for x0, x1, cx0, cx1 in columns:
                keep = np.zeros_like(remaining)
                keep[cy0:cy1, cx0:cx1] = True
                keep &= remaining
                if not keep[y0:y1, x0:x1].any():
                    continue  # nothing to fill in this piece's own part
                piece_mask = np.where(remaining[y0:y1, x0:x1], 255, 0).astype(np.uint8)
                generated = self._run_piece(work[y0:y1, x0:x1], piece_mask)
                commit = keep[y0:y1, x0:x1]
                region = work[y0:y1, x0:x1]
                region[commit] = generated[commit]
                remaining[y0:y1, x0:x1] &= ~commit
                tiles += 1
        if remaining.any():  # cannot happen (the commit ranges cover the whole picture), but never return a hole
            raise InpaintingError("The marked area couldn't be filled completely.")
        self._tiled_fills += 1
        self._tile_count += tiles
        self._tile_side = side if self._tile_side == 0 else min(self._tile_side, side)
        return work

    def _run_piece(self, rgb: np.ndarray, mask: Mask) -> np.ndarray:
        """One model call at the picture's own size (padded to a multiple of 8)."""
        h, w = rgb.shape[:2]
        pad_h, pad_w = (-h) % PAD_MULTIPLE, (-w) % PAD_MULTIPLE
        padded = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        padded_mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return self._runtime.run(padded, padded_mask)[:h, :w]
