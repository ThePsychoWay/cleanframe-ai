"""Local PyTorch models.

Models are loaded from a single file, so no model code has to be installed here and any architecture can be dropped
in: a `torch.export` program (`.pt2`, the current PyTorch format; export it with dynamic height and width) or a
TorchScript archive (any other extension, still supported although PyTorch has deprecated it). Only load files you
trust: they are deserialised like any other pickle-based format. The path comes from server configuration, never
from a request.

Contracts (all tensors float32, RGB, values 0..1; mask 1 = remove):

    image model:  forward(image [1,3,H,W], mask [1,1,H,W])        -> [1,3,H,W]
    video model:  forward(frames [1,T,3,H,W], masks [1,T,1,H,W])  -> [1,T,3,H,W]

H and W are multiples of 8 (crops are padded by reflection and trimmed afterwards).
"""

from __future__ import annotations

import logging
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
    Mask,
)
from app.ai.device import detect_device, torch_module
from app.ai.image import ImageInpaintingEngine, check_result
from app.ai.video import VideoInpaintingEngine

logger = logging.getLogger(__name__)

PAD_MULTIPLE = 8
OUT_OF_MEMORY = "The GPU ran out of memory. Try a smaller marked area, or a shorter clip size."


class _TorchModel:
    """Loads a TorchScript model once and runs it on the best device. Shared by the image and video engines."""

    def __init__(self, model_path: str, cuda_setting: str) -> None:
        self.model_path = model_path
        self.cuda_setting = cuda_setting
        self._model: Any = None
        self._device: Any = None

    # -- availability -----------------------------------------------------------------------------------

    def problem(self) -> str | None:
        if torch_module() is None:
            return "PyTorch isn't installed on this server."
        if not self.model_path:
            return "No model file is configured."
        if not Path(self.model_path).is_file():
            return "The configured model file wasn't found."
        return None

    def device_info(self) -> DeviceInfo:
        return detect_device(self.cuda_setting)

    # -- running ----------------------------------------------------------------------------------------

    def _load(self) -> tuple[Any, Any, Any]:
        torch = torch_module()
        assert torch is not None
        if self._model is None:
            device = torch.device(self.device_info().kind)
            try:
                if self.model_path.lower().endswith(".pt2"):
                    model = torch.export.load(self.model_path).module()
                    model.to(device)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)  # torch.jit is deprecated but still works
                        model = torch.jit.load(self.model_path, map_location=device)
                    model.eval()  # an exported program is already inference-only and has no eval()
            except Exception as error:
                logger.exception("couldn't load the inpainting model")
                raise EngineUnavailableError("The inpainting model couldn't be loaded. Check the model file.") from error
            self._model, self._device = model, device
        return torch, self._model, self._device

    def run(self, *tensors_cpu: Any) -> Any:
        torch, model, device = self._load()
        try:
            with torch.inference_mode():
                return model(*(t.to(device) for t in tensors_cpu)).float().clamp_(0.0, 1.0).cpu()
        except Exception as error:
            if "out of memory" in str(error).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise InpaintingError(OUT_OF_MEMORY) from error
            raise


def _pad(array: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    height, width = array.shape[:2]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        array = cv2.copyMakeBorder(array, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return array, pad_h, pad_w


def _image_to_tensor(crop: Frame) -> Any:
    torch = torch_module()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)


def _mask_to_tensor(mask: Mask) -> Any:
    torch = torch_module()
    return torch.from_numpy((mask > 127).astype(np.float32))[None]


def _tensor_to_image(tensor: Any) -> Frame:
    rgb = np.rint(tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class LocalModelEngine(ImageInpaintingEngine):
    provider = "local"
    name = "Local PyTorch model"
    input_multiple = PAD_MULTIPLE

    def __init__(self, model_path: str, cuda_setting: str = "auto") -> None:
        self._runtime = _TorchModel(model_path, cuda_setting)

    def is_available(self) -> bool:
        return self._runtime.problem() is None

    def unavailable_reason(self) -> str | None:
        return self._runtime.problem()

    def device(self) -> DeviceInfo:
        return self._runtime.device_info()

    def expectation(self) -> str:
        return self._runtime.device_info().note

    def inpaint_image(self, crop: Frame, mask: Mask, options: InpaintingOptions) -> Frame:
        padded_crop, pad_h, pad_w = _pad(crop, PAD_MULTIPLE)
        padded_mask, _, _ = _pad(mask, PAD_MULTIPLE)
        output = self._runtime.run(_image_to_tensor(padded_crop)[None], _mask_to_tensor(padded_mask)[None])
        result = _tensor_to_image(output[0])
        height, width = crop.shape[:2]
        return check_result(np.ascontiguousarray(result[:height, :width]), crop.shape)


class LocalVideoModelEngine(VideoInpaintingEngine):
    provider = "video"
    name = "Local PyTorch video model"

    def __init__(self, model_path: str, cuda_setting: str = "auto", clip_frames: int | None = None, overlap_frames: int | None = None) -> None:
        super().__init__(clip_frames, overlap_frames)
        self._runtime = _TorchModel(model_path, cuda_setting)

    def is_available(self) -> bool:
        return self._runtime.problem() is None

    def unavailable_reason(self) -> str | None:
        return self._runtime.problem()

    def device(self) -> DeviceInfo:
        return self._runtime.device_info()

    def expectation(self) -> str:
        return self._runtime.device_info().note

    def inpaint_clip(self, crops: list[Frame], masks: list[Mask], options: InpaintingOptions) -> list[Frame]:
        torch = torch_module()
        assert torch is not None
        height, width = crops[0].shape[:2]
        frames = torch.stack([_image_to_tensor(_pad(c, PAD_MULTIPLE)[0]) for c in crops])[None]  # 1,T,3,H,W
        mask_stack = torch.stack([_mask_to_tensor(_pad(m, PAD_MULTIPLE)[0]) for m in masks])[None]  # 1,T,1,H,W
        output = self._runtime.run(frames, mask_stack)[0]
        if output.shape != frames[0].shape:
            raise InpaintingError("The video model returned a clip of the wrong size.")
        return [np.ascontiguousarray(_tensor_to_image(t)[:height, :width]) for t in output]
