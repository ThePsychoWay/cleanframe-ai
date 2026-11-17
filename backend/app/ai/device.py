"""Which hardware will run the model, decided without ever crashing.

CUDA is used when PyTorch can use it. Everything else (no PyTorch, no GPU, a broken driver, CUDA switched off)
falls back to the CPU and says why, so the person is told what to expect instead of seeing a stack trace.
"""

from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from types import ModuleType

from app.ai.base import DeviceInfo

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def torch_module() -> ModuleType | None:
    """PyTorch if it is installed and importable, else None. Importing it is slow, so it happens once."""
    try:
        import torch
    except Exception:  # ImportError, or OSError from a broken CUDA install
        logger.info("PyTorch isn't available", exc_info=True)
        return None
    return torch


@lru_cache(maxsize=4)
def detect_device(cuda_setting: str = "auto") -> DeviceInfo:
    """cuda_setting: "auto" (use a GPU if there is one), "true" (want a GPU, still falls back), "false" (CPU)."""
    cpu_slow = "Running on the CPU: processing a long video can take a long time."
    if cuda_setting == "false":
        return DeviceInfo("cpu", "CPU", "GPU use is switched off. " + cpu_slow)

    torch = torch_module()
    if torch is None:
        return DeviceInfo("cpu", "CPU", "PyTorch isn't installed here. " + cpu_slow)
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return DeviceInfo("cuda", name, f"Running on the GPU ({name}).")
    except Exception:
        logger.warning("CUDA check failed; using the CPU", exc_info=True)
        return DeviceInfo("cpu", "CPU", "The GPU couldn't be initialised. " + cpu_slow)

    if cuda_setting == "true":
        why = "A GPU was requested but CUDA isn't available."
    elif shutil.which("nvidia-smi"):
        why = "An NVIDIA GPU was found, but this PyTorch build can't use it."
    else:
        why = "No NVIDIA GPU was found."
    return DeviceInfo("cpu", "CPU", f"{why} {cpu_slow}")
