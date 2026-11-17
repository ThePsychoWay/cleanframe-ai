"""Chooses the inpainting engine from configuration (AI_PROVIDER).

If the configured engine can't run, asking for it raises EngineUnavailableError with a message saying what is
missing; the app never quietly substitutes a weaker engine and presents the result as if it were the configured one.
(`auto` is not a substitution: it is one engine whose documented order of methods is decided from the video and
reported in the render's notes.) The development engine is only available when
DEV_MODE is on.
"""

from __future__ import annotations

import threading

from app.ai.auto import AutoRemovalEngine
from app.ai.base import DeviceInfo, EngineInfo, EngineUnavailableError, InpaintingEngine
from app.ai.dev import DevInpaintingEngine
from app.ai.lama import LamaOnnxEngine, LamaOnnxRuntime
from app.ai.local import LocalModelEngine, LocalVideoModelEngine
from app.ai.overlay import OverlayInversionEngine
from app.ai.remote import APIModelEngine
from app.ai.temporal import TemporalReconstructionEngine
from app.config import Settings, get_settings

PROVIDERS = ("dev", "auto", "overlay", "reconstruct", "lama", "local", "video", "api")


_runtimes: dict[tuple, LamaOnnxRuntime] = {}
_runtimes_lock = threading.Lock()


def _lama_runtime(settings: Settings) -> LamaOnnxRuntime:
    """The LaMa runtime for these settings. One is kept for the life of the process so the model is loaded (and warmed up)
    once, not for every video; with AI_KEEP_MODEL_LOADED=false each call gets its own, which is freed after the video."""
    key = (settings.ai_model_path, settings.ai_model_url, settings.ai_model_sha256, settings.cuda_enabled, settings.dev_mode)

    def build() -> LamaOnnxRuntime:
        return LamaOnnxRuntime(
            settings.ai_model_path,
            settings.ai_model_url,
            settings.ai_model_sha256,
            settings.cuda_enabled,
            allow_insecure_loopback=settings.dev_mode,
        )

    if not settings.ai_keep_model_loaded:
        return build()
    with _runtimes_lock:
        if key not in _runtimes:
            _runtimes[key] = build()
        return _runtimes[key]


def _lama(settings: Settings) -> LamaOnnxEngine:
    return LamaOnnxEngine(_lama_runtime(settings))


def warm_up(settings: Settings | None = None) -> float | None:
    """Load the generative model and run it once, if this configuration uses one and its weights are already on disk
    (nothing is downloaded here). Seconds spent, or None when there was nothing to do. Never raises."""
    from pathlib import Path

    settings = settings or get_settings()
    if not settings.ai_keep_model_loaded or not settings.ai_model_path or settings.ai_provider not in ("auto", "reconstruct", "lama"):
        return None
    if not Path(settings.ai_model_path).is_file():
        return None
    try:
        return _lama_runtime(settings).warm_up()
    except Exception:
        return None


def create_engine(settings: Settings | None = None) -> InpaintingEngine:
    """The configured engine, constructed but not necessarily available (check is_available())."""
    settings = settings or get_settings()
    provider = settings.ai_provider
    if provider == "dev":
        return DevInpaintingEngine()
    if provider == "overlay":
        return OverlayInversionEngine(memory_mb=settings.overlay_memory_mb)
    if provider in ("reconstruct", "auto"):
        wants_lama = settings.reconstruct_fallback == "lama" or (
            settings.reconstruct_fallback == "auto" and bool(settings.ai_model_path)
        )
        fallback = _lama(settings) if wants_lama else None
        if provider == "auto":
            return AutoRemovalEngine(
                fallback=fallback, fallback_label="LaMa", memory_mb=settings.overlay_memory_mb, search_memory_mb=settings.temporal_memory_mb
            )
        return TemporalReconstructionEngine(fallback=fallback, fallback_label="LaMa", memory_mb=settings.temporal_memory_mb)
    if provider == "lama":
        return _lama(settings)
    if provider == "local":
        return LocalModelEngine(settings.ai_model_path, settings.cuda_enabled)
    if provider == "video":
        return LocalVideoModelEngine(
            settings.ai_model_path,
            settings.cuda_enabled,
            clip_frames=settings.ai_video_clip_frames,
            overlap_frames=settings.ai_video_overlap_frames,
        )
    if provider == "api":
        return APIModelEngine(
            settings.ai_api_url,
            settings.ai_api_key,
            timeout_seconds=settings.ai_api_timeout_seconds,
            allow_insecure_loopback=settings.dev_mode,
        )
    raise EngineUnavailableError(f"Unknown inpainting provider: {provider!r}.")


def get_engine(settings: Settings | None = None) -> InpaintingEngine:
    """The configured engine, ready to use. Raises EngineUnavailableError with the reason if it can't run."""
    settings = settings or get_settings()
    if settings.ai_provider == "dev" and not settings.dev_mode:
        raise EngineUnavailableError(
            "The development inpainting engine is disabled in production. Configure a real engine."
        )
    engine = create_engine(settings)
    engine.require_available()
    return engine


def describe_engine(settings: Settings | None = None) -> EngineInfo:
    """What the app can tell people about the configured engine. Never raises."""
    settings = settings or get_settings()
    try:
        if settings.ai_provider == "dev" and not settings.dev_mode:
            engine = create_engine(settings)
            return EngineInfo(
                provider=engine.provider,
                name=engine.name,
                available=False,
                development_only=True,
                sends_frames_to_third_party=False,
                device=None,
                reason="The development inpainting engine is disabled in production.",
                expectation="",
            )
        return create_engine(settings).info()
    except Exception:
        return EngineInfo(
            provider=settings.ai_provider,
            name="Inpainting engine",
            available=False,
            development_only=False,
            sends_frames_to_third_party=False,
            device=DeviceInfo("cpu", "CPU", ""),
            reason="The inpainting engine isn't set up correctly.",
            expectation="",
        )
