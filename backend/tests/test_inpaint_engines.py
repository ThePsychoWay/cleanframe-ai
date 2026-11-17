"""Engine adapters (PyTorch, remote API, clip-based video), the registry, device detection and the status API."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.ai import device as device_module
from app.ai.base import EngineUnavailableError, InpaintingError, InpaintingOptions
from app.ai.dev import DevInpaintingEngine
from app.ai.local import LocalModelEngine, LocalVideoModelEngine
from app.ai.registry import create_engine, describe_engine, get_engine
from app.ai.remote import APIModelEngine, url_problem
from app.ai.roi import Roi
from app.ai.video import VideoInpaintingEngine, _crossfade, _Result
from app.config import ConfigError, Settings, get_settings
from tests.conftest import TEST_SECRET
from tests.inpaint_scene import make_scene, mean_error

OFF = InpaintingOptions(temporal_smoothing=False)
SECRET_KEY = "sk-live-very-secret-key-1234"


def interior(scene, shrink: int = 7) -> np.ndarray:
    return cv2.erode(scene.masks[0], np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8)) > 0


# =========================================================================================================
# PyTorch models (a real TorchScript model, run for real)
# =========================================================================================================

torch = pytest.importorskip("torch", reason="PyTorch isn't installed")


class FillWithMean(torch.nn.Module):
    """A tiny image 'model': paints the masked pixels with the mean colour of the unmasked ones."""

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keep = 1.0 - mask
        mean = (image * keep).sum(dim=(2, 3), keepdim=True) / keep.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        return image * keep + mean * mask


class FillWithClipMean(torch.nn.Module):
    """A tiny video 'model': masked pixels get the mean colour of the unmasked pixels of the whole clip."""

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        keep = 1.0 - masks
        mean = (frames * keep).sum(dim=(1, 3, 4), keepdim=True) / keep.sum(dim=(1, 3, 4), keepdim=True).clamp(min=1.0)
        return frames * keep + mean * masks


@pytest.fixture(scope="module")
def image_model(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("models") / "image.pt"
    torch.jit.script(FillWithMean()).save(str(path))
    return str(path)


@pytest.fixture(scope="module")
def video_model(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("models") / "video.pt"
    torch.jit.script(FillWithClipMean()).save(str(path))
    return str(path)


def test_a_torch_export_model_works_at_any_crop_size(tmp_path):
    height, width = torch.export.Dim("height", min=8, max=4096), torch.export.Dim("width", min=8, max=4096)
    example = (torch.rand(1, 3, 64, 96), torch.rand(1, 1, 64, 96))
    program = torch.export.export(FillWithMean(), example, dynamic_shapes={"image": {2: height, 3: width}, "mask": {2: height, 3: width}})
    path = tmp_path / "image.pt2"
    torch.export.save(program, str(path))
    engine = LocalModelEngine(str(path), "false")
    scene = make_scene(frames=2, boxes=[(220, 16, 300, 48), (10, 120, 70, 170)])  # two crops of different sizes
    out = list(engine.process(scene.marked, scene.masks, OFF))
    assert mean_error(out, scene.clean, scene.region) < mean_error(scene.marked, scene.clean, scene.region)


def test_a_torchscript_image_model_is_run_and_its_output_blended_back(image_model):
    engine = LocalModelEngine(image_model, "false")
    assert engine.is_available() and not engine.info().development_only
    scene = make_scene(frames=3)
    out = list(engine.process(scene.marked, scene.masks, OFF))
    assert len(out) == 3
    centre = interior(scene)
    for frame in out:
        pixels = frame[centre].astype(int)
        assert pixels.std(axis=0).max() <= 1  # the model painted one flat colour
        assert 60 < pixels.mean() < 170  # a plausible background colour, not the watermark's red
    near = cv2.dilate(scene.masks[0], np.ones((13, 13), np.uint8)) > 0
    assert np.array_equal(out[0][~near], scene.marked[0][~near])


def test_the_model_works_on_frames_that_are_not_multiples_of_eight(image_model):
    scene = make_scene(width=333, height=187, frames=2, boxes=[(280, 150, 333, 187)], mask_margin=0)
    out = list(LocalModelEngine(image_model, "false").process(scene.marked, scene.masks, OFF))
    assert out[0].shape == scene.marked[0].shape
    assert not np.array_equal(out[0][scene.region], scene.marked[0][scene.region])


def test_device_notes_tell_the_person_what_to_expect(image_model):
    note = LocalModelEngine(image_model, "false").info().device
    assert note.kind == "cpu" and "CPU" in note.note


@pytest.mark.parametrize(
    ("path", "words"),
    [("", "No model file"), ("/definitely/not/here.pt", "wasn't found")],
)
def test_a_missing_model_is_reported_without_revealing_paths(path, words):
    engine = LocalModelEngine(path, "false")
    info = engine.info()
    assert not info.available and words in (info.reason or "")
    assert "/definitely" not in (info.reason or "")
    scene = make_scene(frames=1)
    with pytest.raises(EngineUnavailableError):
        list(engine.process(scene.marked, scene.masks, OFF))


def test_a_corrupt_model_file_is_a_clear_error_not_a_crash(tmp_path):
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"this is not a model")
    scene = make_scene(frames=1)
    with pytest.raises(EngineUnavailableError, match="couldn't be loaded"):
        list(LocalModelEngine(str(broken), "false").process(scene.marked, scene.masks, OFF))


def test_a_model_returning_the_wrong_shape_is_caught(tmp_path):
    class Wrong(torch.nn.Module):
        def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return image[:, :, :-8, :]

    path = tmp_path / "wrong.pt"
    torch.jit.script(Wrong()).save(str(path))
    scene = make_scene(frames=1)
    with pytest.raises(InpaintingError):
        list(LocalModelEngine(str(path), "false").process(scene.marked, scene.masks, OFF))


def test_running_out_of_gpu_memory_is_reported_in_plain_words(image_model, monkeypatch):
    engine = LocalModelEngine(image_model, "false")
    runtime = engine._runtime
    _, model, _ = runtime._load()

    def boom(*args):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(runtime, "_model", boom)
    scene = make_scene(frames=1)
    with pytest.raises(InpaintingError, match="ran out of memory"):
        list(engine.process(scene.marked, scene.masks, OFF))


def test_a_torchscript_video_model_processes_a_whole_clip(video_model):
    engine = LocalVideoModelEngine(video_model, "false", clip_frames=8, overlap_frames=2)
    scene = make_scene(frames=20)
    out = list(engine.process(scene.marked, scene.masks, OFF))
    assert len(out) == 20
    centre = interior(scene)
    colours = np.array([f[centre].reshape(-1, 3).mean(axis=0) for f in out])
    assert colours.std(axis=0).max() < 6  # steady from frame to frame, including across window joins
    assert mean_error(out, scene.clean, scene.region) < mean_error(scene.marked, scene.clean, scene.region)


# =========================================================================================================
# Clip windows (structure, with a recording engine)
# =========================================================================================================


class Recorder(VideoInpaintingEngine):
    provider = "video"
    name = "recorder"

    def __init__(self, clip_frames: int, overlap_frames: int) -> None:
        super().__init__(clip_frames, overlap_frames)
        self.windows: list[tuple[int, tuple[int, ...]]] = []

    def is_available(self) -> bool:
        return True

    def inpaint_clip(self, crops, masks, options):
        self.windows.append((len(crops), crops[0].shape))
        return [c.copy() for c in crops]


def _numbered(scene) -> list[np.ndarray]:
    """Frames that carry their own index in a corner pixel far from the marks."""
    frames = []
    for t, frame in enumerate(scene.marked):
        frame = frame.copy()
        frame[0, 0] = (t, t, t)
        frames.append(frame)
    return frames


@pytest.mark.parametrize(
    ("count", "windows"),
    [(1, [1]), (5, [5]), (8, [8]), (9, [8, 3]), (17, [8, 8, 5]), (20, [8, 8, 8]), (21, [8, 8, 8, 3])],
)
def test_clips_are_cut_into_overlapping_windows_without_losing_or_repeating_frames(count, windows):
    scene = make_scene(frames=count, boxes=[(250, 100, 300, 140)])
    engine = Recorder(clip_frames=8, overlap_frames=2)
    out = list(engine.process(_numbered(scene), scene.masks, OFF))
    assert [int(f[0, 0, 0]) for f in out] == list(range(count))
    assert [length for length, _ in engine.windows] == windows
    assert len({shape for _, shape in engine.windows}) == 1  # one crop size for the whole video


def test_windows_are_validated():
    with pytest.raises(ValueError):
        Recorder(clip_frames=4, overlap_frames=2)


def test_a_clip_without_any_marks_never_calls_the_model():
    scene = make_scene(frames=10)
    engine = Recorder(8, 2)
    out = list(engine.process(scene.marked, [np.zeros_like(m) for m in scene.masks], OFF))
    assert engine.windows == [] and len(out) == 10


def test_where_two_windows_overlap_their_results_are_cross_faded():
    box = Roi(0, 0, 4, 4)
    old = _Result(np.full((4, 4, 3), 100, np.uint8), box)
    new = _Result(np.full((4, 4, 3), 200, np.uint8), box)
    assert int(_crossfade(old, new, 0.25).frame[0, 0, 0]) == 125
    assert int(_crossfade(old, new, 0.75).frame[2, 2, 0]) == 175
    assert _crossfade(_Result(old.frame, None), new, 0.5) is new


# =========================================================================================================
# Remote service (a real HTTP server on this machine, speaking the documented contract)
# =========================================================================================================


class FakeService:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.fail_with: list[int] = []  # statuses returned before succeeding
        self.always: int | None = None
        self.wrong_size = False
        self.not_image = False
        self.sleep = 0.0
        self.redirect_to: str | None = None
        self.retry_after: str | None = None

        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                message = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True) for p in message.iter_parts()}
                image = cv2.imdecode(np.frombuffer(parts["image"], np.uint8), cv2.IMREAD_COLOR)
                mask = cv2.imdecode(np.frombuffer(parts["mask"], np.uint8), cv2.IMREAD_GRAYSCALE)
                service.requests.append(
                    {"auth": self.headers.get("Authorization"), "image": image.shape, "mask": mask.shape, "quality": parts["quality"].decode()}
                )
                if service.sleep:
                    time.sleep(service.sleep)
                if service.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", service.redirect_to)
                    self.end_headers()
                    return
                status = service.fail_with.pop(0) if service.fail_with else service.always
                if status:
                    self.send_response(status)
                    if service.retry_after:
                        self.send_header("Retry-After", service.retry_after)
                    self.end_headers()
                    return
                result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
                if service.wrong_size:
                    result = result[:-4]
                payload = cv2.imencode(".png", result)[1].tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain" if service.not_image else "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}/inpaint"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def service(monkeypatch) -> Iterator[FakeService]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")  # the sandbox's proxy must not intercept loopback
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    fake = FakeService()
    yield fake
    fake.stop()


def remote(service: FakeService, **kwargs) -> APIModelEngine:
    waits: list[float] = []
    kwargs.setdefault("sleep", waits.append)
    engine = APIModelEngine(service.url, SECRET_KEY, allow_insecure_loopback=True, **kwargs)
    engine.waits = waits  # type: ignore[attr-defined]
    return engine


def test_the_remote_engine_sends_only_crops_with_the_key_and_gets_a_real_result(service):
    engine = remote(service)
    info = engine.info()
    assert info.available and info.sends_frames_to_third_party and info.device.kind == "remote"
    scene = make_scene(width=640, height=360, frames=3, boxes=[(540, 20, 620, 56)])
    out = list(engine.process(scene.marked, scene.masks, InpaintingOptions(temporal_smoothing=False, quality="high")))
    assert len(service.requests) == 3
    for seen in service.requests:
        assert seen["auth"] == f"Bearer {SECRET_KEY}" and seen["quality"] == "high"
        assert seen["image"][0] * seen["image"][1] < 0.5 * 640 * 360  # a crop, not the frame
        assert seen["image"][:2] == seen["mask"]
    assert mean_error(out, scene.clean, scene.region) < 0.5 * mean_error(scene.marked, scene.clean, scene.region)
    assert engine._client is None  # connections are released when processing ends


def test_a_temporary_failure_is_retried_with_backoff_and_then_succeeds(service):
    service.fail_with = [503, 502]
    engine = remote(service, retries=2)
    scene = make_scene(frames=1)
    out = list(engine.process(scene.marked, scene.masks, OFF))
    assert len(service.requests) == 3 and len(out) == 1
    assert engine.waits == [0.5, 1.0]


def test_retry_after_is_respected_but_capped(service):
    service.fail_with = [429]
    service.retry_after = "3"
    engine = remote(service, retries=1)
    scene = make_scene(frames=1)
    list(engine.process(scene.marked, scene.masks, OFF))
    assert engine.waits == [3.0]
    service.fail_with, service.retry_after = [429], "9999"
    engine = remote(service, retries=1)
    list(engine.process(scene.marked, scene.masks, OFF))
    assert engine.waits == [30.0]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "rejected the API key"),
        (403, "rejected the API key"),
        (400, "couldn't process this video"),
        (500, "couldn't process this video"),
        (429, "is busy"),
    ],
)
def test_failures_become_readable_messages_that_never_contain_the_key_or_address(service, status, message):
    service.always = status
    engine = remote(service, retries=1)
    scene = make_scene(frames=1)
    with pytest.raises(InpaintingError, match=message) as caught:
        list(engine.process(scene.marked, scene.masks, OFF))
    text = str(caught.value)
    assert SECRET_KEY not in text and "127.0.0.1" not in text


def test_a_service_that_does_not_answer_in_time_is_reported(service):
    service.sleep = 1.0
    engine = remote(service, retries=0, timeout_seconds=0.3)
    scene = make_scene(frames=1)
    with pytest.raises(InpaintingError, match="didn't respond in time"):
        list(engine.process(scene.marked, scene.masks, OFF))


def test_an_unreachable_service_is_reported_not_crashed():
    engine = APIModelEngine("http://127.0.0.1:9/inpaint", SECRET_KEY, allow_insecure_loopback=True, retries=0, sleep=lambda s: None)
    scene = make_scene(frames=1)
    with pytest.raises(InpaintingError, match="couldn't be reached"):
        list(engine.process(scene.marked, scene.masks, OFF))


def test_redirects_are_not_followed(service):
    other = FakeService()
    try:
        service.redirect_to = other.url
        engine = remote(service, retries=0)
        scene = make_scene(frames=1)
        with pytest.raises(InpaintingError):
            list(engine.process(scene.marked, scene.masks, OFF))
        assert other.requests == []  # neither the key nor any picture went to the redirect target
    finally:
        other.stop()


@pytest.mark.parametrize("flag", ["wrong_size", "not_image"])
def test_a_response_that_is_not_the_expected_picture_is_rejected(service, flag):
    setattr(service, flag, True)
    engine = remote(service, retries=0)
    scene = make_scene(frames=1)
    with pytest.raises(InpaintingError):
        list(engine.process(scene.marked, scene.masks, OFF))


@pytest.mark.parametrize(
    ("url", "loopback", "ok"),
    [
        ("https://api.example.com/v1/inpaint", False, True),
        ("http://api.example.com/inpaint", False, False),  # pictures never travel in the clear
        ("http://api.example.com/inpaint", True, False),  # even in development, only to this machine
        ("http://localhost:9000/inpaint", False, False),
        ("http://localhost:9000/inpaint", True, True),
        ("https://user:pass@api.example.com/x", False, False),  # no credentials in the address
        ("ftp://api.example.com/x", False, False),
        ("not a url", False, False),
        ("", False, False),
    ],
)
def test_only_safe_addresses_are_accepted(url, loopback, ok):
    assert (url_problem(url, allow_insecure_loopback=loopback) is None) is ok


def test_a_remote_engine_without_a_key_is_unavailable():
    engine = APIModelEngine("https://api.example.com/x", "")
    assert not engine.is_available() and "no API key" in engine.unavailable_reason()


# =========================================================================================================
# Registry, configuration, device detection and the status API
# =========================================================================================================


def settings(env: pytest.MonkeyPatch, **values: str) -> Settings:
    env.setenv("DATABASE_URL", "sqlite+pysqlite:///x.db")
    env.setenv("JWT_SECRET", TEST_SECRET)
    for key, value in values.items():
        env.setenv(key.upper(), value)
    get_settings.cache_clear()
    return Settings()


def test_the_development_engine_is_only_available_in_development(env):
    dev = settings(env, ai_provider="dev", dev_mode="true")
    assert isinstance(get_engine(dev), DevInpaintingEngine)
    production = settings(env, ai_provider="dev", dev_mode="false", rate_limit_enabled="true")  # isolate the engine rule
    with pytest.raises(EngineUnavailableError, match="disabled in production"):
        get_engine(production)
    info = describe_engine(production)
    assert not info.available and info.development_only
    with pytest.raises(ConfigError, match="AI_PROVIDER=dev"):
        production.validate_runtime()


def test_each_provider_builds_the_matching_engine_and_never_falls_back(env):
    assert isinstance(create_engine(settings(env, ai_provider="local")), LocalModelEngine)
    assert isinstance(create_engine(settings(env, ai_provider="video")), LocalVideoModelEngine)
    assert isinstance(create_engine(settings(env, ai_provider="api")), APIModelEngine)
    for provider in ("local", "video", "api"):
        configured = settings(env, ai_provider=provider, dev_mode="true", ai_model_path="", ai_api_url="", ai_api_key="")
        with pytest.raises(EngineUnavailableError):
            get_engine(configured)  # not set up: an error, not a quiet switch to the development engine


def test_a_configured_local_model_is_picked_up(env, image_model):
    configured = settings(env, ai_provider="local", ai_model_path=image_model, cuda_enabled="false")
    engine = get_engine(configured)
    assert isinstance(engine, LocalModelEngine) and engine.is_available()


def test_the_video_engine_takes_its_window_size_from_settings(env, video_model):
    configured = settings(env, ai_provider="video", ai_model_path=video_model, ai_video_clip_frames="12", ai_video_overlap_frames="3")
    engine = get_engine(configured)
    assert (engine.clip_frames, engine.overlap_frames) == (12, 3)


@pytest.fixture
def fresh_device_cache() -> Iterator[None]:
    device_module.detect_device.cache_clear()
    yield
    device_module.detect_device.cache_clear()


def _fake_torch(*, cuda: bool, broken: bool = False):
    def is_available():
        if broken:
            raise RuntimeError("driver/library version mismatch")
        return cuda

    return SimpleNamespace(cuda=SimpleNamespace(is_available=is_available, get_device_name=lambda index: "NVIDIA Test GPU"))


def test_a_gpu_is_used_when_there_is_one(monkeypatch, fresh_device_cache):
    monkeypatch.setattr(device_module, "torch_module", lambda: _fake_torch(cuda=True))
    info = device_module.detect_device("auto")
    assert (info.kind, info.name) == ("cuda", "NVIDIA Test GPU")
    assert device_module.detect_device("false").kind == "cpu"  # unless the operator switched it off


@pytest.mark.parametrize(
    ("setting", "torch_object", "words"),
    [
        ("auto", None, "isn't installed"),
        ("auto", _fake_torch(cuda=False), "CPU"),
        ("true", _fake_torch(cuda=False), "requested"),
        ("auto", _fake_torch(cuda=False, broken=True), "couldn't be initialised"),
        ("false", _fake_torch(cuda=True), "switched off"),
    ],
)
def test_without_cuda_it_falls_back_to_the_cpu_and_explains_why(monkeypatch, fresh_device_cache, setting, torch_object, words):
    monkeypatch.setattr(device_module, "torch_module", lambda: torch_object)
    info = device_module.detect_device(setting)
    assert info.kind == "cpu" and words in info.note and "can take a long time" in info.note


def test_the_engine_status_needs_a_signed_in_user(client):
    assert client.get("/api/engine").status_code == 401


def test_the_status_api_labels_the_development_engine(user_client):
    body = user_client.get("/api/engine").json()
    assert body["provider"] == "dev" and body["available"] is True
    assert body["development_only"] is True and body["sends_frames_to_third_party"] is False
    assert body["device"]["kind"] == "cpu" and "Not for final results" in body["expectation"]


def test_the_status_api_never_reveals_paths_addresses_or_keys(user_client, env):
    env.setenv("AI_PROVIDER", "local")
    env.setenv("AI_MODEL_PATH", "/srv/secret-models/model.pt")
    get_settings.cache_clear()
    body = user_client.get("/api/engine").json()
    assert body["available"] is False and body["reason"] == "The configured model file wasn't found."
    assert "secret-models" not in json.dumps(body)

    env.setenv("AI_PROVIDER", "api")
    env.setenv("AI_API_URL", "https://inpaint.internal.example/v1")
    env.setenv("AI_API_KEY", SECRET_KEY)
    get_settings.cache_clear()
    body = user_client.get("/api/engine").json()
    assert body["available"] is True and body["sends_frames_to_third_party"] is True
    assert SECRET_KEY not in json.dumps(body) and "internal.example" not in json.dumps(body)
