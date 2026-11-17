"""A remote inpainting service reached over HTTP.

Wire contract (a service that wants to plug in implements this; an adapter for any provider's own API is a small
subclass overriding `_request`):

    POST {AI_API_URL}
    Authorization: Bearer {AI_API_KEY}
    multipart/form-data:  image (PNG), mask (PNG, 255 = remove), quality (fast|balanced|high)
    200 -> a PNG of the same size as `image`, with the marked area reconstructed

Only the cropped area around the marks (plus context) is sent, never whole frames. Because this does send picture
data to another service, `sends_frames_to_third_party` is True and the interface says so before processing.
The API key never appears in logs or in messages shown to people.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from app.ai.base import DeviceInfo, Frame, InpaintingError, InpaintingOptions, Mask
from app.ai.image import ImageInpaintingEngine, check_result

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def url_problem(url: str, *, allow_insecure_loopback: bool) -> str | None:
    """Why this URL can't be used, or None. Pictures must travel over HTTPS (plain HTTP only to this machine, in development)."""
    if not url:
        return "No remote service is configured."
    parts = urlsplit(url)
    if not parts.hostname or parts.username or parts.password:
        return "The remote service address isn't valid."
    if parts.scheme == "https":
        return None
    if parts.scheme == "http" and allow_insecure_loopback and parts.hostname in LOOPBACK_HOSTS:
        return None
    return "The remote service must use HTTPS."


class APIModelEngine(ImageInpaintingEngine):
    provider = "api"
    name = "Remote inpainting service"
    sends_frames_to_third_party = True

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        retries: int = 2,
        allow_insecure_loopback: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._retries = retries
        self._allow_insecure_loopback = allow_insecure_loopback
        self._sleep = sleep
        self._client: httpx.Client | None = None

    # -- availability -----------------------------------------------------------------------------------

    def _problem(self) -> str | None:
        if not self._api_key:
            return "The remote service isn't set up (no API key)."
        return url_problem(self._url, allow_insecure_loopback=self._allow_insecure_loopback)

    def is_available(self) -> bool:
        return self._problem() is None

    def unavailable_reason(self) -> str | None:
        return self._problem()

    def device(self) -> DeviceInfo:
        return DeviceInfo("remote", "Remote service", "Processing happens on a third-party service.")

    def expectation(self) -> str:
        return "Picture data around the marked areas is sent to a third-party service. Speed depends on that service."

    # -- running ----------------------------------------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                follow_redirects=False,  # a redirect could carry the key and pictures somewhere unvetted
                headers={"Authorization": f"Bearer {self._api_key}", "User-Agent": "CleanFrame/0.1"},
            )
        return self._client

    def inpaint_image(self, crop: Frame, mask: Mask, options: InpaintingOptions) -> Frame:
        ok_image, image_png = cv2.imencode(".png", crop)
        ok_mask, mask_png = cv2.imencode(".png", mask)
        if not (ok_image and ok_mask):
            raise InpaintingError("The picture couldn't be prepared for the remote service.")
        body = self._request(image_png.tobytes(), mask_png.tobytes(), options.quality)
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise InpaintingError("The remote service returned something that isn't a picture.")
        return check_result(decoded, crop.shape)

    def _request(self, image: bytes, mask: bytes, quality: str) -> bytes:
        files = {"image": ("image.png", image, "image/png"), "mask": ("mask.png", mask, "image/png")}
        for attempt in range(self._retries + 1):
            last = attempt == self._retries
            try:
                with self._http().stream("POST", self._url, files=files, data={"quality": quality}) as response:
                    if response.status_code == 200:
                        return self._read_body(response)
                    if response.status_code in (401, 403):
                        raise InpaintingError("The remote service rejected the API key.")
                    if response.status_code in RETRYABLE_STATUS and not last:
                        delay = self._retry_delay(response, attempt)
                    elif response.status_code == 429:
                        raise InpaintingError("The remote service is busy. Please try again in a few minutes.")
                    else:
                        logger.warning("remote inpainting returned %d", response.status_code)
                        raise InpaintingError("The remote service couldn't process this video. Please try again later.")
            except httpx.TimeoutException as error:
                if last:
                    raise InpaintingError("The remote service didn't respond in time.") from error
                delay = 0.5 * 2**attempt
            except httpx.HTTPError as error:
                logger.warning("remote inpainting request failed: %s", type(error).__name__)
                if last:
                    raise InpaintingError("The remote service couldn't be reached.") from error
                delay = 0.5 * 2**attempt
            self._sleep(delay)
        raise InpaintingError("The remote service couldn't be reached.")  # unreachable; keeps the type checker honest

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("retry-after", "")
        if header.isdigit():
            return min(float(header), 30.0)
        return 0.5 * 2**attempt

    @staticmethod
    def _read_body(response: httpx.Response) -> bytes:
        if not response.headers.get("content-type", "").startswith("image/"):
            raise InpaintingError("The remote service returned something that isn't a picture.")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise InpaintingError("The remote service returned a response that is too large.")
            chunks.append(chunk)
        return b"".join(chunks)
