import { detectStaticOverlays, type DetectOutcome, type GrayFrame } from "./detect";

/** Thrown when this browser or this setup can't read the picture's pixels. The message is shown to the person as is. */
export class DetectionUnavailableError extends Error {}

const SAMPLE_WIDTH = 120;
const SAMPLE_COUNT = 12;
const STEP_TIMEOUT_MS = 8000;

function once(target: EventTarget, type: string, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => done(new DetectionUnavailableError("The preview video took too long to respond.")), STEP_TIMEOUT_MS);
    const onAbort = () => done(new DOMException("Cancelled", "AbortError"));
    function done(error?: Error) {
      window.clearTimeout(timer);
      target.removeEventListener(type, onType);
      target.removeEventListener("error", onError);
      signal.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else resolve();
    }
    const onType = () => done();
    const onError = () => done(new DetectionUnavailableError("The preview video couldn't be read for detection."));
    target.addEventListener(type, onType, { once: true });
    target.addEventListener("error", onError, { once: true });
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

/** Reads a few evenly spaced frames of the preview video as small grey images. */
export async function sampleFrames(url: string, signal: AbortSignal): Promise<GrayFrame[]> {
  const video = document.createElement("video");
  video.crossOrigin = "anonymous"; // without this the browser refuses to let a page read the pixels
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = url;
  try {
    if (video.readyState < 1) await once(video, "loadedmetadata", signal);
    const { videoWidth, videoHeight, duration } = video;
    if (!videoWidth || !videoHeight || !Number.isFinite(duration) || duration <= 0) {
      throw new DetectionUnavailableError("The preview video has no readable picture.");
    }
    const width = SAMPLE_WIDTH;
    const height = Math.max(8, Math.round((SAMPLE_WIDTH * videoHeight) / videoWidth));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new DetectionUnavailableError("This browser can't read video frames.");

    const frames: GrayFrame[] = [];
    for (let i = 0; i < SAMPLE_COUNT; i++) {
      const at = ((i + 0.5) / SAMPLE_COUNT) * duration;
      const seeked = once(video, "seeked", signal);
      video.currentTime = at;
      await seeked;
      ctx.drawImage(video, 0, 0, width, height);
      let pixels: ImageData;
      try {
        pixels = ctx.getImageData(0, 0, width, height);
      } catch {
        throw new DetectionUnavailableError("The browser blocked reading this video's pixels, so automatic detection isn't available here.");
      }
      const gray = new Uint8Array(width * height);
      for (let p = 0; p < gray.length; p++) {
        const o = p * 4;
        gray[p] = (pixels.data[o] * 299 + pixels.data[o + 1] * 587 + pixels.data[o + 2] * 114) / 1000;
      }
      frames.push({ width, height, data: gray });
    }
    return frames;
  } finally {
    video.removeAttribute("src");
    video.load();
  }
}

export async function detectInVideo(url: string, signal: AbortSignal): Promise<DetectOutcome> {
  return detectStaticOverlays(await sampleFrames(url, signal));
}
