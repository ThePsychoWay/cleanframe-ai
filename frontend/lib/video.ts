/**
 * Client-side video helpers.
 *
 * Everything here is a convenience for the person using the page. The server
 * re-validates the real file contents; never treat these checks as security.
 */

export const ACCEPTED_EXTENSIONS = [".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"] as const;
export const ACCEPT_ATTRIBUTE = ".mp4,.m4v,.mov,.webm,.mkv,.avi,video/mp4,video/x-m4v,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,video/avi";

// Same as the server's default (MAX_UPLOAD_SIZE); set NEXT_PUBLIC_MAX_UPLOAD_SIZE to match a different server limit.
const DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024;

export function getMaxUploadBytes(): number {
  const fromEnv = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE);
  return Number.isFinite(fromEnv) && fromEnv > 0 ? fromEnv : DEFAULT_MAX_BYTES;
}

export type ValidationResult = { ok: true } | { ok: false; message: string };

export function validateVideoFile(
  file: Pick<File, "name" | "size" | "type">,
  maxBytes: number = getMaxUploadBytes(),
): ValidationResult {
  const lowerName = file.name.toLowerCase();
  const hasAcceptedExtension = ACCEPTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
  const mimeLooksWrong = file.type !== "" && !file.type.startsWith("video/");

  if (!hasAcceptedExtension || mimeLooksWrong) {
    return {
      ok: false,
      message: "Unsupported video format. Please upload an MP4, MOV, M4V, WebM, MKV or AVI file.",
    };
  }
  if (file.size <= 0) {
    return { ok: false, message: "This file is empty. Please choose another video." };
  }
  if (file.size > maxBytes) {
    return {
      ok: false,
      message: `This file is ${formatBytes(file.size)}, which is over the ${formatBytes(maxBytes)} limit. Choose a smaller file or trim the video first.`,
    };
  }
  return { ok: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

export function formatDuration(totalSeconds: number): string {
  const rounded = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const seconds = rounded % 60;
  const ss = String(seconds).padStart(2, "0");
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, "0")}:${ss}`;
  return `${minutes}:${ss}`;
}

const COMMON_RATIOS: ReadonlyArray<readonly [string, number]> = [
  ["9:16", 9 / 16],
  ["16:9", 16 / 9],
  ["1:1", 1],
  ["4:5", 4 / 5],
  ["5:4", 5 / 4],
  ["4:3", 4 / 3],
  ["3:4", 3 / 4],
  ["21:9", 21 / 9],
];

export function aspectRatioLabel(width: number, height: number): string {
  if (!(width > 0) || !(height > 0)) return "Unknown";
  const ratio = width / height;
  for (const [label, value] of COMMON_RATIOS) {
    if (Math.abs(ratio - value) / value < 0.02) return label;
  }
  return `${ratio.toFixed(2)}:1`;
}

export interface BrowserVideoMetadata {
  durationSeconds: number | null;
  width: number;
  height: number;
}

/** Reads what the browser can know about a video. Frame rate and codec are not available here. */
export function readVideoMetadata(file: File, timeoutMs = 8000): Promise<BrowserVideoMetadata> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    let settled = false;

    const finish = (action: () => void) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(url);
      action();
    };

    const timer = window.setTimeout(
      () => finish(() => reject(new Error("Timed out reading video metadata."))),
      timeoutMs,
    );

    video.preload = "metadata";
    video.muted = true;
    video.onloadedmetadata = () => {
      // Read the values first: finish() unloads the video, which resets them to zero.
      const result: BrowserVideoMetadata = {
        durationSeconds: Number.isFinite(video.duration) ? video.duration : null,
        width: video.videoWidth,
        height: video.videoHeight,
      };
      finish(() =>
        result.width > 0 && result.height > 0
          ? resolve(result)
          : reject(new Error("The browser could not read this video's size.")),
      );
    };
    video.onerror = () => finish(() => reject(new Error("The browser could not read this video.")));
    video.src = url;
  });
}
