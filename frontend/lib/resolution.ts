/**
 * Naming picture sizes the way people do. Mirrors backend/app/video/resolution.py, which is the source of truth for what
 * can be exported; this file only words what the server already reported.
 */

// [short name, label, smallest shorter side, smallest longer side]. A size gets the highest class either side reaches,
// so 3840×1600 (ultrawide) is "4K" like 3840×2160 and 1080×1920 (portrait) is "1080p" like 1920×1080.
const CLASSES = [
  ["4k", "4K Ultra HD", 2160, 3840],
  ["1440p", "1440p QHD", 1440, 2560],
  ["1080p", "1080p Full HD", 1080, 1920],
  ["720p", "720p HD", 720, 1280],
  ["480p", "480p SD", 480, 854],
] as const;

export const SUPPORTED_RESOLUTIONS_TEXT = "Supported resolutions: 720p, 1080p, 1440p and 4K.";

/** Output sizes that can be asked for, by shorter side. Largest first. */
export const OUTPUT_HEIGHTS = [2160, 1440, 1080, 720] as const;
export type OutputHeight = (typeof OUTPUT_HEIGHTS)[number];

const TARGET_LABELS: Record<number, string> = { 2160: "4K Ultra HD", 1440: "1440p QHD", 1080: "1080p Full HD", 720: "720p HD" };

export function resolutionLabel(width: number, height: number): string {
  const short = Math.min(width, height);
  const long = Math.max(width, height);
  for (const [, label, minShort, minLong] of CLASSES) {
    if (short >= minShort * 0.97 || long >= minLong * 0.97) return label;
  }
  return `${short}p`;
}

/** "3840 × 2160" */
export function formatSize(width: number, height: number): string {
  return `${width} × ${height}`;
}

export function targetLabel(height: number): string {
  return TARGET_LABELS[height] ?? `${height}p`;
}

/** "30 FPS", "29.97 FPS". */
export function formatFrameRate(fps: number): string {
  return `${Number(fps.toFixed(2))} FPS`;
}

const CONTAINER_NAMES: Record<string, string> = { mp4: "MP4", mov: "MOV", m4v: "MP4", webm: "WebM", avi: "AVI", matroska: "WebM" };
// ffprobe names the *demuxer*, which is the same ("mov") for .mp4 and .mov files and the same ("matroska") for .webm and .mkv, so
// for those families the file's own extension says what the person would call it.
const FAMILIES: Record<string, string[]> = { mov: ["mp4", "mov", "m4v"], matroska: ["webm"] };

export function formatContainer(container: string | null | undefined, filename?: string | null): string {
  if (!container) return "Unknown";
  const key = container.toLowerCase();
  const extension = filename?.toLowerCase().match(/\.([a-z0-9]+)$/)?.[1];
  if (extension && FAMILIES[key]?.includes(extension)) return CONTAINER_NAMES[extension];
  return CONTAINER_NAMES[key] ?? container.toUpperCase();
}
