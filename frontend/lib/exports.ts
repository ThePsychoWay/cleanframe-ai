/** Pure helpers for the "more sizes" panel: what to offer, how to word an export, when to keep polling. */

import { formatContainer, targetLabel } from "./resolution";
import type { ExportJob, ExportOptions, ExportRequest, FormatOption } from "@/types/exports";

export const SAME_SIZE = "same";

export function isRunning(job: ExportJob): boolean {
  return job.status === "queued" || job.status === "running";
}

/** The panel is worth showing only when there is something to choose besides the file already on offer. */
export function hasChoices(options: ExportOptions): boolean {
  return options.sizes.length > 0 || options.formats.some((f) => !isFinishedFormat(f, options));
}

export function isFinishedFormat(format: Pick<FormatOption, "container" | "codec">, options: Pick<ExportOptions, "container" | "codec">): boolean {
  return format.container === options.container && format.codec === options.codec;
}

export function formatKey(format: Pick<FormatOption, "container" | "codec">): string {
  return `${format.container}:${format.codec}`;
}

/** Null when the chosen combination is the finished video itself (the server would refuse it; the page doesn't offer it). */
export function buildExportRequest(options: ExportOptions, size: string, format: string): ExportRequest | null {
  const chosen = options.formats.find((f) => formatKey(f) === format);
  if (!chosen) return null;
  const height = size === SAME_SIZE ? null : Number(size);
  if (height === null && isFinishedFormat(chosen, options)) return null;
  if (height !== null && !options.sizes.some((s) => s.height === height)) return null;
  return { height: height as ExportRequest["height"], container: chosen.container, codec: chosen.codec, quality: "balanced" };
}

/** "1080p Full HD · H.264 in MP4", from what the server recorded for the export. */
export function describeExport(job: ExportJob): string {
  const result = job.result;
  const size = result ? `${result.width} × ${result.height}` : job.settings.height ? targetLabel(Number(job.settings.height)) : "Same size";
  const container = formatContainer(String(result?.container ?? job.settings.container ?? "mp4"));
  const codec = String(result?.codec ?? job.settings.codec ?? "").toUpperCase().replace("H264", "H.264").replace("HEVC", "H.265 (HEVC)");
  return `${size} · ${codec ? `${codec} in ` : ""}${container}`;
}
