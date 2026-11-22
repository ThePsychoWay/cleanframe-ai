/** Mirrors backend/app/schemas/export.py. */

import type { RenderSettings } from "./render";

export type ExportContainer = "mp4" | "mov" | "webm" | "mkv";
export type ExportCodec = "h264" | "hevc" | "vp9" | "av1";

export interface ExportRequest {
  /** Shorter side of the new picture; null keeps the finished video's size and only changes the format. */
  height: 720 | 1080 | 1440 | 2160 | null;
  container: ExportContainer;
  codec: ExportCodec | null;
  quality: RenderSettings["quality"];
}

export interface SizeOption {
  height: number;
  label: string;
  width_px: number;
  height_px: number;
}

export interface FormatOption {
  container: ExportContainer;
  codec: ExportCodec;
  label: string;
}

/** What can honestly be exported from the finished video: only smaller sizes, only encoders this server has. */
export interface ExportOptions {
  original_width: number;
  original_height: number;
  original_label: string;
  output_width: number;
  output_height: number;
  output_label: string;
  codec: string | null;
  container: string;
  sizes: SizeOption[];
  formats: FormatOption[];
  hardware_encoder: boolean;
}

export type ExportStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface ExportJob {
  id: string;
  status: ExportStatus;
  stage: string;
  frames_done: number;
  frames_total: number;
  percent: number | null;
  error: string | null;
  warnings: string[];
  settings: Record<string, unknown>;
  result: { width: number; height: number; size_bytes?: number; container?: string; codec?: string; encoder?: string } | null;
  created_at: string;
  completed_at: string | null;
}

export interface ExportList {
  exports: ExportJob[];
}
