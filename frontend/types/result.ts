/** Mirrors backend/app/schemas/result.py. */

import type { RenderResult } from "./render";

export interface PreviewInfo {
  result_url: string;
  original_url: string;
  expires_in: number;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  container: string;
  size_bytes: number | null;
  audio: RenderResult["audio"] | null;
  development_only: boolean;
  warnings: string[];
  finished_at: string | null;
  facts: Partial<RenderResult>;
}

export interface ExportInfo {
  url: string;
  filename: string;
  expires_in: number;
  size_bytes: number | null;
}
