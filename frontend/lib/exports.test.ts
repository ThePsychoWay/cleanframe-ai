import { describe, expect, it } from "vitest";
import { SAME_SIZE, buildExportRequest, describeExport, formatKey, hasChoices, isRunning } from "./exports";
import type { ExportJob, ExportOptions } from "@/types/exports";

const sizes = (heights: number[]) => heights.map((h) => ({ height: h, label: `${h}p`, width_px: Math.round((h * 16) / 9), height_px: h }));
const FORMATS = [
  { container: "mp4", codec: "h264", label: "H.264 in MP4" },
  { container: "mp4", codec: "hevc", label: "H.265 (HEVC) in MP4" },
  { container: "webm", codec: "vp9", label: "VP9 in WebM" },
] as const;

function options(over: Partial<ExportOptions> = {}): ExportOptions {
  return {
    original_width: 3840, original_height: 2160, original_label: "4K Ultra HD",
    output_width: 3840, output_height: 2160, output_label: "4K Ultra HD",
    codec: "h264", container: "mp4", sizes: sizes([1440, 1080, 720]), formats: [...FORMATS], hardware_encoder: false, ...over,
  };
}

describe("buildExportRequest", () => {
  it("builds a request for a smaller size", () => {
    expect(buildExportRequest(options(), "1080", formatKey(FORMATS[0]))).toEqual({ height: 1080, container: "mp4", codec: "h264", quality: "balanced" });
  });

  it("refuses the finished video itself (same size, same format): there is nothing to make", () => {
    expect(buildExportRequest(options(), SAME_SIZE, formatKey(FORMATS[0]))).toBeNull();
  });

  it("allows a format change at the same size", () => {
    expect(buildExportRequest(options(), SAME_SIZE, formatKey(FORMATS[1]))).toEqual({ height: null, container: "mp4", codec: "hevc", quality: "balanced" });
  });

  it("never builds a request for a size the server didn't offer (no enlargements)", () => {
    expect(buildExportRequest(options({ sizes: sizes([720]) }), "2160", formatKey(FORMATS[0]))).toBeNull();
    expect(buildExportRequest(options({ sizes: sizes([720]) }), "1080", formatKey(FORMATS[0]))).toBeNull();
  });

  it("returns null for a format the server doesn't have", () => {
    expect(buildExportRequest(options({ formats: [FORMATS[0]] }), "720", formatKey(FORMATS[1]))).toBeNull();
  });
});

describe("hasChoices", () => {
  it("is false for a 720p video that only has its own format to offer", () => {
    expect(hasChoices(options({ sizes: [], formats: [FORMATS[0]], output_width: 1280, output_height: 720 }))).toBe(false);
  });
  it("is true when a smaller size exists, or another format does", () => {
    expect(hasChoices(options({ sizes: sizes([720]), formats: [FORMATS[0]] }))).toBe(true);
    expect(hasChoices(options({ sizes: [] }))).toBe(true);
  });
});

describe("describeExport", () => {
  const job = (over: Partial<ExportJob>): ExportJob => ({
    id: "1", status: "completed", stage: "done", frames_done: 1, frames_total: 1, percent: 100, error: null, warnings: [], settings: {},
    result: null, created_at: "", completed_at: null, ...over,
  });
  it("uses the measured size once finished", () => {
    expect(describeExport(job({ result: { width: 1920, height: 1080, container: "mp4", codec: "h264" } }))).toBe("1920 × 1080 · H.264 in MP4");
  });
  it("falls back to what was asked while running", () => {
    expect(describeExport(job({ status: "running", settings: { height: 720, container: "webm", codec: "vp9" } }))).toBe("720p HD · VP9 in WebM");
  });
  it("knows what is still running", () => {
    expect(isRunning(job({ status: "queued" }))).toBe(true);
    expect(isRunning(job({ status: "completed" }))).toBe(false);
  });
});
