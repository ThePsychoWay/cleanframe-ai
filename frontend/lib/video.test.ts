// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { aspectRatioLabel, formatBytes, formatDuration, readVideoMetadata, validateVideoFile } from "./video";

const MB = 1024 * 1024;

describe("validateVideoFile", () => {
  it("accepts mp4, m4v, mov, webm, mkv and avi", () => {
    for (const name of ["a.mp4", "b.MOV", "c.webm", "d.avi", "e.mkv", "f.M4V"]) {
      expect(validateVideoFile({ name, size: MB, type: "" }, 10 * MB)).toEqual({ ok: true });
    }
  });

  it("rejects unsupported extensions with the spec'd message", () => {
    const result = validateVideoFile({ name: "clip.flv", size: MB, type: "video/x-flv" }, 10 * MB);
    expect(result).toEqual({
      ok: false,
      message: "Unsupported video format. Please upload an MP4, MOV, M4V, WebM, MKV or AVI file.",
    });
  });

  it("takes a multi-gigabyte 4K file by default (the limit is the server's, 2 GiB unless configured)", () => {
    expect(validateVideoFile({ name: "4k.mov", size: 1.5 * 1024 * MB, type: "video/quicktime" })).toEqual({ ok: true });
    expect(validateVideoFile({ name: "4k.mov", size: 3 * 1024 * MB, type: "video/quicktime" }).ok).toBe(false);
  });

  it("rejects a non-video MIME type even with a video extension", () => {
    const result = validateVideoFile({ name: "clip.mp4", size: MB, type: "application/pdf" }, 10 * MB);
    expect(result.ok).toBe(false);
  });

  it("rejects empty and oversized files", () => {
    expect(validateVideoFile({ name: "a.mp4", size: 0, type: "video/mp4" }, 10 * MB).ok).toBe(false);
    const big = validateVideoFile({ name: "a.mp4", size: 11 * MB, type: "video/mp4" }, 10 * MB);
    expect(big.ok).toBe(false);
    if (!big.ok) expect(big.message).toContain("10.0 MB");
  });
});

describe("formatters", () => {
  it("formats bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(5 * MB)).toBe("5.0 MB");
  });

  it("formats durations", () => {
    expect(formatDuration(0)).toBe("0:00");
    expect(formatDuration(75)).toBe("1:15");
    expect(formatDuration(3725)).toBe("1:02:05");
  });

  it("labels common aspect ratios", () => {
    expect(aspectRatioLabel(1080, 1920)).toBe("9:16");
    expect(aspectRatioLabel(1920, 1080)).toBe("16:9");
    expect(aspectRatioLabel(1080, 1350)).toBe("4:5");
    expect(aspectRatioLabel(1000, 1000)).toBe("1:1");
    expect(aspectRatioLabel(2560, 1080)).toBe("21:9");
    expect(aspectRatioLabel(1000, 500)).toBe("2.00:1");
    expect(aspectRatioLabel(0, 100)).toBe("Unknown");
  });
});

describe("readVideoMetadata", () => {
  afterEach(() => vi.restoreAllMocks());

  /** A stand-in <video> whose size resets to 0 when it is unloaded, like a real one. */
  function fakeVideo(width: number, height: number) {
    const real = document.createElement.bind(document);
    const video: Record<string, unknown> = { duration: 4, videoWidth: width, videoHeight: height };
    video.removeAttribute = () => {};
    video.load = () => {
      video.videoWidth = 0;
      video.videoHeight = 0;
    };
    Object.defineProperty(video, "src", {
      set: () => queueMicrotask(() => (video.onloadedmetadata as () => void)()),
    });
    vi.spyOn(document, "createElement").mockImplementation((tag: string) =>
      tag === "video" ? (video as unknown as HTMLVideoElement) : real(tag),
    );
    URL.createObjectURL = () => "blob:x";
    URL.revokeObjectURL = () => {};
  }

  it("returns the size the video had when its metadata loaded", async () => {
    fakeVideo(540, 960);
    await expect(readVideoMetadata(new File([""], "a.mp4"))).resolves.toEqual({ durationSeconds: 4, width: 540, height: 960 });
  });

  it("refuses a video whose size is unknown instead of reporting 0 x 0", async () => {
    fakeVideo(0, 0);
    await expect(readVideoMetadata(new File([""], "a.mp4"))).rejects.toThrow();
  });
});
