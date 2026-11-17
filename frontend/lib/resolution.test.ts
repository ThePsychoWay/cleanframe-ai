import { describe, expect, it } from "vitest";
import { formatContainer, formatFrameRate, formatSize, resolutionLabel, targetLabel } from "./resolution";

describe("resolutionLabel", () => {
  it.each([
    [1280, 720, "720p HD"],
    [1920, 1080, "1080p Full HD"],
    [2560, 1440, "1440p QHD"],
    [3840, 2160, "4K Ultra HD"],
    [4096, 2160, "4K Ultra HD"],
    [3840, 1600, "4K Ultra HD"], // ultrawide: the long side reaches 4K
    [1080, 1920, "1080p Full HD"], // portrait is named by the same rule
    [2160, 3840, "4K Ultra HD"],
    [854, 480, "480p SD"],
    [320, 240, "240p"],
  ])("%i × %i is %s", (w, h, label) => {
    expect(resolutionLabel(w, h)).toBe(label);
  });
});

describe("display helpers", () => {
  it("writes sizes the way the app shows them", () => {
    expect(formatSize(3840, 2160)).toBe("3840 × 2160");
    expect(formatFrameRate(30)).toBe("30 FPS");
    expect(formatFrameRate(29.97003)).toBe("29.97 FPS");
    expect(targetLabel(1440)).toBe("1440p QHD");
    expect(targetLabel(2160)).toBe("4K Ultra HD");
  });

  it("names containers", () => {
    expect(formatContainer("mp4")).toBe("MP4");
    expect(formatContainer("webm")).toBe("WebM");
    expect(formatContainer("avi")).toBe("AVI");
    expect(formatContainer(null)).toBe("Unknown");
  });

  it("tells .mp4 from .mov by the file's name, because ffprobe calls both 'mov'", () => {
    expect(formatContainer("mov", "clip.mp4")).toBe("MP4");
    expect(formatContainer("mov", "Clip.MOV")).toBe("MOV");
    expect(formatContainer("mov", "clip.m4v")).toBe("MP4");
    expect(formatContainer("mov")).toBe("MOV");
    expect(formatContainer("matroska", "clip.webm")).toBe("WebM");
    expect(formatContainer("avi", "clip.avi")).toBe("AVI");
    expect(formatContainer("mov", "weird.xyz")).toBe("MOV"); // a name that doesn't fit the family changes nothing
  });
});
