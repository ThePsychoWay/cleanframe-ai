import { describe, expect, it } from "vitest";
import { clampPan, clampPercent, dividerFromKey, dividerFromPointer, linkNeedsRefresh, needsResync, nextZoom, panForZoom } from "./compare";

describe("divider", () => {
  it("stays between 0 and 100 and survives bad numbers", () => {
    expect(clampPercent(-5)).toBe(0);
    expect(clampPercent(140)).toBe(100);
    expect(clampPercent(NaN)).toBe(50);
  });

  it("follows the pointer across the picture", () => {
    expect(dividerFromPointer(300, 100, 400)).toBe(50);
    expect(dividerFromPointer(50, 100, 400)).toBe(0);
    expect(dividerFromPointer(900, 100, 400)).toBe(100);
    expect(dividerFromPointer(10, 0, 0)).toBe(50);
  });

  it("moves with the keyboard", () => {
    expect(dividerFromKey(50, "ArrowRight", false)).toBe(52);
    expect(dividerFromKey(50, "ArrowLeft", true)).toBe(40);
    expect(dividerFromKey(99, "ArrowRight", true)).toBe(100);
    expect(dividerFromKey(50, "Home", false)).toBe(0);
    expect(dividerFromKey(50, "End", false)).toBe(100);
    expect(dividerFromKey(50, "a", false)).toBeNull();
  });
});

describe("zoom and pan", () => {
  const box = { width: 400, height: 300 };

  it("steps through the levels and stops at the ends", () => {
    expect(nextZoom(1, 1)).toBe(2);
    expect(nextZoom(2, 1)).toBe(4);
    expect(nextZoom(4, 1)).toBe(4);
    expect(nextZoom(1, -1)).toBe(1);
    expect(nextZoom(4, -1)).toBe(2);
  });

  it("never lets the picture be dragged off its frame", () => {
    expect(clampPan({ x: 500, y: -500 }, 2, box)).toEqual({ x: 200, y: -150 });
    expect(clampPan({ x: 30, y: 10 }, 2, box)).toEqual({ x: 30, y: 10 });
    expect(clampPan({ x: 99, y: 99 }, 1, box)).toEqual({ x: 0, y: 0 });
  });

  it("keeps the same spot in view when the zoom changes", () => {
    expect(panForZoom({ x: 100, y: 75 }, 2, 4, box)).toEqual({ x: 300, y: 225 });
    expect(panForZoom({ x: 300, y: 225 }, 4, 2, box)).toEqual({ x: 100, y: 75 });
    expect(panForZoom({ x: 300, y: 225 }, 4, 1, box)).toEqual({ x: 0, y: 0 });
    expect(panForZoom({ x: 0, y: 0 }, 1, 2, box)).toEqual({ x: 0, y: 0 });
  });
});

describe("needsResync", () => {
  it("tolerates a little drift while playing but not when paused", () => {
    expect(needsResync(1.0, 1.02, 30, true)).toBe(false); // under 1.5 frames
    expect(needsResync(1.0, 1.08, 30, true)).toBe(true);
    expect(needsResync(1.0, 1.02, 30, false)).toBe(true); // over half a frame
    expect(needsResync(1.0, 1.01, 30, false)).toBe(false);
  });

  it("ignores values that aren't times", () => {
    expect(needsResync(NaN, 1, 30, true)).toBe(false);
    expect(needsResync(1, 2, 0, true)).toBe(false);
  });
});

describe("linkNeedsRefresh", () => {
  it("asks for a fresh link before the old one runs out", () => {
    expect(linkNeedsRefresh(0, 3600, 60_000)).toBe(false);
    expect(linkNeedsRefresh(0, 3600, 3_299_000)).toBe(false);
    expect(linkNeedsRefresh(0, 3600, 3_300_000)).toBe(true);
    expect(linkNeedsRefresh(0, 100, 1)).toBe(true); // a link shorter than the margin is always stale
  });
});
