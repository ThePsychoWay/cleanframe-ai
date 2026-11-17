import { describe, expect, it } from "vitest";
import { detectStaticOverlays, type GrayFrame } from "./detect";

const W = 96;
const H = 160;

/** A moving, smooth background (waves that drift over time) with optional still overlays drawn on top. */
function video(frames: number, overlays: Array<{ x: number; y: number; w: number; h: number; plate?: boolean }>, motion = true): GrayFrame[] {
  return Array.from({ length: frames }, (_, t) => {
    const data = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const phase = motion ? t * 0.9 : 0;
        data[y * W + x] = 128 + 70 * Math.sin(x / 9 + phase) * Math.cos(y / 13 - phase * 0.7);
      }
    }
    for (const o of overlays) {
      for (let y = o.y; y < o.y + o.h; y++) {
        for (let x = o.x; x < o.x + o.w; x++) {
          const inLetters = !o.plate || (x > o.x + 6 && x < o.x + o.w - 6 && y > o.y + 3 && y < o.y + o.h - 3);
          if (!inLetters) {
            // A see-through white plate: half of the moving picture still shows through it.
            data[y * W + x] = data[y * W + x] * 0.5 + 255 * 0.5;
          } else {
            // Bright and dark stripes ("letters"), so there are edges inside the box as well as around it.
            data[y * W + x] = (x - o.x) % 6 < 3 ? 240 : 30;
          }
        }
      }
    }
    return { width: W, height: H, data };
  });
}

function overlap(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (ix * iy) / (a.w * a.h + b.w * b.h - ix * iy);
}

describe("detectStaticOverlays", () => {
  it("finds a still logo over a moving picture", () => {
    const logo = { x: 60, y: 12, w: 30, h: 14 };
    const outcome = detectStaticOverlays(video(10, [logo]));
    expect(outcome.kind).toBe("found");
    if (outcome.kind !== "found") return;
    const truth = { x: logo.x / W, y: logo.y / H, w: logo.w / W, h: logo.h / H };
    expect(overlap(outcome.boxes[0], truth)).toBeGreaterThan(0.5);
  });

  it("covers the see-through plate behind the letters, not just the letters", () => {
    const logo = { x: 40, y: 10, w: 50, h: 20, plate: true };
    const outcome = detectStaticOverlays(video(12, [logo]));
    expect(outcome.kind).toBe("found");
    if (outcome.kind !== "found") return;
    const truth = { x: logo.x / W, y: logo.y / H, w: logo.w / W, h: logo.h / H };
    expect(overlap(outcome.boxes[0], truth)).toBeGreaterThan(0.6);
  });

  it("finds two separate overlays", () => {
    const a = { x: 4, y: 6, w: 26, h: 12 };
    const b = { x: 58, y: 138, w: 32, h: 14 };
    const outcome = detectStaticOverlays(video(10, [a, b]));
    expect(outcome.kind).toBe("found");
    if (outcome.kind !== "found") return;
    expect(outcome.boxes).toHaveLength(2);
  });

  it("reports nothing when the picture moves and there is no overlay", () => {
    expect(detectStaticOverlays(video(10, [])).kind).toBe("none");
  });

  it("does not guess when the picture never changes", () => {
    const outcome = detectStaticOverlays(video(10, [{ x: 60, y: 12, w: 30, h: 14 }], false));
    expect(outcome.kind).toBe("unknown");
  });

  it("does not guess from too few frames", () => {
    expect(detectStaticOverlays(video(2, [])).kind).toBe("unknown");
  });

  it("ignores a letterbox bar across the whole width", () => {
    const outcome = detectStaticOverlays(video(10, [{ x: 0, y: 0, w: W, h: 12 }]));
    expect(outcome.kind === "found" ? outcome.boxes : []).toHaveLength(0);
  });

  it("returns boxes inside the picture", () => {
    const outcome = detectStaticOverlays(video(10, [{ x: 66, y: 146, w: 30, h: 14 }]));
    expect(outcome.kind).toBe("found");
    if (outcome.kind !== "found") return;
    for (const r of outcome.boxes) {
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.w).toBeLessThanOrEqual(1.000001);
      expect(r.y + r.h).toBeLessThanOrEqual(1.000001);
    }
  });
});
