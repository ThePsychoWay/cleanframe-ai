/**
 * Finds overlays that stay in one place while the picture behind them moves (a corner logo, a channel bug, a burned-in
 * handle). It is a heuristic over a handful of frames, so it only ever *suggests*: the person confirms or adjusts every box.
 *
 * What it cannot see, and says so instead of guessing:
 *  - a video that barely changes (nothing separates an overlay from the scene),
 *  - an overlay that moves (it isn't still, so it isn't found).
 *
 * Pure functions over grey images; the browser side that reads the frames lives in lib/detectVideo.ts.
 */

import type { Rect } from "../types/mask";

export interface GrayFrame {
  width: number;
  height: number;
  /** One byte per pixel, row by row. */
  data: ArrayLike<number>;
}

export type DetectOutcome =
  | { kind: "found"; boxes: Rect[] }
  | { kind: "none" }
  | { kind: "unknown"; reason: string };

export const MIN_FRAMES = 4;
/** The scene must move at least this much (grey levels, median over pixels) for "still" to mean anything. */
export const MIN_SCENE_MOTION = 4;
const STILL_FRACTION = 0.3; // a pixel is "still" if its variation is below this share of the scene's
const MIN_EDGE = 14; // grey-level step a logo's outline or letters make
// A see-through plate behind the letters lets some of the moving picture through, so it varies a little. Its outline still
// survives averaging over time, which is how its edges are told from the moving picture's (those blur away).
const PLATE_STILL_FRACTION = 0.92;
const PLATE_EDGE = 20;
const MIN_ANCHOR = 4; // a blob counts only if it contains at least this many clearly still pixels
const MIN_AREA = 0.0015;
const MAX_AREA = 0.2;
const MIN_FILL = 0.2;
const MAX_BOXES = 3;

function medianOf(values: Float32Array): number {
  const copy = Float32Array.from(values).sort();
  return copy[Math.floor(copy.length / 2)] ?? 0;
}

/** Mean and standard deviation of each pixel over time. */
function pixelStats(frames: GrayFrame[]): { mean: Float32Array; std: Float32Array } {
  const size = frames[0].width * frames[0].height;
  const mean = new Float32Array(size);
  const std = new Float32Array(size);
  for (const f of frames) for (let i = 0; i < size; i++) mean[i] += f.data[i] / frames.length;
  for (const f of frames) {
    for (let i = 0; i < size; i++) {
      const d = f.data[i] - mean[i];
      std[i] += d * d;
    }
  }
  for (let i = 0; i < size; i++) std[i] = Math.sqrt(std[i] / frames.length);
  return { mean, std };
}

/** Strength of edges in an image (largest neighbour step, cheap and good enough for outlines and letters). */
function edgeStrength(img: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const c = img[i];
      let best = 0;
      if (x > 0) best = Math.max(best, Math.abs(c - img[i - 1]));
      if (x < w - 1) best = Math.max(best, Math.abs(c - img[i + 1]));
      if (y > 0) best = Math.max(best, Math.abs(c - img[i - w]));
      if (y < h - 1) best = Math.max(best, Math.abs(c - img[i + w]));
      out[i] = best;
    }
  }
  return out;
}

function dilate(mask: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  // Separable box dilation: horizontal, then vertical.
  const tmp = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < w) tmp[y * w + nx] = 1;
      }
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!tmp[y * w + x]) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny >= 0 && ny < h) out[ny * w + x] = 1;
      }
    }
  }
  return out;
}

interface Component {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  pixels: number;
}

function components(mask: Uint8Array, w: number, h: number): Component[] {
  const seen = new Uint8Array(w * h);
  const found: Component[] = [];
  const stack: number[] = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    const c: Component = { x0: w, y0: h, x1: -1, y1: -1, pixels: 0 };
    stack.push(start);
    seen[start] = 1;
    while (stack.length > 0) {
      const i = stack.pop() as number;
      const x = i % w;
      const y = (i - x) / w;
      c.pixels++;
      if (x < c.x0) c.x0 = x;
      if (x > c.x1) c.x1 = x;
      if (y < c.y0) c.y0 = y;
      if (y > c.y1) c.y1 = y;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (mask[n] && !seen[n]) {
          seen[n] = 1;
          stack.push(n);
        }
      }
    }
    found.push(c);
  }
  return found;
}

export function detectStaticOverlays(frames: GrayFrame[]): DetectOutcome {
  if (frames.length < MIN_FRAMES) return { kind: "unknown", reason: "The video is too short to compare enough frames." };
  const { width: w, height: h } = frames[0];
  if (frames.some((f) => f.width !== w || f.height !== h)) return { kind: "unknown", reason: "The frames couldn't be compared." };

  const { mean, std } = pixelStats(frames);
  const sceneMotion = medianOf(std);
  if (sceneMotion < MIN_SCENE_MOTION) {
    return {
      kind: "unknown",
      reason: "The picture barely changes during the video, so an overlay can't be told apart from the scene.",
    };
  }

  const limit = sceneMotion * STILL_FRACTION;
  const looseLimit = sceneMotion * PLATE_STILL_FRACTION;
  const edges = edgeStrength(mean, w, h);
  const strict = new Uint8Array(w * h); // clearly still, with an edge: letters, glyphs, solid logos
  const loose = new Uint8Array(w * h); // strict, or an outline that survives averaging: a see-through plate's border
  for (let i = 0; i < strict.length; i++) {
    strict[i] = std[i] < limit && edges[i] >= MIN_EDGE ? 1 : 0;
    loose[i] = strict[i] || (std[i] < looseLimit && edges[i] >= PLATE_EDGE) ? 1 : 0;
  }

  // Letters and outlines are thin: join them into one blob per overlay.
  const radius = Math.max(2, Math.round(Math.min(w, h) * 0.04));
  const blobs = components(dilate(loose, w, h, radius), w, h);

  const boxes: Array<{ rect: Rect; score: number }> = [];
  for (const b of blobs) {
    const bw = b.x1 - b.x0 + 1;
    const bh = b.y1 - b.y0 + 1;
    const area = (bw * bh) / (w * h);
    if (area < MIN_AREA || area > MAX_AREA) continue;
    // A bar across the whole picture is a letterbox border, not an overlay.
    if (bw >= w * 0.98 || bh >= h * 0.98) continue;
    if (b.pixels / (bw * bh) < MIN_FILL) continue;
    // Take the box from the real candidate pixels inside the blob, not from the grown blob. Outlines alone never count:
    // something clearly still has to be in there.
    let x0 = w, y0 = h, x1 = -1, y1 = -1, anchors = 0;
    for (let y = b.y0; y <= b.y1; y++) {
      for (let x = b.x0; x <= b.x1; x++) {
        const i = y * w + x;
        if (!loose[i]) continue;
        if (strict[i]) anchors++;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
    if (anchors < MIN_ANCHOR) continue;
    // A see-through plate may reach a little past the outline found above (its edge can be too faint to register).
    // Picture motion is not even across the frame, so a strip counts as plate only when it is clearly calmer than the
    // picture just beyond the farthest it could reach on that side. Letters with nothing behind them never grow.
    const growX = Math.max(2, Math.round(w * 0.06));
    const growY = Math.max(2, Math.round(h * 0.04));
    const colMean = (x: number) => {
      let sum = 0;
      for (let y = y0; y <= y1; y++) sum += std[y * w + x];
      return sum / (y1 - y0 + 1);
    };
    const rowMean = (y: number) => {
      let sum = 0;
      for (let x = x0; x <= x1; x++) sum += std[y * w + x];
      return sum / (x1 - x0 + 1);
    };
    const GROW_RATIO = 0.88;
    const fx0 = x0, fx1 = x1, fy0 = y0, fy1 = y1;
    const refL = fx0 - growX - 3 >= 0 ? colMean(fx0 - growX - 3) : -1;
    const refR = fx1 + growX + 3 < w ? colMean(fx1 + growX + 3) : -1;
    if (refL > 0) for (let g = 0; g < growX && x0 > 0 && colMean(x0 - 1) < refL * GROW_RATIO; g++) x0--;
    if (refR > 0) for (let g = 0; g < growX && x1 < w - 1 && colMean(x1 + 1) < refR * GROW_RATIO; g++) x1++;
    const refT = fy0 - growY - 3 >= 0 ? rowMean(fy0 - growY - 3) : -1;
    const refB = fy1 + growY + 3 < h ? rowMean(fy1 + growY + 3) : -1;
    if (refT > 0) for (let g = 0; g < growY && y0 > 0 && rowMean(y0 - 1) < refT * GROW_RATIO; g++) y0--;
    if (refB > 0) for (let g = 0; g < growY && y1 < h - 1 && rowMean(y1 + 1) < refB * GROW_RATIO; g++) y1++;
    const tw = x1 - x0 + 1;
    const th = y1 - y0 + 1;
    const pad = 1;
    const rect: Rect = {
      x: Math.max(0, (x0 - pad) / w),
      y: Math.max(0, (y0 - pad) / h),
      w: Math.min(1, (tw + pad * 2) / w),
      h: Math.min(1, (th + pad * 2) / h),
    };
    rect.w = Math.min(rect.w, 1 - rect.x);
    rect.h = Math.min(rect.h, 1 - rect.y);
    boxes.push({ rect, score: anchors });
  }

  if (boxes.length === 0) return { kind: "none" };
  boxes.sort((a, b) => b.score - a.score);
  return { kind: "found", boxes: boxes.slice(0, MAX_BOXES).map((b) => b.rect) };
}
