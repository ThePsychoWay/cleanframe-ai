/**
 * Mask editing logic, kept free of React and the DOM so it can be tested directly.
 *
 * Every function that changes a document returns a new one and leaves the input untouched; that is
 * what makes undo/redo a matter of keeping the old references.
 */

import type { Keyframe, MaskDocument, Point, Rect, Region, RegionType, ShapeResolver, Stroke } from "../types/mask";

// Limits mirror the server's (backend/app/schemas/mask.py) so problems are caught before saving.
export const MAX_REGIONS = 20;
export const MAX_KEYFRAMES_PER_REGION = 200;
export const MAX_POLYGON_POINTS = 500;
export const MAX_STROKES_PER_KEYFRAME = 500;
export const MAX_POINTS_PER_STROKE = 2000;
export const MAX_TOTAL_POINTS = 50_000;
export const MIN_RADIUS = 0.001;
export const MAX_RADIUS = 0.25;
export const MIN_RECT_SIZE = 0.01;

export const EMPTY_DOCUMENT: MaskDocument = { version: 1, regions: [] };

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
export const clamp01 = (value: number) => clamp(value, 0, 1);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// --- frames and time ----------------------------------------------------------------------------

/**
 * The middle of a frame's display interval. Seeking to a frame's exact start can land on the previous
 * frame after rounding; the middle always shows the frame asked for.
 */
export function frameToTime(frame: number, fps: number): number {
  return (frame + 0.5) / fps;
}

export function timeToFrame(seconds: number, fps: number, frameCount: number): number {
  return clamp(Math.floor(seconds * fps), 0, Math.max(0, frameCount - 1));
}

// --- regions ------------------------------------------------------------------------------------

export function nextRegionNumber(doc: MaskDocument): number {
  let highest = 0;
  for (const region of doc.regions) {
    const match = /^region_(\d+)$/.exec(region.id);
    if (match) highest = Math.max(highest, Number(match[1]));
  }
  return highest + 1;
}

export function createRegion(doc: MaskDocument, type: RegionType, keyframe: Keyframe): Region {
  const n = nextRegionNumber(doc);
  return { id: `region_${n}`, name: `Region ${n}`, type, keyframes: [keyframe] };
}

export function addRegion(doc: MaskDocument, region: Region): MaskDocument {
  return { ...doc, regions: [...doc.regions, region] };
}

export function removeRegion(doc: MaskDocument, id: string): MaskDocument {
  return { ...doc, regions: doc.regions.filter((r) => r.id !== id) };
}

export function renameRegion(doc: MaskDocument, id: string, name: string): MaskDocument {
  return { ...doc, regions: doc.regions.map((r) => (r.id === id ? { ...r, name } : r)) };
}

export function findRegion(doc: MaskDocument, id: string | null): Region | undefined {
  return id === null ? undefined : doc.regions.find((r) => r.id === id);
}

// --- keyframes ----------------------------------------------------------------------------------

export function hasKeyframeAt(region: Region, frame: number): boolean {
  return region.keyframes.some((k) => k.frame === frame);
}

/** Insert a keyframe in order, replacing any that is already on that frame. */
export function setKeyframe(doc: MaskDocument, regionId: string, keyframe: Keyframe): MaskDocument {
  return {
    ...doc,
    regions: doc.regions.map((region) => {
      if (region.id !== regionId) return region;
      const others = region.keyframes.filter((k) => k.frame !== keyframe.frame);
      const keyframes = [...others, keyframe].sort((a, b) => a.frame - b.frame);
      return { ...region, keyframes };
    }),
  };
}

/** A region always keeps at least one keyframe; deleting the last one means removing the region. */
export function removeKeyframe(doc: MaskDocument, regionId: string, frame: number): MaskDocument {
  const region = findRegion(doc, regionId);
  // Returning the same document when nothing changes keeps no-ops out of the undo history.
  if (!region || region.keyframes.length <= 1 || !hasKeyframeAt(region, frame)) return doc;
  return {
    ...doc,
    regions: doc.regions.map((r) =>
      r.id === regionId ? { ...r, keyframes: r.keyframes.filter((k) => k.frame !== frame) } : r,
    ),
  };
}

const lerpRect = (a: Rect, b: Rect, t: number): Rect => ({
  x: lerp(a.x, b.x, t),
  y: lerp(a.y, b.y, t),
  w: lerp(a.w, b.w, t),
  h: lerp(a.h, b.h, t),
});

const lerpPoints = (a: Point[], b: Point[], t: number): Point[] =>
  a.map((p, i) => [lerp(p[0], b[i][0], t), lerp(p[1], b[i][1], t)]);

/**
 * The shape a region has on `frame`.
 *
 * A single keyframe applies to the whole video. Before the first and after the last keyframe the
 * nearest one holds. Between two keyframes, rectangles and same-size polygons are blended linearly;
 * brush strokes and polygons whose point counts differ hold the earlier keyframe until the next one.
 * This is manual keyframing only: following a moving object automatically is tracking, a later phase.
 */
export function shapeAt(region: Region, frame: number): Keyframe {
  const keyframes = region.keyframes;
  let previous = keyframes[0];
  let next: Keyframe | undefined;
  for (const keyframe of keyframes) {
    if (keyframe.frame <= frame) previous = keyframe;
    else {
      next = keyframe;
      break;
    }
  }
  // A region that appears or vanishes does so at its keyframe: nothing is blended with "not there".
  if (frame <= previous.frame || next === undefined || previous.hidden || next.hidden) return previous;

  const t = (frame - previous.frame) / (next.frame - previous.frame);
  if (region.type === "rectangle" && previous.rect && next.rect) {
    return { ...previous, frame, rect: lerpRect(previous.rect, next.rect, t) };
  }
  if (region.type === "polygon" && previous.points && next.points && previous.points.length === next.points.length) {
    return { ...previous, frame, points: lerpPoints(previous.points, next.points, t) };
  }
  return previous;
}

const clampStroke = (stroke: Stroke): Stroke => ({
  radius: clamp(stroke.radius, MIN_RADIUS, MAX_RADIUS),
  points: stroke.points.map((p): Point => [clamp01(p[0]), clamp01(p[1])]),
});

/**
 * Keep a shape inside the frame and within the saved format's limits. Tracked shapes can extend past
 * the frame edge while they move, and the server rejects coordinates outside 0 to 1.
 */
export function clampKeyframe(keyframe: Keyframe): Keyframe {
  const result: Keyframe = { ...keyframe };
  if (keyframe.rect) {
    const w = clamp(keyframe.rect.w, MIN_RECT_SIZE, 1);
    const h = clamp(keyframe.rect.h, MIN_RECT_SIZE, 1);
    result.rect = { x: clamp(keyframe.rect.x, 0, 1 - w), y: clamp(keyframe.rect.y, 0, 1 - h), w, h };
  }
  if (keyframe.points) result.points = keyframe.points.map((p): Point => [clamp01(p[0]), clamp01(p[1])]);
  if (keyframe.strokes) result.strokes = keyframe.strokes.map(clampStroke);
  if (keyframe.erase) result.erase = keyframe.erase.map(clampStroke);
  return result;
}

/**
 * Change a region's shape on `frame`. The edit starts from the shape the region currently has there
 * (blended, held, or tracked, depending on `resolve`) and is stored as a keyframe on exactly that frame.
 */
export function editShapeAt(
  doc: MaskDocument,
  regionId: string,
  frame: number,
  edit: (current: Keyframe) => Keyframe,
  resolve: ShapeResolver = shapeAt,
): MaskDocument {
  const region = findRegion(doc, regionId);
  if (!region) return doc;
  const edited = edit(resolve(region, frame));
  return setKeyframe(doc, regionId, clampKeyframe({ ...edited, frame }));
}

// --- geometry -----------------------------------------------------------------------------------

export type HandleName = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

export function rectFromCorners(a: Point, b: Point): Rect | null {
  const x = clamp01(Math.min(a[0], b[0]));
  const y = clamp01(Math.min(a[1], b[1]));
  const w = clamp01(Math.max(a[0], b[0])) - x;
  const h = clamp01(Math.max(a[1], b[1])) - y;
  return w >= MIN_RECT_SIZE && h >= MIN_RECT_SIZE ? { x, y, w, h } : null;
}

export function handlePositions(rect: Rect): Record<HandleName, Point> {
  const { x, y, w, h } = rect;
  return {
    nw: [x, y],
    n: [x + w / 2, y],
    ne: [x + w, y],
    e: [x + w, y + h / 2],
    se: [x + w, y + h],
    s: [x + w / 2, y + h],
    sw: [x, y + h],
    w: [x, y + h / 2],
  };
}

/** The handle within `tolerance` (a fraction of the frame width) of the point, nearest first. */
export function hitHandle(rect: Rect, p: Point, tolerance: number, aspect: number): HandleName | null {
  let best: { name: HandleName; distance: number } | null = null;
  for (const [name, position] of Object.entries(handlePositions(rect)) as [HandleName, Point][]) {
    const distance = Math.hypot(p[0] - position[0], (p[1] - position[1]) / aspect);
    if (distance <= tolerance && (best === null || distance < best.distance)) best = { name, distance };
  }
  return best?.name ?? null;
}

/** Drag one handle to `p`. Crossing the opposite edge flips the rectangle instead of inverting it. */
export function resizeRect(rect: Rect, handle: HandleName, p: Point): Rect {
  let left = rect.x;
  let right = rect.x + rect.w;
  let top = rect.y;
  let bottom = rect.y + rect.h;
  const px = clamp01(p[0]);
  const py = clamp01(p[1]);
  if (handle.includes("w")) left = px;
  if (handle.includes("e")) right = px;
  if (handle.includes("n")) top = py;
  if (handle.includes("s")) bottom = py;

  const x = Math.min(left, right);
  const y = Math.min(top, bottom);
  let w = Math.abs(right - left);
  let h = Math.abs(bottom - top);
  w = Math.max(w, MIN_RECT_SIZE);
  h = Math.max(h, MIN_RECT_SIZE);
  return { x: Math.min(x, 1 - w), y: Math.min(y, 1 - h), w, h };
}

export function translateRect(rect: Rect, dx: number, dy: number): Rect {
  return {
    ...rect,
    x: clamp(rect.x + dx, 0, 1 - rect.w),
    y: clamp(rect.y + dy, 0, 1 - rect.h),
  };
}

/** The largest shift toward (dx, dy) that keeps every point inside the frame. */
function safeShift(points: Point[], dx: number, dy: number): Point {
  if (points.length === 0) return [0, 0];
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return [
    clamp(dx, -Math.min(...xs), 1 - Math.max(...xs)),
    clamp(dy, -Math.min(...ys), 1 - Math.max(...ys)),
  ];
}

/** Shift every point together, stopping when any point would leave the frame so the shape keeps its form. */
export function translatePoints(points: Point[], dx: number, dy: number): Point[] {
  const [safeDx, safeDy] = safeShift(points, dx, dy);
  return points.map((p) => [p[0] + safeDx, p[1] + safeDy]);
}

/** Move all of a brush keyframe's paint and erasing together, keeping the whole thing in frame. */
export function translateBrush(keyframe: Keyframe, dx: number, dy: number): Keyframe {
  const strokes = keyframe.strokes ?? [];
  const erase = keyframe.erase ?? [];
  const [safeDx, safeDy] = safeShift([...strokes, ...erase].flatMap((s) => s.points), dx, dy);
  const move = (list: Stroke[]): Stroke[] =>
    list.map((s) => ({ ...s, points: s.points.map((p): Point => [p[0] + safeDx, p[1] + safeDy]) }));
  return { ...keyframe, strokes: move(strokes), ...(erase.length > 0 ? { erase: move(erase) } : {}) };
}

/** The box around a shape, for drawing a selection outline. `aspect` is frame width / height. */
export function shapeBounds(shape: Keyframe, aspect: number): Rect | null {
  if (shape.rect) return shape.rect;
  let xs: number[] = [];
  let ys: number[] = [];
  let padX = 0;
  let padY = 0;
  if (shape.points) {
    xs = shape.points.map((p) => p[0]);
    ys = shape.points.map((p) => p[1]);
  } else if (shape.strokes) {
    xs = shape.strokes.flatMap((s) => s.points.map((p) => p[0]));
    ys = shape.strokes.flatMap((s) => s.points.map((p) => p[1]));
    const radius = Math.max(0, ...shape.strokes.map((s) => s.radius));
    padX = radius;
    padY = radius / aspect;
  }
  if (xs.length === 0) return null;
  const x = clamp01(Math.min(...xs) - padX);
  const y = clamp01(Math.min(...ys) - padY);
  return { x, y, w: clamp01(Math.max(...xs) + padX) - x, h: clamp01(Math.max(...ys) + padY) - y };
}

export function pointInRect(p: Point, rect: Rect): boolean {
  return p[0] >= rect.x && p[0] <= rect.x + rect.w && p[1] >= rect.y && p[1] <= rect.y + rect.h;
}

/** Ray casting. Points exactly on an edge may fall either side, which doesn't matter for picking. */
export function pointInPolygon(p: Point, polygon: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Distance from `p` to the segment a-b, measured in units of frame width (y is scaled by 1/aspect). */
function distanceToSegment(p: Point, a: Point, b: Point, aspect: number): number {
  const px = p[0];
  const py = p[1] / aspect;
  const ax = a[0];
  const ay = a[1] / aspect;
  const dx = b[0] - ax;
  const dy = b[1] / aspect - ay;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared === 0 ? 0 : clamp(((px - ax) * dx + (py - ay) * dy) / lengthSquared, 0, 1);
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

export function pointInStroke(p: Point, stroke: Stroke, aspect: number): boolean {
  const { points, radius } = stroke;
  if (points.length === 1) return distanceToSegment(p, points[0], points[0], aspect) <= radius;
  for (let i = 1; i < points.length; i += 1) {
    if (distanceToSegment(p, points[i - 1], points[i], aspect) <= radius) return true;
  }
  return false;
}

/** Is `p` inside what this region covers on `frame`? `aspect` is frame width / height. */
export function hitRegion(
  region: Region,
  frame: number,
  p: Point,
  aspect: number,
  resolve: ShapeResolver = shapeAt,
): boolean {
  const shape = resolve(region, frame);
  if (shape.hidden) return false;
  let covered = false;
  if (shape.rect) covered = pointInRect(p, shape.rect);
  else if (shape.points) covered = pointInPolygon(p, shape.points);
  else if (shape.strokes) covered = shape.strokes.some((s) => pointInStroke(p, s, aspect));
  return covered && !(shape.erase ?? []).some((s) => pointInStroke(p, s, aspect));
}

/** The top-most region under `p`, or null. Later regions are drawn on top, so they're checked first. */
export function pickRegion(
  doc: MaskDocument,
  frame: number,
  p: Point,
  aspect: number,
  resolve: ShapeResolver = shapeAt,
): Region | null {
  for (let i = doc.regions.length - 1; i >= 0; i -= 1) {
    if (hitRegion(doc.regions[i], frame, p, aspect, resolve)) return doc.regions[i];
  }
  return null;
}

/** Drop points that are closer than `minDistance` to the previous kept one, always keeping the last. */
export function simplifyStroke(points: Point[], minDistance = 0.004): Point[] {
  if (points.length <= 2) return points;
  const kept: Point[] = [points[0]];
  for (let i = 1; i < points.length - 1; i += 1) {
    const last = kept[kept.length - 1];
    if (Math.hypot(points[i][0] - last[0], points[i][1] - last[1]) >= minDistance) kept.push(points[i]);
  }
  kept.push(points[points.length - 1]);
  return kept.slice(0, MAX_POINTS_PER_STROKE);
}

// --- limits -------------------------------------------------------------------------------------

export function countPoints(doc: MaskDocument): number {
  let total = 0;
  for (const region of doc.regions) {
    for (const k of region.keyframes) {
      total += k.points?.length ?? 0;
      total += (k.strokes ?? []).reduce((sum, s) => sum + s.points.length, 0);
      total += (k.erase ?? []).reduce((sum, s) => sum + s.points.length, 0);
    }
  }
  return total;
}

/** The first problem that would make the server refuse this document, in words, or null. */
export function problemWithDocument(doc: MaskDocument, frameCount: number): string | null {
  if (doc.regions.length > MAX_REGIONS) return `A project can have at most ${MAX_REGIONS} regions.`;
  for (const region of doc.regions) {
    if (region.keyframes.length > MAX_KEYFRAMES_PER_REGION) {
      return `"${region.name}" has more than ${MAX_KEYFRAMES_PER_REGION} keyframes.`;
    }
    const last = region.keyframes[region.keyframes.length - 1];
    if (last && last.frame >= frameCount) return `"${region.name}" has a keyframe past the end of the video.`;
    for (const k of region.keyframes) {
      if ((k.points?.length ?? 0) > MAX_POLYGON_POINTS) return `"${region.name}" has too many polygon points.`;
      if ((k.strokes?.length ?? 0) > MAX_STROKES_PER_KEYFRAME || (k.erase?.length ?? 0) > MAX_STROKES_PER_KEYFRAME) {
        return `"${region.name}" has too many brush strokes on one frame.`;
      }
    }
  }
  if (countPoints(doc) > MAX_TOTAL_POINTS) {
    return "This mask is too detailed to save. Remove some keyframes or strokes.";
  }
  return null;
}

export function strokeRadius(widthFraction: number): number {
  return clamp(widthFraction, MIN_RADIUS, MAX_RADIUS);
}
