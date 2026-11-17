/**
 * What a region looks like on a frame once tracking has run: the TypeScript twin of
 * backend/app/tracking/shapes.py. Both are tested against backend/tests/fixtures/shape_cases.json.
 *
 * Kept free of React and the DOM so it can be tested directly.
 */

import { shapeAt } from "./masks";
import type { Keyframe, MaskDocument, Point, Region, ShapeResolver, Stroke } from "../types/mask";
import type { LostArea, TrackingData, TrackSegment, Transform } from "../types/tracking";

const IDENTITY: Transform = [0, 0, 1, 0];

/** Centre of the shape's bounding box, in normalised coordinates. The brush radius isn't included. */
export function shapeCenter(shape: Keyframe): Point {
  if (shape.rect) return [shape.rect.x + shape.rect.w / 2, shape.rect.y + shape.rect.h / 2];
  const points: Point[] = shape.points ?? (shape.strokes ?? []).flatMap((s) => s.points);
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
}

function movePoint(p: Point, centre: Point, [dx, dy, scale, angle]: Transform, aspect: number): Point {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  // Work in frame-width units: y is scaled by the frame's height / width ratio.
  const rx = p[0] - centre[0];
  const ry = (p[1] - centre[1]) * aspect;
  return [
    centre[0] + dx + scale * (cos * rx - sin * ry),
    (centre[1] * aspect + dy + scale * (sin * rx + cos * ry)) / aspect,
  ];
}

/** Move a keyframe's shape by one tracked transform. A rectangle stays a rectangle, clipped to the frame. */
export function transformShape(shape: Keyframe, transform: Transform, aspect: number): Keyframe {
  if (transform.every((v, i) => v === IDENTITY[i])) return shape;
  const centre = shapeCenter(shape);
  const [dx, dy, scale] = transform;
  const strokes = (items: Stroke[]): Stroke[] =>
    items.map((s) => ({ radius: s.radius * scale, points: s.points.map((p) => movePoint(p, centre, transform, aspect)) }));
  const erase = strokes(shape.erase ?? []);

  if (shape.rect) {
    const cx = centre[0] + dx;
    const cy = centre[1] + dy / aspect;
    const w = shape.rect.w * scale;
    const h = shape.rect.h * scale;
    const left = Math.max(0, cx - w / 2);
    const top = Math.max(0, cy - h / 2);
    const right = Math.min(1, cx + w / 2);
    const bottom = Math.min(1, cy + h / 2);
    return { frame: shape.frame, rect: { x: left, y: top, w: Math.max(0, right - left), h: Math.max(0, bottom - top) }, erase };
  }
  if (shape.points) {
    return { frame: shape.frame, points: shape.points.map((p) => movePoint(p, centre, transform, aspect)), erase };
  }
  return { frame: shape.frame, strokes: strokes(shape.strokes ?? []), erase };
}

function covers(segment: TrackSegment, frame: number): boolean {
  const distance = (frame - segment.anchor) * segment.step;
  return distance >= 1 && distance <= Math.abs(segment.until - segment.anchor);
}

function transformAt(segment: TrackSegment, frame: number): Transform {
  if (segment.transforms.length === 0) return IDENTITY;
  const distance = (frame - segment.anchor) * segment.step;
  return segment.transforms[Math.min(distance, segment.transforms.length) - 1];
}

/** A keyframe's own shape exactly, else tracked motion, else keyframes blended. */
export function trackedShapeAt(region: Region, frame: number, segments: TrackSegment[] | undefined, aspect: number): Keyframe {
  const exact = region.keyframes.find((k) => k.frame === frame);
  if (exact) return exact;
  for (const segment of segments ?? []) {
    if (!covers(segment, frame)) continue;
    const anchor = region.keyframes.find((k) => k.frame === segment.anchor);
    if (anchor) return { ...transformShape(anchor, transformAt(segment, frame), aspect), frame };
  }
  return shapeAt(region, frame);
}

/** A resolver that follows tracked motion, or plain keyframes when there is none. */
export function resolverFor(data: TrackingData | null): ShapeResolver {
  if (!data) return shapeAt;
  return (region, frame) => trackedShapeAt(region, frame, data.regions[region.id]?.segments, data.aspect);
}

/** Every place tracking gave up, in frame order, for regions that still exist. */
export function lostAreas(data: TrackingData | null, doc: MaskDocument): LostArea[] {
  if (!data) return [];
  const areas: LostArea[] = [];
  for (const region of doc.regions) {
    for (const segment of data.regions[region.id]?.segments ?? []) {
      if (segment.lost_at === undefined) continue;
      areas.push({ regionId: region.id, regionName: region.name, frame: segment.lost_at, reason: segment.reason ?? "Tracking lost this area here." });
    }
  }
  return areas.sort((a, b) => a.frame - b.frame || a.regionName.localeCompare(b.regionName));
}

/** The share of a running job that is done, or null before it has said how big it is (never a guess). */
export function trackingFraction(framesDone: number, framesTotal: number): number | null {
  if (framesTotal <= 0) return null;
  return Math.min(1, Math.max(0, framesDone / framesTotal));
}
