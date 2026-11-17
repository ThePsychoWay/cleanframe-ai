/**
 * The mask document, mirroring backend/app/schemas/mask.py.
 *
 * Coordinates are fractions of the frame (0 to 1 from the top-left), so a mask doesn't depend on the
 * proxy's or the source's resolution. Brush radii are fractions of the frame width.
 */

export type Point = [number, number];

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Stroke {
  radius: number;
  points: Point[];
}

/** One shape at one source frame. A region's keyframes all use the same kind of shape. */
export interface Keyframe {
  frame: number;
  rect?: Rect;
  points?: Point[];
  strokes?: Stroke[];
  /** Subtracted from this keyframe's shape. */
  erase?: Stroke[];
  /** The region isn't on screen from this frame until its next keyframe (an overlay that comes and goes). */
  hidden?: boolean;
}

export type RegionType = "rectangle" | "polygon" | "brush";

export interface Region {
  id: string;
  name: string;
  type: RegionType;
  keyframes: Keyframe[];
}

export interface MaskDocument {
  version: 1;
  regions: Region[];
}

export interface MaskResponse {
  mask_data: MaskDocument;
  tracking_status: string;
  updated_at: string | null;
}

export interface MediaInfo {
  proxy_url: string;
  thumbnail_url: string;
  expires_in: number;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration: number;
}

/**
 * Gives the shape a region has on a frame. The default is keyframes only (shapeAt); after tracking it
 * also follows the tracked motion (see lib/tracking.ts).
 */
export type ShapeResolver = (region: Region, frame: number) => Keyframe;
