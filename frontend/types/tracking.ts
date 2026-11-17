/**
 * Mask tracking, mirroring backend/app/schemas/tracking.py and backend/app/tracking/shapes.py.
 *
 * Tracking stores one small transform per frame, applied to the shape of the keyframe it started from:
 * [dx, dy, scale, angle], in frame-width units, about the centre of that shape's bounding box.
 */

export type TrackingStatus = "none" | "running" | "done" | "failed";

export type Transform = [dx: number, dy: number, scale: number, angle: number];

export interface TrackSegment {
  /** The keyframe this was tracked from. */
  anchor: number;
  /** 1 for frames after the anchor, -1 for frames before it. */
  step: 1 | -1;
  /** The last frame this segment is responsible for. */
  until: number;
  /** transforms[i] is the motion on frame anchor + step * (i + 1). */
  transforms: Transform[];
  /** Set when tracking gave up: the first frame it couldn't follow. */
  lost_at?: number;
  reason?: string;
}

export interface TrackingData {
  version: 1;
  width: number;
  height: number;
  aspect: number;
  frame_count: number;
  regions: Record<string, { segments: TrackSegment[] }>;
}

export interface TrackingResponse {
  status: TrackingStatus;
  frames_done: number;
  frames_total: number;
  error: string | null;
  started_at: string | null;
  data: TrackingData | null;
}

/** A place where tracking couldn't continue and the person should correct the mask. */
export interface LostArea {
  regionId: string;
  regionName: string;
  frame: number;
  reason: string;
}
