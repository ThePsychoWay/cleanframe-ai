/**
 * Maths and rules for the before/after view: the divider, zoom and pan, keeping two videos in step, and when a
 * signed link needs replacing. Kept free of React and the DOM so it can be tested directly.
 */

export const ZOOM_LEVELS = [1, 2, 4] as const;
export type Zoom = (typeof ZOOM_LEVELS)[number];

export interface Size {
  width: number;
  height: number;
}

export interface Offset {
  x: number;
  y: number;
}

export function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 50;
  return Math.min(100, Math.max(0, value));
}

/** Where a pointer is along the picture, as a percentage from its left edge. */
export function dividerFromPointer(clientX: number, left: number, width: number): number {
  if (width <= 0) return 50;
  return clampPercent(((clientX - left) / width) * 100);
}

/** Move the divider with the keyboard: 2% a press, 10% with shift, Home/End to the ends. Null for other keys. */
export function dividerFromKey(current: number, key: string, shift: boolean): number | null {
  const step = shift ? 10 : 2;
  switch (key) {
    case "ArrowLeft":
    case "ArrowDown":
      return clampPercent(current - step);
    case "ArrowRight":
    case "ArrowUp":
      return clampPercent(current + step);
    case "Home":
      return 0;
    case "End":
      return 100;
    default:
      return null;
  }
}

export function nextZoom(current: Zoom, direction: 1 | -1): Zoom {
  const index = ZOOM_LEVELS.indexOf(current);
  return ZOOM_LEVELS[Math.min(ZOOM_LEVELS.length - 1, Math.max(0, index + direction))];
}

/** The picture can be dragged only far enough that its edge reaches the frame's edge, never into empty space. */
export function clampPan(offset: Offset, zoom: number, box: Size): Offset {
  const limitX = ((zoom - 1) * box.width) / 2;
  const limitY = ((zoom - 1) * box.height) / 2;
  const clamp = (value: number, limit: number) => Math.min(limit, Math.max(-limit, value));
  return { x: limit(clamp(offset.x, limitX)), y: limit(clamp(offset.y, limitY)) };
}

// `-0` would show up as a different value to equality checks; keep zero as zero.
function limit(value: number): number {
  return value === 0 ? 0 : value;
}

/** Zooming about the middle of the frame: shrink the pan with the zoom so the same spot stays in view. */
export function panForZoom(offset: Offset, from: number, to: number, box: Size): Offset {
  if (to <= 1) return { x: 0, y: 0 };
  const ratio = (to - 1) / Math.max(1, from - 1);
  const scaled = from <= 1 ? { x: 0, y: 0 } : { x: offset.x * ratio, y: offset.y * ratio };
  return clampPan(scaled, to, box);
}

/**
 * Should the follower video be moved to match the master? While playing, a little drift is normal and re-seeking
 * causes a visible stutter, so only correct beyond 1.5 frames; when paused the two must show the same frame.
 */
export function needsResync(masterTime: number, followerTime: number, fps: number, playing: boolean): boolean {
  if (!Number.isFinite(masterTime) || !Number.isFinite(followerTime) || fps <= 0) return false;
  const tolerance = (playing ? 1.5 : 0.5) / fps;
  return Math.abs(masterTime - followerTime) > tolerance;
}

/** True when a signed link has used up most of its life and a fresh one should be fetched. */
export function linkNeedsRefresh(fetchedAtMs: number, expiresInSeconds: number, nowMs: number, marginSeconds = 300): boolean {
  return nowMs - fetchedAtMs >= Math.max(0, expiresInSeconds - marginSeconds) * 1000;
}
