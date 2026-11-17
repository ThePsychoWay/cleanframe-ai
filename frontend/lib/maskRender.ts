/**
 * Draws the mask overlay onto a canvas that sits over the video.
 *
 * Everything is derived from the document, the current frame and the in-progress gesture, so the
 * canvas is a pure picture of state and never holds anything of its own.
 */

import { findRegion, handlePositions, shapeBounds } from "./masks";
import type { Keyframe, MaskDocument, Point, Rect, ShapeResolver, Stroke } from "../types/mask";

export type Tool = "select" | "rectangle" | "polygon" | "brush" | "eraser";

/** A gesture that hasn't been committed to the document yet. */
export type Draft =
  | { kind: "rect"; a: Point; b: Point }
  | { kind: "polygon"; points: Point[]; hover: Point | null }
  | { kind: "stroke"; points: Point[]; erase: boolean; radius: number };

export interface Scene {
  doc: MaskDocument;
  /** The shape each region has on this frame (keyframes, plus tracked motion once tracking has run). */
  resolve: ShapeResolver;
  frame: number;
  selectedId: string | null;
  draft: Draft | null;
  /** Pointer position over the canvas, for the brush cursor and the polygon's rubber band. */
  hover: Point | null;
  tool: Tool;
  brushRadius: number;
  /** Moves the dashes of selection borders; changing it over time makes them march. 0 = still. */
  dashOffset?: number;
}

const ACCENT = "183, 243, 74"; // --color-mask
const FILL = `rgba(${ACCENT}, 0.28)`;
const OUTLINE = `rgb(${ACCENT})`;
export const HANDLE_PX = 6;

function strokePath(ctx: CanvasRenderingContext2D, stroke: Stroke, w: number, h: number): void {
  const [first, ...rest] = stroke.points;
  if (!first) return;
  if (rest.length === 0) {
    ctx.beginPath();
    ctx.arc(first[0] * w, first[1] * h, stroke.radius * w, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.lineWidth = stroke.radius * 2 * w;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(first[0] * w, first[1] * h);
  for (const p of rest) ctx.lineTo(p[0] * w, p[1] * h);
  ctx.stroke();
}

function polygonPath(ctx: CanvasRenderingContext2D, points: Point[], w: number, h: number): void {
  ctx.beginPath();
  points.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0] * w, p[1] * h) : ctx.lineTo(p[0] * w, p[1] * h)));
  ctx.closePath();
}

/** Fill one keyframe's shape and cut its erase strokes out of it. Draws onto `ctx` as-is. */
function paintShape(ctx: CanvasRenderingContext2D, shape: Keyframe, w: number, h: number): void {
  ctx.fillStyle = FILL;
  ctx.strokeStyle = FILL;
  if (shape.rect) ctx.fillRect(shape.rect.x * w, shape.rect.y * h, shape.rect.w * w, shape.rect.h * h);
  else if (shape.points) {
    polygonPath(ctx, shape.points, w, h);
    ctx.fill();
  } else if (shape.strokes) for (const s of shape.strokes) strokePath(ctx, s, w, h);

  if (shape.erase && shape.erase.length > 0) {
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "#000";
    ctx.strokeStyle = "#000";
    for (const s of shape.erase) strokePath(ctx, s, w, h);
    ctx.restore();
  }
}

function drawHandle(ctx: CanvasRenderingContext2D, x: number, y: number, round: boolean): void {
  ctx.fillStyle = "#f5f5f5";
  ctx.strokeStyle = OUTLINE;
  ctx.lineWidth = 2;
  ctx.beginPath();
  if (round) ctx.arc(x, y, HANDLE_PX, 0, Math.PI * 2);
  else ctx.rect(x - HANDLE_PX, y - HANDLE_PX, HANDLE_PX * 2, HANDLE_PX * 2);
  ctx.fill();
  ctx.stroke();
}

function drawBox(ctx: CanvasRenderingContext2D, box: Rect, w: number, h: number, dashed: boolean, offset = 0): void {
  ctx.save();
  ctx.lineDashOffset = -offset;
  ctx.strokeStyle = OUTLINE;
  ctx.lineWidth = dashed ? 1.5 : 2;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.strokeRect(box.x * w, box.y * h, box.w * w, box.h * h);
  ctx.restore();
}

function drawDraft(ctx: CanvasRenderingContext2D, draft: Draft, w: number, h: number): void {
  ctx.save();
  if (draft.kind === "rect") {
    const x = Math.min(draft.a[0], draft.b[0]) * w;
    const y = Math.min(draft.a[1], draft.b[1]) * h;
    const rw = Math.abs(draft.a[0] - draft.b[0]) * w;
    const rh = Math.abs(draft.a[1] - draft.b[1]) * h;
    ctx.fillStyle = FILL;
    ctx.fillRect(x, y, rw, rh);
    ctx.strokeStyle = OUTLINE;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, rw, rh);
  } else if (draft.kind === "polygon") {
    const pts = draft.hover ? [...draft.points, draft.hover] : draft.points;
    ctx.strokeStyle = OUTLINE;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0] * w, p[1] * h) : ctx.lineTo(p[0] * w, p[1] * h)));
    ctx.stroke();
    draft.points.forEach((p, i) => drawHandle(ctx, p[0] * w, p[1] * h, i !== 0)); // the first point is square: click it to close
  } else {
    const stroke: Stroke = { radius: draft.radius, points: draft.points };
    if (draft.erase) {
      ctx.fillStyle = "rgba(245, 245, 245, 0.45)";
      ctx.strokeStyle = "rgba(245, 245, 245, 0.45)";
    } else {
      ctx.fillStyle = FILL;
      ctx.strokeStyle = FILL;
    }
    strokePath(ctx, stroke, w, h);
  }
  ctx.restore();
}

/**
 * Paint the whole scene. `layer` is a scratch canvas of the same size, used so that one region's erase
 * strokes only cut that region and never another's.
 */
export function drawScene(
  ctx: CanvasRenderingContext2D,
  layer: CanvasRenderingContext2D,
  cssWidth: number,
  cssHeight: number,
  scene: Scene,
): void {
  const w = cssWidth;
  const h = cssHeight;
  const aspect = w / h;
  ctx.clearRect(0, 0, w, h);

  for (const region of scene.doc.regions) {
    const shape = scene.resolve(region, scene.frame);
    if (shape.hidden) continue;
    layer.clearRect(0, 0, w, h);
    paintShape(layer, shape, w, h);
    ctx.drawImage(layer.canvas, 0, 0, w, h);
  }

  const selected = findRegion(scene.doc, scene.selectedId);
  for (const region of scene.doc.regions) {
    const shape = scene.resolve(region, scene.frame);
    if (shape.hidden) continue;
    const isSelected = region.id === selected?.id;
    if (shape.rect) drawBox(ctx, shape.rect, w, h, !isSelected, scene.dashOffset);
    else if (shape.points) {
      ctx.save();
      ctx.strokeStyle = OUTLINE;
      ctx.lineWidth = isSelected ? 2 : 1.5;
      ctx.setLineDash(isSelected ? [] : [6, 4]);
      ctx.lineDashOffset = -(scene.dashOffset ?? 0);
      polygonPath(ctx, shape.points, w, h);
      ctx.stroke();
      ctx.restore();
    } else if (isSelected) {
      const bounds = shapeBounds(shape, aspect);
      if (bounds) drawBox(ctx, bounds, w, h, true, scene.dashOffset);
    }
  }

  if (selected && scene.tool === "select" && !scene.draft) {
    const shape = scene.resolve(selected, scene.frame);
    if (shape.hidden) {
      // nothing on screen to adjust
    } else if (shape.rect) {
      for (const [x, y] of Object.values(handlePositions(shape.rect))) drawHandle(ctx, x * w, y * h, false);
    } else if (shape.points) {
      for (const [x, y] of shape.points) drawHandle(ctx, x * w, y * h, true);
    }
  }

  if (scene.draft) drawDraft(ctx, scene.draft, w, h);

  if ((scene.tool === "brush" || scene.tool === "eraser") && scene.hover && !scene.draft) {
    ctx.save();
    ctx.strokeStyle = scene.tool === "eraser" ? "rgba(245, 245, 245, 0.9)" : OUTLINE;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(scene.hover[0] * w, scene.hover[1] * h, scene.brushRadius * w, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}
