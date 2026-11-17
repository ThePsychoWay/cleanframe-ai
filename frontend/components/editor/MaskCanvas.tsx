"use client";

import { useCallback, useEffect, useRef, type PointerEvent, type ReactNode, type RefObject } from "react";
import { drawScene, type Draft, type Scene, type Tool } from "@/lib/maskRender";
import {
  MAX_POINTS_PER_STROKE,
  MAX_POLYGON_POINTS,
  MAX_REGIONS,
  MAX_STROKES_PER_KEYFRAME,
  addRegion,
  clamp01,
  createRegion,
  editShapeAt,
  findRegion,
  hitHandle,
  pickRegion,
  pointInPolygon,
  pointInRect,
  rectFromCorners,
  resizeRect,
  hitRegion,
  simplifyStroke,
  strokeRadius,
  translateBrush,
  translatePoints,
  translateRect,
} from "@/lib/masks";
import type { HandleName } from "@/lib/masks";
import type { Keyframe, MaskDocument, MediaInfo, Point, ShapeResolver, Stroke } from "@/types/mask";

interface MaskCanvasProps {
  media: MediaInfo;
  doc: MaskDocument;
  /** Where each region is on a frame: keyframes, plus tracked motion once tracking has run. */
  resolve: ShapeResolver;
  /** While tracking runs the marks can't change, so the canvas only displays them. */
  locked: boolean;
  frame: number;
  tool: Tool;
  selectedId: string | null;
  /** Brush and eraser radius, as a fraction of the frame width. */
  brushRadius: number;
  videoRef: RefObject<HTMLVideoElement | null>;
  onChange: (doc: MaskDocument, selectId?: string | null) => void;
  onSelect: (id: string | null) => void;
  onToolChange: (tool: Tool) => void;
  onNotice: (message: string | null) => void;
  /** Called when a gesture starts, so playback can pause: editing a moving picture isn't possible. */
  onInteract: () => void;
  onVideoError: () => void;
  /** Drawn over the picture, sharing its size (suggestions, for example). */
  children?: ReactNode;
}

type Drag =
  | { kind: "resize"; id: string; handle: HandleName; base: Keyframe }
  | { kind: "move"; id: string; start: Point; base: Keyframe }
  | { kind: "vertex"; id: string; index: number; base: Keyframe };

const HANDLE_TOLERANCE_PX = { mouse: 11, pen: 14, touch: 20 } as const;

export function MaskCanvas(props: MaskCanvasProps) {
  const { media, tool, videoRef } = props;
  const aspect = media.width / media.height;

  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layerRef = useRef<HTMLCanvasElement | null>(null);
  const sizeRef = useRef({ width: 0, height: 0, dpr: 1 });
  const frameRequest = useRef<number | null>(null);

  // Gesture state lives in refs: it changes on every pointer move and shouldn't re-render the page.
  const draftRef = useRef<Draft | null>(null);
  const dragRef = useRef<Drag | null>(null);
  const previewRef = useRef<MaskDocument | null>(null);
  const hoverRef = useRef<Point | null>(null);
  // Marching dashes on the selection borders, unless the person asked for less motion.
  const motionRef = useRef(false);

  // Handlers always read the newest props through this, so they never act on a stale document.
  const latest = useRef(props);
  latest.current = props;

  const draw = useCallback(() => {
    frameRequest.current = null;
    const canvas = canvasRef.current;
    const layerCanvas = layerRef.current;
    const { width, height, dpr } = sizeRef.current;
    if (!canvas || !layerCanvas || width === 0) return;
    const ctx = canvas.getContext("2d");
    const layer = layerCanvas.getContext("2d");
    if (!ctx || !layer) return;
    const p = latest.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layer.setTransform(dpr, 0, 0, dpr, 0, 0);
    const scene: Scene = {
      doc: previewRef.current ?? p.doc,
      resolve: p.resolve,
      frame: p.frame,
      selectedId: p.selectedId,
      draft: draftRef.current,
      hover: hoverRef.current,
      tool: p.tool,
      brushRadius: strokeRadius(p.brushRadius),
      dashOffset: motionRef.current ? (performance.now() / 70) % 10 : 0,
    };
    drawScene(ctx, layer, width, height, scene);
  }, []);

  const scheduleDraw = useCallback(() => {
    if (frameRequest.current === null) frameRequest.current = requestAnimationFrame(draw);
  }, [draw]);

  // Keep the canvas matched to its on-screen size, sharp on high-density screens.
  useEffect(() => {
    const wrapper = wrapperRef.current;
    const canvas = canvasRef.current;
    if (!wrapper || !canvas) return;
    layerRef.current ??= document.createElement("canvas");
    const layer = layerRef.current;

    const resize = () => {
      const { width, height } = wrapper.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      sizeRef.current = { width, height, dpr };
      for (const c of [canvas, layer]) {
        c.width = Math.round(width * dpr);
        c.height = Math.round(height * dpr);
      }
      scheduleDraw();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(wrapper);
    return () => {
      observer.disconnect();
      if (frameRequest.current !== null) cancelAnimationFrame(frameRequest.current);
    };
  }, [scheduleDraw]);

  const hasRegions = props.doc.regions.length > 0;
  useEffect(() => {
    motionRef.current = hasRegions && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!motionRef.current) {
      scheduleDraw();
      return;
    }
    const timer = window.setInterval(scheduleDraw, 90);
    return () => window.clearInterval(timer);
  }, [hasRegions, scheduleDraw]);

  // Redraw whenever what's on screen changes.
  useEffect(() => {
    scheduleDraw();
  }, [props.doc, props.resolve, props.frame, props.tool, props.selectedId, props.brushRadius, scheduleDraw]);

  // Half-finished gestures don't survive a tool or frame change.
  useEffect(() => {
    draftRef.current = null;
    dragRef.current = null;
    previewRef.current = null;
    scheduleDraw();
  }, [props.tool, props.frame, scheduleDraw]);

  const toPoint = (event: PointerEvent): Point => {
    const box = event.currentTarget.getBoundingClientRect();
    return [clamp01((event.clientX - box.left) / box.width), clamp01((event.clientY - box.top) / box.height)];
  };

  const toleranceFor = (event: PointerEvent): number => {
    const kind = event.pointerType === "touch" ? "touch" : event.pointerType === "pen" ? "pen" : "mouse";
    return HANDLE_TOLERANCE_PX[kind] / Math.max(1, sizeRef.current.width);
  };

  // --- committing gestures --------------------------------------------------------------------

  const startRegion = (type: "rectangle" | "polygon" | "brush", shape: Omit<Keyframe, "frame">, keepTool = false) => {
    const p = latest.current;
    if (p.doc.regions.length >= MAX_REGIONS) {
      p.onNotice(`A project can have at most ${MAX_REGIONS} regions. Remove one to add another.`);
      return;
    }
    const region = createRegion(p.doc, type, { frame: p.frame, ...shape });
    p.onChange(addRegion(p.doc, region), region.id);
    if (!keepTool) p.onToolChange("select");
  };

  const closePolygon = () => {
    const draft = draftRef.current;
    if (draft?.kind !== "polygon") return;
    if (draft.points.length < 3) {
      latest.current.onNotice("A polygon needs at least 3 points.");
      return;
    }
    draftRef.current = null;
    startRegion("polygon", { points: draft.points });
    scheduleDraw();
  };

  const commitStroke = (draft: Extract<Draft, { kind: "stroke" }>) => {
    const p = latest.current;
    const stroke: Stroke = { radius: strokeRadius(p.brushRadius), points: simplifyStroke(draft.points).slice(0, MAX_POINTS_PER_STROKE) };
    const region = findRegion(p.doc, p.selectedId);

    if (draft.erase) {
      if (!region) {
        p.onNotice("Select a region first, then erase from it.");
        return;
      }
      if ((p.resolve(region, p.frame).erase?.length ?? 0) >= MAX_STROKES_PER_KEYFRAME) {
        p.onNotice("This frame has too many erase strokes. Undo some or start a new keyframe on another frame.");
        return;
      }
      p.onChange(editShapeAt(p.doc, region.id, p.frame, (k) => ({ ...k, erase: [...(k.erase ?? []), stroke] }), p.resolve));
      return;
    }

    if (region?.type === "brush") {
      if ((p.resolve(region, p.frame).strokes?.length ?? 0) >= MAX_STROKES_PER_KEYFRAME) {
        p.onNotice("This frame has too many brush strokes. Undo some or paint on another frame.");
        return;
      }
      p.onChange(editShapeAt(p.doc, region.id, p.frame, (k) => ({ ...k, strokes: [...(k.strokes ?? []), stroke] }), p.resolve));
    } else {
      startRegion("brush", { strokes: [stroke] }, true);
    }
  };

  const finishDrag = () => {
    const preview = previewRef.current;
    dragRef.current = null;
    previewRef.current = null;
    if (preview && preview !== latest.current.doc) latest.current.onChange(preview);
    scheduleDraw();
  };

  // --- pointer events -------------------------------------------------------------------------

  const onPointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    if (latest.current.locked) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const p = latest.current;
    p.onInteract();
    p.onNotice(null);
    const point = toPoint(event);
    hoverRef.current = point;

    if (p.tool === "rectangle") {
      draftRef.current = { kind: "rect", a: point, b: point };
    } else if (p.tool === "brush" || p.tool === "eraser") {
      draftRef.current = { kind: "stroke", points: [point], erase: p.tool === "eraser", radius: strokeRadius(p.brushRadius) };
    } else if (p.tool === "polygon") {
      const draft = draftRef.current?.kind === "polygon" ? draftRef.current : { kind: "polygon" as const, points: [], hover: point };
      const first = draft.points[0];
      const tolerance = toleranceFor(event);
      if (first && draft.points.length >= 3 && Math.hypot(point[0] - first[0], (point[1] - first[1]) / aspect) <= tolerance) {
        draftRef.current = draft;
        closePolygon();
        return;
      }
      if (draft.points.length >= MAX_POLYGON_POINTS) {
        p.onNotice(`A polygon can have at most ${MAX_POLYGON_POINTS} points.`);
        return;
      }
      draftRef.current = { ...draft, points: [...draft.points, point] };
    } else {
      beginSelectGesture(point, toleranceFor(event));
    }
    scheduleDraw();
  };

  const beginSelectGesture = (point: Point, tolerance: number) => {
    const p = latest.current;
    const selected = findRegion(p.doc, p.selectedId);
    if (selected) {
      const shape = p.resolve(selected, p.frame);
      if (shape.rect) {
        const handle = hitHandle(shape.rect, point, tolerance, aspect);
        if (handle) {
          dragRef.current = { kind: "resize", id: selected.id, handle, base: shape };
          return;
        }
        if (pointInRect(point, shape.rect)) {
          dragRef.current = { kind: "move", id: selected.id, start: point, base: shape };
          return;
        }
      } else if (shape.points) {
        const index = shape.points.findIndex((q) => Math.hypot(point[0] - q[0], (point[1] - q[1]) / aspect) <= tolerance);
        if (index >= 0) {
          dragRef.current = { kind: "vertex", id: selected.id, index, base: shape };
          return;
        }
        if (pointInPolygon(point, shape.points)) {
          dragRef.current = { kind: "move", id: selected.id, start: point, base: shape };
          return;
        }
      } else if (hitRegion(selected, p.frame, point, aspect, p.resolve)) {
        dragRef.current = { kind: "move", id: selected.id, start: point, base: shape };
        return;
      }
    }

    // Otherwise pick whatever is under the pointer, and let the same press start moving it.
    const hit = pickRegion(p.doc, p.frame, point, aspect, p.resolve);
    p.onSelect(hit?.id ?? null);
    if (hit) dragRef.current = { kind: "move", id: hit.id, start: point, base: p.resolve(hit, p.frame) };
  };

  const onPointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const p = latest.current;
    const point = toPoint(event);
    hoverRef.current = point;

    const drag = dragRef.current;
    const draft = draftRef.current;
    if (drag) {
      const dx = point[0] - (drag.kind === "move" ? drag.start[0] : 0);
      const dy = point[1] - (drag.kind === "move" ? drag.start[1] : 0);
      previewRef.current = editShapeAt(p.doc, drag.id, p.frame, () => {
        const base = drag.base;
        if (drag.kind === "resize" && base.rect) return { ...base, rect: resizeRect(base.rect, drag.handle, point) };
        if (drag.kind === "vertex" && base.points) {
          return { ...base, points: base.points.map((q, i): Point => (i === drag.index ? point : q)) };
        }
        if (drag.kind === "move") {
          if (base.rect) return { ...base, rect: translateRect(base.rect, dx, dy) };
          if (base.points) return { ...base, points: translatePoints(base.points, dx, dy) };
          return translateBrush(base, dx, dy);
        }
        return base;
      }, p.resolve);
    } else if (draft?.kind === "rect") {
      draftRef.current = { ...draft, b: point };
    } else if (draft?.kind === "stroke" && event.buttons !== 0) {
      const last = draft.points[draft.points.length - 1];
      if (draft.points.length < MAX_POINTS_PER_STROKE && Math.hypot(point[0] - last[0], point[1] - last[1]) > 0.0015) {
        draftRef.current = { ...draft, points: [...draft.points, point] };
      }
    } else if (draft?.kind === "polygon") {
      draftRef.current = { ...draft, hover: point };
    }
    scheduleDraw();
  };

  const onPointerUp = (event: PointerEvent<HTMLCanvasElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    const draft = draftRef.current;

    if (dragRef.current) {
      finishDrag();
    } else if (draft?.kind === "rect") {
      draftRef.current = null;
      const rect = rectFromCorners(draft.a, draft.b);
      if (rect) startRegion("rectangle", { rect });
      else latest.current.onNotice("Drag across the area to draw a rectangle.");
    } else if (draft?.kind === "stroke") {
      draftRef.current = null;
      commitStroke(draft);
    }
    scheduleDraw();
  };

  const onPointerLeave = () => {
    hoverRef.current = null;
    scheduleDraw();
  };

  const onDoubleClick = () => {
    const draft = draftRef.current;
    if (draft?.kind !== "polygon") return;
    // The two presses of a double-click each placed a point in the same spot; keep one.
    draftRef.current = { ...draft, points: draft.points.slice(0, -1) };
    closePolygon();
  };

  // Polygon keys: Enter finishes, Escape cancels, Backspace removes the last point.
  useEffect(() => {
    if (tool !== "polygon") return;
    const onKey = (event: KeyboardEvent) => {
      const draft = draftRef.current;
      if (draft?.kind !== "polygon") return;
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.key === "Enter") {
        event.preventDefault();
        closePolygon();
      } else if (event.key === "Escape") {
        draftRef.current = null;
        scheduleDraw();
      } else if (event.key === "Backspace") {
        event.preventDefault();
        draftRef.current = { ...draft, points: draft.points.slice(0, -1) };
        scheduleDraw();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // closePolygon only reads refs and `latest`, so it is safe to leave out.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, scheduleDraw]);

  return (
    <div
      ref={wrapperRef}
      className="relative mx-auto select-none overflow-hidden rounded-lg bg-black"
      style={{
        aspectRatio: `${media.width} / ${media.height}`,
        // As large as fits, but never taller than the page leaves room for (the page sets --stage-h).
        width: `min(100%, calc(var(--stage-h, 68vh) * ${aspect}))`,
      }}
    >
      <video
        ref={videoRef}
        src={media.proxy_url}
        poster={media.thumbnail_url}
        muted
        playsInline
        preload="auto"
        className="absolute inset-0 h-full w-full object-contain"
        onError={props.onVideoError}
      />
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Video frame. Use the tools to mark the areas to remove."
        className={`absolute inset-0 h-full w-full ${tool === "select" || props.locked ? "cursor-default" : "cursor-crosshair"}`}
        style={{ touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={onPointerLeave}
        onDoubleClick={onDoubleClick}
      />
      {props.children}
    </div>
  );
}
