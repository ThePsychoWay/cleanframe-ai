import { describe, expect, it } from "vitest";
import {
  EMPTY_DOCUMENT,
  MAX_REGIONS,
  MAX_TOTAL_POINTS,
  MIN_RECT_SIZE,
  addRegion,
  createRegion,
  editShapeAt,
  frameToTime,
  hasKeyframeAt,
  hitHandle,
  hitRegion,
  pickRegion,
  pointInPolygon,
  problemWithDocument,
  rectFromCorners,
  removeKeyframe,
  removeRegion,
  renameRegion,
  resizeRect,
  setKeyframe,
  shapeAt,
  shapeBounds,
  simplifyStroke,
  timeToFrame,
  translateBrush,
  translatePoints,
  translateRect,
} from "./masks";
import { canRedo, canUndo, commit, createHistory, redo, undo, HISTORY_LIMIT } from "./history";
import type { MaskDocument, Point, Region } from "../types/mask";

const rectRegion = (id = "region_1", ...frames: Array<[number, number]>): Region => ({
  id,
  name: "Logo",
  type: "rectangle",
  keyframes: frames.map(([frame, x]) => ({ frame, rect: { x, y: 0.2, w: 0.2, h: 0.1 } })),
});

const polygonRegion = (): Region => ({
  id: "region_2",
  name: "Text",
  type: "polygon",
  keyframes: [
    { frame: 0, points: [[0, 0], [0.4, 0], [0.4, 0.4]] },
    { frame: 10, points: [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6]] },
  ],
});

const docOf = (...regions: Region[]): MaskDocument => ({ version: 1, regions });

describe("frames and time", () => {
  it("seeks to the middle of a frame and reads the same frame back", () => {
    for (const fps of [24, 29.97, 30, 59.94, 120]) {
      for (const frame of [0, 1, 17, 239]) {
        expect(timeToFrame(frameToTime(frame, fps), fps, 1000)).toBe(frame);
      }
    }
  });

  it("never reports a frame outside the video", () => {
    expect(timeToFrame(-1, 30, 60)).toBe(0);
    expect(timeToFrame(999, 30, 60)).toBe(59);
    expect(timeToFrame(5, 30, 0)).toBe(0);
  });
});

describe("shapeAt", () => {
  it("applies a single keyframe to the whole video", () => {
    const region = rectRegion("r", [30, 0.5]);
    for (const frame of [0, 29, 30, 31, 500]) expect(shapeAt(region, frame).rect?.x).toBe(0.5);
  });

  it("holds the nearest keyframe outside the keyframed range", () => {
    const region = rectRegion("r", [10, 0.1], [20, 0.5]);
    expect(shapeAt(region, 0).rect?.x).toBe(0.1);
    expect(shapeAt(region, 99).rect?.x).toBe(0.5);
  });

  it("blends rectangles linearly between keyframes", () => {
    const region = rectRegion("r", [0, 0.0], [10, 0.5]);
    expect(shapeAt(region, 0).rect?.x).toBe(0);
    expect(shapeAt(region, 5).rect?.x).toBeCloseTo(0.25);
    expect(shapeAt(region, 10).rect?.x).toBe(0.5);
    expect(shapeAt(region, 5).frame).toBe(5);
  });

  it("blends polygons with matching point counts", () => {
    const middle = shapeAt(polygonRegion(), 5).points as Point[];
    expect(middle[0][0]).toBeCloseTo(0.1);
    expect(middle[2][0]).toBeCloseTo(0.5);
    expect(middle[2][1]).toBeCloseTo(0.5);
  });

  it("holds the earlier keyframe when polygons have different point counts", () => {
    const region = polygonRegion();
    region.keyframes[1] = { frame: 10, points: [[0, 0], [1, 0], [1, 1], [0, 1]] };
    expect(shapeAt(region, 5).points).toHaveLength(3);
    expect(shapeAt(region, 10).points).toHaveLength(4);
  });

  it("holds brush strokes until the next keyframe rather than inventing in-between paint", () => {
    const stroke = { radius: 0.02, points: [[0.1, 0.1]] as Point[] };
    const region: Region = {
      id: "b",
      name: "Brush",
      type: "brush",
      keyframes: [
        { frame: 0, strokes: [stroke] },
        { frame: 10, strokes: [{ ...stroke, points: [[0.9, 0.9]] }] },
      ],
    };
    expect(shapeAt(region, 7).strokes?.[0].points[0]).toEqual([0.1, 0.1]);
    expect(shapeAt(region, 10).strokes?.[0].points[0]).toEqual([0.9, 0.9]);
  });
});

describe("keyframe editing", () => {
  it("keeps keyframes in order and replaces one on the same frame", () => {
    let doc = docOf(rectRegion("r", [10, 0.1], [30, 0.3]));
    doc = setKeyframe(doc, "r", { frame: 20, rect: { x: 0.2, y: 0.2, w: 0.1, h: 0.1 } });
    expect(doc.regions[0].keyframes.map((k) => k.frame)).toEqual([10, 20, 30]);
    doc = setKeyframe(doc, "r", { frame: 20, rect: { x: 0.9, y: 0.2, w: 0.1, h: 0.1 } });
    expect(doc.regions[0].keyframes).toHaveLength(3);
    expect(doc.regions[0].keyframes[1].rect?.x).toBe(0.9);
  });

  it("never mutates the document it was given", () => {
    const original = docOf(rectRegion("r", [0, 0.1]));
    const snapshot = JSON.stringify(original);
    setKeyframe(original, "r", { frame: 5, rect: { x: 0.5, y: 0.5, w: 0.1, h: 0.1 } });
    editShapeAt(original, "r", 3, (k) => ({ ...k, rect: { x: 0.7, y: 0.7, w: 0.1, h: 0.1 } }));
    removeRegion(original, "r");
    renameRegion(original, "r", "New name");
    expect(JSON.stringify(original)).toBe(snapshot);
  });

  it("starts an edit from the blended shape and stores it on the current frame", () => {
    const doc = docOf(rectRegion("r", [0, 0.0], [10, 0.5]));
    const edited = editShapeAt(doc, "r", 5, (k) => ({ ...k, rect: { ...k.rect!, y: 0.8 } }));
    const keyframe = edited.regions[0].keyframes.find((k) => k.frame === 5);
    expect(keyframe?.rect?.x).toBeCloseTo(0.25); // came from the blend, not from either neighbour
    expect(keyframe?.rect?.y).toBe(0.8);
    expect(hasKeyframeAt(edited.regions[0], 5)).toBe(true);
  });

  it("refuses to delete a region's last keyframe", () => {
    const doc = docOf(rectRegion("r", [0, 0.1]));
    expect(removeKeyframe(doc, "r", 0)).toBe(doc); // unchanged, so it's the very same object
    const two = docOf(rectRegion("r", [0, 0.1], [5, 0.2]));
    expect(removeKeyframe(two, "r", 0).regions[0].keyframes.map((k) => k.frame)).toEqual([5]);
    expect(removeKeyframe(two, "r", 3)).toBe(two); // no keyframe there
    expect(removeKeyframe(two, "missing", 0)).toBe(two);
  });

  it("numbers new regions past the highest existing id, even after deletions", () => {
    let doc: MaskDocument = EMPTY_DOCUMENT;
    const k = { frame: 0, rect: { x: 0, y: 0, w: 0.1, h: 0.1 } };
    doc = addRegion(doc, createRegion(doc, "rectangle", k));
    doc = addRegion(doc, createRegion(doc, "rectangle", k));
    doc = removeRegion(doc, "region_1");
    const next = createRegion(doc, "rectangle", k);
    expect(next.id).toBe("region_3");
    expect(next.name).toBe("Region 3");
  });
});

describe("rectangles", () => {
  it("builds a rectangle from any two corners and rejects accidental clicks", () => {
    expect(rectFromCorners([0.5, 0.6], [0.2, 0.3])).toEqual({ x: 0.2, y: 0.3, w: expect.closeTo(0.3), h: expect.closeTo(0.3) });
    expect(rectFromCorners([0.5, 0.5], [0.5 + MIN_RECT_SIZE / 2, 0.8])).toBeNull();
  });

  it("keeps a dragged-out rectangle inside the frame", () => {
    const rect = rectFromCorners([0.8, 0.8], [1.4, 1.4]);
    expect(rect && rect.x + rect.w).toBeLessThanOrEqual(1);
  });

  it("resizes from a corner and from an edge", () => {
    const rect = { x: 0.2, y: 0.2, w: 0.2, h: 0.2 };
    expect(resizeRect(rect, "se", [0.6, 0.7])).toEqual({ x: 0.2, y: 0.2, w: expect.closeTo(0.4), h: expect.closeTo(0.5) });
    expect(resizeRect(rect, "w", [0.1, 0.9])).toEqual({ x: expect.closeTo(0.1), y: 0.2, w: expect.closeTo(0.3), h: 0.2 });
  });

  it("flips instead of inverting when a handle is dragged past the opposite edge", () => {
    const flipped = resizeRect({ x: 0.2, y: 0.2, w: 0.2, h: 0.2 }, "e", [0.05, 0.3]);
    expect(flipped.x).toBeCloseTo(0.05);
    expect(flipped.w).toBeCloseTo(0.15);
    expect(flipped.w).toBeGreaterThan(0);
  });

  it("never shrinks below the minimum or leaves the frame", () => {
    const tiny = resizeRect({ x: 0.2, y: 0.2, w: 0.2, h: 0.2 }, "se", [0.2, 0.2]);
    expect(tiny.w).toBeGreaterThanOrEqual(MIN_RECT_SIZE);
    const edge = resizeRect({ x: 0.2, y: 0.2, w: 0.2, h: 0.2 }, "se", [3, 3]);
    expect(edge.x + edge.w).toBeLessThanOrEqual(1 + 1e-9);
  });

  it("finds the nearest handle within tolerance", () => {
    const rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.2 };
    expect(hitHandle(rect, [0.201, 0.2], 0.02, 1)).toBe("nw");
    expect(hitHandle(rect, [0.4, 0.2], 0.02, 1)).toBe("n");
    expect(hitHandle(rect, [0.4, 0.3], 0.02, 1)).toBeNull();
  });

  it("moves without leaving the frame", () => {
    const moved = translateRect({ x: 0.5, y: 0.5, w: 0.4, h: 0.4 }, 0.5, -0.9);
    expect(moved).toEqual({ x: expect.closeTo(0.6), y: 0, w: 0.4, h: 0.4 });
  });
});

describe("polygons", () => {
  it("translates as one piece and stops at the frame edge", () => {
    const shape: Point[] = [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]];
    const moved = translatePoints(shape, 2, 0);
    expect(Math.max(...moved.map((p) => p[0]))).toBeCloseTo(1);
    expect(moved[1][0] - moved[0][0]).toBeCloseTo(0.4); // shape unchanged
  });

  it("tells inside from outside, including concave shapes", () => {
    const l: Point[] = [[0, 0], [0.5, 0], [0.5, 0.5], [1, 0.5], [1, 1], [0, 1]];
    expect(pointInPolygon([0.25, 0.25], l)).toBe(true);
    expect(pointInPolygon([0.75, 0.25], l)).toBe(false); // the notch
    expect(pointInPolygon([0.75, 0.75], l)).toBe(true);
  });
});

describe("brush shapes", () => {
  const painted = {
    frame: 0,
    strokes: [{ radius: 0.05, points: [[0.2, 0.3], [0.4, 0.3]] as Point[] }],
    erase: [{ radius: 0.02, points: [[0.3, 0.3]] as Point[] }],
  };

  it("moves paint and erasing together, and stops at the frame edge", () => {
    const moved = translateBrush(painted, 0.1, 0.1);
    expect(moved.strokes?.[0].points[0][0]).toBeCloseTo(0.3);
    expect(moved.erase?.[0].points[0][0]).toBeCloseTo(0.4);

    const pushed = translateBrush(painted, 5, 0);
    expect(pushed.strokes?.[0].points[1][0]).toBeCloseTo(1); // the right-most point reached the edge
    expect(pushed.erase?.[0].points[0][0]).toBeCloseTo(0.9); // and the erase stroke kept its offset
  });

  it("doesn't add an erase list where there was none", () => {
    expect(translateBrush({ frame: 0, strokes: painted.strokes }, 0.1, 0)).not.toHaveProperty("erase");
  });

  it("bounds a brush by its strokes plus the brush radius", () => {
    const bounds = shapeBounds(painted, 1)!;
    expect(bounds.x).toBeCloseTo(0.15);
    expect(bounds.w).toBeCloseTo(0.3);
    // On a tall frame the same radius reaches further in y (as a fraction of the height).
    expect(shapeBounds(painted, 0.5)!.h).toBeGreaterThan(shapeBounds(painted, 1)!.h);
  });

  it("bounds rectangles and polygons, and reports nothing for an empty shape", () => {
    expect(shapeBounds({ frame: 0, rect: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 } }, 1)).toEqual({ x: 0.1, y: 0.2, w: 0.3, h: 0.4 });
    expect(shapeBounds({ frame: 0, points: [[0.1, 0.2], [0.5, 0.2], [0.3, 0.9]] }, 1)).toEqual({
      x: 0.1,
      y: 0.2,
      w: expect.closeTo(0.4),
      h: expect.closeTo(0.7),
    });
    expect(shapeBounds({ frame: 0 }, 1)).toBeNull();
  });
});

describe("hit testing", () => {
  const brush: Region = {
    id: "b",
    name: "Brush",
    type: "brush",
    keyframes: [{ frame: 0, strokes: [{ radius: 0.05, points: [[0.2, 0.5], [0.6, 0.5]] }] }],
  };

  it("measures brush radius in frame widths, so the brush stays round on a tall frame", () => {
    // On a 9:16 frame (aspect 0.5625) a step of 0.04 of the height covers 0.04 / 0.5625 = 0.071 of the
    // width in pixels: further than the 0.05 radius. On a square frame the same step is only 0.04.
    expect(hitRegion(brush, 0, [0.4, 0.5 + 0.04], 0.5625)).toBe(false);
    expect(hitRegion(brush, 0, [0.4, 0.5 + 0.04], 1)).toBe(true);
    // Half that step (0.02 of the height = 0.036 of the width) is inside even on the tall frame.
    expect(hitRegion(brush, 0, [0.4, 0.5 + 0.02], 0.5625)).toBe(true);
  });

  it("treats a single-point stroke as a dot", () => {
    const dot: Region = { ...brush, keyframes: [{ frame: 0, strokes: [{ radius: 0.05, points: [[0.5, 0.5]] }] }] };
    expect(hitRegion(dot, 0, [0.52, 0.5], 1)).toBe(true);
    expect(hitRegion(dot, 0, [0.6, 0.5], 1)).toBe(false);
  });

  it("does not hit where an erase stroke has removed paint", () => {
    const erased: Region = {
      ...brush,
      keyframes: [{ ...brush.keyframes[0], erase: [{ radius: 0.03, points: [[0.4, 0.5]] }] }],
    };
    expect(hitRegion(erased, 0, [0.3, 0.5], 1)).toBe(true);
    expect(hitRegion(erased, 0, [0.4, 0.5], 1)).toBe(false);
  });

  it("picks the top-most region and follows the keyframes through time", () => {
    const doc = docOf(rectRegion("under", [0, 0.1]), { ...rectRegion("over", [0, 0.1], [20, 0.7]), name: "Over" });
    expect(pickRegion(doc, 0, [0.15, 0.25], 1)?.id).toBe("over");
    expect(pickRegion(doc, 20, [0.15, 0.25], 1)?.id).toBe("under"); // "over" has moved away by frame 20
    expect(pickRegion(doc, 0, [0.9, 0.9], 1)).toBeNull();
  });
});

describe("simplifyStroke", () => {
  it("drops near-duplicate points but keeps the ends", () => {
    const dense: Point[] = Array.from({ length: 100 }, (_, i) => [i * 0.0005, 0.5]);
    const simple = simplifyStroke(dense, 0.004);
    expect(simple.length).toBeLessThan(dense.length / 4);
    expect(simple[0]).toEqual(dense[0]);
    expect(simple[simple.length - 1]).toEqual(dense[99]);
  });

  it("leaves tiny strokes alone", () => {
    const dot: Point[] = [[0.5, 0.5]];
    expect(simplifyStroke(dot)).toBe(dot);
  });
});

describe("problemWithDocument", () => {
  const ok = docOf(rectRegion("r", [0, 0.1]));

  it("accepts a normal document", () => {
    expect(problemWithDocument(ok, 60)).toBeNull();
    expect(problemWithDocument(EMPTY_DOCUMENT, 60)).toBeNull();
  });

  it("flags a keyframe past the end of the video", () => {
    expect(problemWithDocument(docOf(rectRegion("r", [60, 0.1])), 60)).toMatch(/past the end/);
    expect(problemWithDocument(docOf(rectRegion("r", [59, 0.1])), 60)).toBeNull();
  });

  it("flags too many regions and too many points", () => {
    const many = docOf(...Array.from({ length: MAX_REGIONS + 1 }, (_, i) => rectRegion(`r${i}`, [0, 0.1])));
    expect(problemWithDocument(many, 60)).toMatch(/at most 20/);

    const heavy: Region = {
      id: "h",
      name: "Heavy",
      type: "brush",
      keyframes: Array.from({ length: Math.ceil(MAX_TOTAL_POINTS / 2000) + 1 }, (_, frame) => ({
        frame,
        strokes: [{ radius: 0.01, points: Array.from({ length: 2000 }, () => [0.5, 0.5] as Point) }],
      })),
    };
    expect(problemWithDocument(docOf(heavy), 100)).toMatch(/too detailed/);
  });
});

describe("history", () => {
  it("undoes and redoes, and a new change discards the redo branch", () => {
    let h = createHistory("a");
    h = commit(h, "b");
    h = commit(h, "c");
    expect(h.present).toBe("c");

    h = undo(h);
    expect([h.present, canUndo(h), canRedo(h)]).toEqual(["b", true, true]);
    h = redo(h);
    expect(h.present).toBe("c");

    h = undo(undo(h));
    expect(h.present).toBe("a");
    expect(undo(h)).toBe(h); // nothing further back
    h = commit(h, "x");
    expect(canRedo(h)).toBe(false);
    expect(redo(h)).toBe(h);
  });

  it("does not record a change that produced the same value", () => {
    const h = createHistory({ n: 1 });
    expect(commit(h, h.present)).toBe(h);
  });

  it("is bounded", () => {
    let h = createHistory(0);
    for (let i = 1; i <= HISTORY_LIMIT + 50; i += 1) h = commit(h, i);
    expect(h.past).toHaveLength(HISTORY_LIMIT);
  });
});

describe("hidden keyframes (an overlay that comes and goes)", () => {
  const region = {
    id: "r", name: "Logo 1", type: "rectangle" as const,
    keyframes: [
      { frame: 0, rect: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, hidden: true },
      { frame: 10, rect: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 } },
      { frame: 20, rect: { x: 0.5, y: 0.5, w: 0.2, h: 0.2 } },
      { frame: 30, rect: { x: 0.5, y: 0.5, w: 0.2, h: 0.2 }, hidden: true },
    ],
  };
  it("is not on screen before it appears and after it goes", () => {
    expect(shapeAt(region, 5).hidden).toBe(true);
    expect(shapeAt(region, 35).hidden).toBe(true);
  });
  it("blends between visible keyframes, but never towards a hidden one", () => {
    expect(shapeAt(region, 15).rect?.x).toBeCloseTo(0.3);
    expect(shapeAt(region, 25).rect?.x).toBeCloseTo(0.5);
    expect(shapeAt(region, 25).hidden).toBeFalsy();
  });
  it("can't be picked with the pointer where it isn't on screen", () => {
    expect(hitRegion(region, 5, [0.2, 0.2], 1)).toBe(false);
    expect(hitRegion(region, 10, [0.2, 0.2], 1)).toBe(true);
  });
});
