import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { clampKeyframe, editShapeAt, hitRegion, pickRegion } from "./masks";
import { lostAreas, resolverFor, trackedShapeAt, trackingFraction, transformShape } from "./tracking";
import type { Keyframe, MaskDocument, Region } from "../types/mask";
import type { TrackingData, TrackSegment } from "../types/tracking";

interface GoldenCase {
  name: string;
  aspect: number;
  frame: number;
  region: Region;
  segments: TrackSegment[];
  expected: { rect?: number[]; points?: number[][]; strokes?: { radius: number; points: number[][] }[]; erase?: { radius: number; points: number[][] }[] };
}

const golden = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../backend/tests/fixtures/shape_cases.json", import.meta.url)), "utf8"),
) as { cases: GoldenCase[] };

function close(got: unknown, want: unknown, label: string): void {
  if (Array.isArray(want)) {
    expect(Array.isArray(got), label).toBe(true);
    expect((got as unknown[]).length, label).toBe(want.length);
    want.forEach((w, i) => close((got as unknown[])[i], w, `${label}[${i}]`));
  } else if (typeof want === "object" && want !== null) {
    const entries = Object.entries(want);
    expect(Object.keys(got as object).sort(), label).toEqual(entries.map(([k]) => k).sort());
    for (const [k, w] of entries) close((got as Record<string, unknown>)[k], w, `${label}.${k}`);
  } else {
    expect(got as number, label).toBeCloseTo(want as number, 9);
  }
}

describe("matches the Python implementation", () => {
  it("covers a useful number of cases", () => expect(golden.cases.length).toBeGreaterThanOrEqual(6));

  for (const c of golden.cases) {
    it(c.name, () => {
      const shape = trackedShapeAt(c.region, c.frame, c.segments, c.aspect);
      // The golden file leaves out empty parts; so does this comparison.
      const got: Record<string, unknown> = {};
      if (shape.rect) got.rect = [shape.rect.x, shape.rect.y, shape.rect.w, shape.rect.h];
      if (shape.points) got.points = shape.points;
      if (shape.strokes) got.strokes = shape.strokes;
      if (shape.erase && shape.erase.length > 0) got.erase = shape.erase;
      close(got, c.expected, c.name);
    });
  }
});

const rectRegion = (): Region => ({
  id: "r",
  name: "Logo",
  type: "rectangle",
  keyframes: [{ frame: 0, rect: { x: 0.4, y: 0.4, w: 0.2, h: 0.1 } }],
});

const dataFor = (segments: TrackSegment[], regionId = "r"): TrackingData => ({
  version: 1, width: 320, height: 568, aspect: 568 / 320, frame_count: 60, regions: { [regionId]: { segments } },
});

describe("transformShape", () => {
  it("leaves a shape alone for the identity transform", () => {
    const shape: Keyframe = { frame: 0, rect: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 } };
    expect(transformShape(shape, [0, 0, 1, 0], 2)).toBe(shape);
  });

  it("measures distances in frame widths, so a tall frame moves less vertically", () => {
    const moved = transformShape({ frame: 0, rect: { x: 0.2, y: 0.2, w: 0.2, h: 0.2 } }, [0.1, 0.1, 1, 0], 2);
    expect(moved.rect?.x).toBeCloseTo(0.3);
    expect(moved.rect?.y).toBeCloseTo(0.25);
  });

  it("clips a rectangle that drifts off the edge and never returns a negative size", () => {
    const clipped = transformShape({ frame: 0, rect: { x: 0.8, y: 0.4, w: 0.2, h: 0.2 } }, [0.5, 0, 1, 0], 1);
    expect(clipped.rect?.w).toBe(0);
    expect(clipped.rect?.x).toBeGreaterThanOrEqual(0.9);
  });
});

describe("trackedShapeAt", () => {
  const segment: TrackSegment = { anchor: 0, step: 1, until: 59, transforms: [[0.1, 0, 1, 0], [0.2, 0, 1, 0]], lost_at: 3, reason: "gone" };

  it("follows the tracked motion, then holds the last position after tracking is lost", () => {
    const region = rectRegion();
    expect(trackedShapeAt(region, 1, [segment], 1).rect?.x).toBeCloseTo(0.5);
    expect(trackedShapeAt(region, 2, [segment], 1).rect?.x).toBeCloseTo(0.6);
    expect(trackedShapeAt(region, 40, [segment], 1).rect?.x).toBeCloseTo(0.6);
  });

  it("uses the keyframe itself on its own frame", () => {
    const region = rectRegion();
    region.keyframes.push({ frame: 10, rect: { x: 0.0, y: 0.0, w: 0.1, h: 0.1 } });
    const limited = { ...segment, until: 9 };
    expect(trackedShapeAt(region, 10, [limited], 1).rect?.x).toBe(0);
    expect(trackedShapeAt(region, 9, [limited], 1).rect?.x).toBeCloseTo(0.6);
  });

  it("falls back to blending keyframes where nothing was tracked", () => {
    const region = rectRegion();
    region.keyframes.push({ frame: 10, rect: { x: 0.6, y: 0.4, w: 0.2, h: 0.1 } });
    expect(trackedShapeAt(region, 5, undefined, 1).rect?.x).toBeCloseTo(0.5);
  });
});

describe("resolverFor", () => {
  it("is plain keyframing without tracking data", () => {
    expect(resolverFor(null)(rectRegion(), 30).rect?.x).toBe(0.4);
  });

  it("ignores regions it has no results for", () => {
    const resolve = resolverFor(dataFor([{ anchor: 0, step: 1, until: 9, transforms: [[0.3, 0, 1, 0]] }], "someone-else"));
    expect(resolve(rectRegion(), 1).rect?.x).toBe(0.4);
  });
});

describe("editing a tracked frame", () => {
  const doc = (): MaskDocument => ({ version: 1, regions: [rectRegion()] });
  const resolve = resolverFor(dataFor([{ anchor: 0, step: 1, until: 59, transforms: Array.from({ length: 59 }, (_, i): [number, number, number, number] => [0.01 * (i + 1), 0, 1, 0]) }]));

  it("starts from where tracking put the region, not from the keyframe", () => {
    const next = editShapeAt(doc(), "r", 10, (k) => ({ ...k, rect: { ...k.rect!, y: 0.7 } }), resolve);
    const added = next.regions[0].keyframes.find((k) => k.frame === 10)!;
    expect(added.rect?.x).toBeCloseTo(0.5); // 0.4 + 10 * 0.01
    expect(added.rect?.y).toBe(0.7);
  });

  it("keeps a keyframe inside the frame even when the tracked shape had drifted past the edge", () => {
    const far = resolverFor(dataFor([{ anchor: 0, step: 1, until: 59, transforms: [[0.4, 0, 1, 0]] }]));
    const next = editShapeAt(doc(), "r", 1, (k) => k, far);
    const rect = next.regions[0].keyframes.find((k) => k.frame === 1)!.rect!;
    expect(rect.x + rect.w).toBeLessThanOrEqual(1 + 1e-12);
    expect(rect.x).toBeGreaterThanOrEqual(0);
  });

  it("hit-tests where the region is on that frame", () => {
    const region = rectRegion();
    expect(hitRegion(region, 10, [0.55, 0.45], 1, resolve)).toBe(true); // moved right by 0.1
    expect(hitRegion(region, 10, [0.41, 0.45], 1, resolve)).toBe(false); // where it used to be
    expect(pickRegion(doc(), 10, [0.55, 0.45], 1, resolve)?.id).toBe("r");
    expect(pickRegion(doc(), 10, [0.41, 0.45], 1, resolve)).toBeNull();
  });
});

describe("clampKeyframe", () => {
  it("pulls every shape kind back into the saved format's limits", () => {
    expect(clampKeyframe({ frame: 0, rect: { x: 0.95, y: -0.1, w: 0.2, h: 0.3 } }).rect).toEqual({ x: 0.8, y: 0, w: 0.2, h: 0.3 });
    expect(clampKeyframe({ frame: 0, points: [[-0.2, 0.5], [1.4, 0.5], [0.5, 2]] }).points).toEqual([[0, 0.5], [1, 0.5], [0.5, 1]]);
    const brush = clampKeyframe({ frame: 0, strokes: [{ radius: 0.9, points: [[1.2, 0.5]] }], erase: [{ radius: 0, points: [[0.5, -1]] }] });
    expect(brush.strokes?.[0]).toEqual({ radius: 0.25, points: [[1, 0.5]] });
    expect(brush.erase?.[0]).toEqual({ radius: 0.001, points: [[0.5, 0]] });
  });

  it("leaves a valid shape unchanged", () => {
    const shape: Keyframe = { frame: 3, rect: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 } };
    expect(clampKeyframe(shape)).toEqual(shape);
  });
});

describe("lostAreas", () => {
  const doc: MaskDocument = {
    version: 1,
    regions: [
      { id: "a", name: "Alpha", type: "rectangle", keyframes: [{ frame: 0, rect: { x: 0, y: 0, w: 0.1, h: 0.1 } }] },
      { id: "b", name: "Beta", type: "rectangle", keyframes: [{ frame: 0, rect: { x: 0, y: 0, w: 0.1, h: 0.1 } }] },
    ],
  };

  it("lists only segments that gave up, earliest first, for regions that still exist", () => {
    const data = dataFor([{ anchor: 0, step: 1, until: 59, transforms: [], lost_at: 40, reason: "A went away." }], "a");
    data.regions.b = { segments: [{ anchor: 0, step: 1, until: 59, transforms: [], lost_at: 12, reason: "B left." }, { anchor: 30, step: 1, until: 59, transforms: [] }] };
    data.regions.deleted = { segments: [{ anchor: 0, step: 1, until: 59, transforms: [], lost_at: 5, reason: "x" }] };
    expect(lostAreas(data, doc).map((a) => [a.regionName, a.frame, a.reason])).toEqual([
      ["Beta", 12, "B left."],
      ["Alpha", 40, "A went away."],
    ]);
  });

  it("is empty with no data", () => expect(lostAreas(null, doc)).toEqual([]));
});

describe("trackingFraction", () => {
  it("is unknown until the job reports a size, so no percentage is invented", () => {
    expect(trackingFraction(0, 0)).toBeNull();
    expect(trackingFraction(30, 60)).toBe(0.5);
    expect(trackingFraction(70, 60)).toBe(1);
  });
});
