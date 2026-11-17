import { describe, expect, it } from "vitest";
import { describeOverlay, left, outcomeOf, removed } from "./automation";
import type { Automation, DetectedOverlay, Project } from "@/types/project";

const project = (over: Partial<Project> = {}): Project => ({
  id: "p", name: "n", original_filename: "n.mp4", size_bytes: 1, status: "ANALYZING", error: null, container: "mp4", codec: "h264",
  audio_codec: null, width: 1280, height: 720, fps: 30, duration: 5, bitrate: null, frame_count: 150, created_at: "", updated_at: "", expires_at: "",
  ...over,
});

const overlay = (over: Partial<DetectedOverlay> = {}): DetectedOverlay => ({
  id: 1, kind: "logo", decision: "remove", reason: "", basis: "motion", confidence: 0.95, first_frame: 0, last_frame: 149, frames_present: 150, notes: [], ...over,
});

describe("outcomeOf", () => {
  it("is working while the server analyses or renders", () => {
    expect(outcomeOf(project({ status: "ANALYZING" }))).toEqual({ kind: "working", step: "analysing" });
    expect(outcomeOf(project({ status: "PROCESSING" }))).toEqual({ kind: "working", step: "restoring" });
  });
  it("is done only when the server says COMPLETED", () => {
    expect(outcomeOf(project({ status: "COMPLETED" }))).toEqual({ kind: "done" });
  });
  it("never calls a video processed when nothing was found", () => {
    const automation: Automation = { state: "nothing_found", found: 0, removing: 0, overlays: [], notes: [] };
    const outcome = outcomeOf(project({ status: "MASKING", automation }));
    expect(outcome.kind).toBe("needs_help");
    if (outcome.kind === "needs_help") expect(outcome.message).toMatch(/Nothing was removed/);
  });
  it("says why when the engine isn't ready", () => {
    const automation: Automation = { state: "engine_unavailable", state_reason: "The reconstruction engine isn't ready: not set up", found: 1, removing: 1, overlays: [], notes: [] };
    const outcome = outcomeOf(project({ status: "MASKING", automation }));
    expect(outcome).toMatchObject({ kind: "needs_help", message: "The reconstruction engine isn't ready: not set up" });
  });
  it("shows the server's own error for a failed project", () => {
    expect(outcomeOf(project({ status: "FAILED", error: "bad file" }))).toEqual({ kind: "failed", message: "bad file" });
  });
});

describe("describeOverlay", () => {
  it("says an overlay on every frame is on every frame", () => {
    expect(describeOverlay(overlay(), 150)).toBe("Logo or mark, on every frame");
  });
  it("gives the real frame range for one that comes and goes, and says when it moves", () => {
    expect(describeOverlay(overlay({ kind: "text", first_frame: 30, last_frame: 89, notes: ["moves between positions"] }), 150)).toBe(
      "Text, frames 30 to 89, moves between positions",
    );
  });
});

describe("removed / left", () => {
  const automation: Automation = { found: 2, removing: 1, notes: [], overlays: [overlay(), overlay({ id: 2, decision: "left", reason: "caption" })] };
  it("splits what was taken from what was spared", () => {
    expect(removed(automation).map((o) => o.id)).toEqual([1]);
    expect(left(automation).map((o) => o.id)).toEqual([2]);
    expect(removed(null)).toEqual([]);
  });
});
