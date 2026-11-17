import { describe, expect, it } from "vitest";
import {
  buildRequest,
  describeResult,
  formatBytes,
  formatRemaining,
  formatSpeed,
  heightOptions,
  isActive,
  mergeJob,
  parseJobEvent,
  queueMessage,
  DEFAULT_SETTINGS,
} from "./render";
import type { EngineInfo, Job } from "../types/render";

function job(over: Partial<Job> = {}): Job {
  return {
    id: "j1",
    project_id: "p1",
    status: "running",
    stage: "reconstructing",
    stages: [],
    frames_done: 10,
    frames_total: 100,
    percent: 10,
    speed_fps: null,
    eta_seconds: null,
    jobs_ahead: null,
    error: null,
    warnings: [],
    engine: { provider: "dev", development_only: true },
    settings: {},
    result: null,
    created_at: "2026-11-03T00:00:00Z",
    started_at: null,
    completed_at: null,
    ...over,
  };
}

const engine = (over: Partial<EngineInfo> = {}): EngineInfo => ({
  provider: "api",
  name: "Remote",
  available: true,
  development_only: false,
  sends_frames_to_third_party: true,
  device: null,
  reason: null,
  expectation: "",
  ...over,
});

describe("isActive", () => {
  it("is true only while a job waits or runs", () => {
    expect(isActive(null)).toBe(false);
    expect(isActive(job({ status: "queued" }))).toBe(true);
    expect(isActive(job({ status: "running" }))).toBe(true);
    for (const status of ["completed", "failed", "cancelled"] as const) expect(isActive(job({ status }))).toBe(false);
  });
});

describe("heightOptions", () => {
  it("offers only sizes smaller than the original", () => {
    expect(heightOptions(3840, 2160)).toEqual([1440, 1080, 720]);
    expect(heightOptions(2560, 1440)).toEqual([1080, 720]);
    expect(heightOptions(1920, 1080)).toEqual([720]);
    expect(heightOptions(1280, 720)).toEqual([]);
    expect(heightOptions(null, null)).toEqual([]);
  });

  it("goes by the shorter side, so a portrait 1080x1920 clip is a 1080p clip", () => {
    expect(heightOptions(1080, 1920)).toEqual([720]);
    expect(heightOptions(2160, 3840)).toEqual([1440, 1080, 720]);
    expect(heightOptions(720, 1280)).toEqual([]);
  });
});

describe("buildRequest", () => {
  it("only acknowledges outside processing when the engine needs it and the person agreed", () => {
    expect(buildRequest(DEFAULT_SETTINGS, engine(), true).acknowledge_third_party).toBe(true);
    expect(buildRequest(DEFAULT_SETTINGS, engine(), false).acknowledge_third_party).toBe(false);
    expect(buildRequest(DEFAULT_SETTINGS, engine({ sends_frames_to_third_party: false }), true).acknowledge_third_party).toBe(false);
    expect(buildRequest(DEFAULT_SETTINGS, null, true).acknowledge_third_party).toBe(false);
  });
});

describe("wording", () => {
  it("rounds the time remaining to something honest", () => {
    expect(formatRemaining(null)).toBeNull();
    expect(formatRemaining(-1)).toBeNull();
    expect(formatRemaining(2)).toBe("a few seconds");
    expect(formatRemaining(23)).toBe("about 25 seconds");
    expect(formatRemaining(60)).toBe("about 1 minute");
    expect(formatRemaining(600)).toBe("about 10 minutes");
    expect(formatRemaining(5400)).toBe("about 1 h 30 min");
    expect(formatRemaining(7200)).toBe("about 2 h");
  });

  it("shows speed only when there is one", () => {
    expect(formatSpeed(null)).toBeNull();
    expect(formatSpeed(0)).toBeNull();
    expect(formatSpeed(3.456)).toBe("3.5 frames/s");
    expect(formatSpeed(24.4)).toBe("24 frames/s");
  });

  it("says how many are ahead only when that is known", () => {
    expect(queueMessage(job({ status: "running" }))).toBeNull();
    expect(queueMessage(job({ status: "queued", jobs_ahead: 0 }))).toBe("Waiting for a free worker…");
    expect(queueMessage(job({ status: "queued", jobs_ahead: 1 }))).toContain("1 video is ahead");
    expect(queueMessage(job({ status: "queued", jobs_ahead: 4 }))).toContain("4 videos are ahead");
  });

  it("formats sizes", () => {
    expect(formatBytes(100)).toBe("1 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.00 GB");
  });

  it("describes a result from its recorded facts", () => {
    const lines = describeResult({
      width: 1920,
      height: 1080,
      frames: 90,
      fps: 29.97,
      container: "mp4",
      size_bytes: 2 * 1024 * 1024,
      audio: "converted",
      audio_codec: "aac",
      tracking_used: true,
      render_seconds: 4,
    });
    expect(lines[0]).toBe("1920×1080, 90 frames at 29.97 fps");
    expect(lines[1]).toBe("MP4, 2.0 MB");
    expect(lines[2]).toContain("converted once");
    expect(lines[3]).toContain("tracked");
  });
});

describe("parseJobEvent", () => {
  it("reads a job and rejects everything else", () => {
    expect(parseJobEvent(JSON.stringify(job()))?.id).toBe("j1");
    expect(parseJobEvent("not json")).toBeNull();
    expect(parseJobEvent("{}")).toBeNull();
    expect(parseJobEvent("null")).toBeNull();
    expect(parseJobEvent(JSON.stringify({ id: "x", status: "running" }))).toBeNull();
  });
});

describe("mergeJob", () => {
  it("takes the first job and a different job as they come", () => {
    const first = job();
    expect(mergeJob(null, first)).toBe(first);
    const other = job({ id: "j2" });
    expect(mergeJob(first, other)).toBe(other);
  });

  it("never lets a late, older update undo progress or an ending", () => {
    const ahead = job({ frames_done: 50 });
    expect(mergeJob(ahead, job({ frames_done: 20 }))).toBe(ahead);
    const done = job({ status: "completed" });
    expect(mergeJob(done, job({ status: "running" }))).toBe(done);
  });

  it("accepts progress, stage changes and endings", () => {
    const now = job({ frames_done: 10 });
    const more = job({ frames_done: 11 });
    expect(mergeJob(now, more)).toBe(more);
    const audio = job({ stage: "audio", frames_done: 10 });
    expect(mergeJob(now, audio)).toBe(audio);
    const failed = job({ status: "failed" });
    expect(mergeJob(now, failed)).toBe(failed);
  });
});

import { effectiveSettings, groupStages } from "./render";
import type { Stage } from "../types/render";

const stage = (key: string, state: Stage["state"], detail: string | null = null): Stage => ({ key, label: key, state, detail });
const all = (states: Partial<Record<string, Stage["state"]>>): Stage[] =>
  ["analyze", "mask", "track", "preparing", "measuring", "reconstructing", "audio", "verifying", "saving"].map((k) => stage(k, states[k] ?? "pending"));

describe("processing mode", () => {
  const base = { quality: "high", container: "mp4", output_height: null, keep_audio: true } as const;
  it("sends fast quality for Fast processing, whatever quality was chosen", () => {
    expect(effectiveSettings(base, "fast").quality).toBe("fast");
  });
  it("sends the chosen quality for AI reconstruction", () => {
    expect(effectiveSettings(base, "ai").quality).toBe("high");
    expect(effectiveSettings({ ...base, quality: "balanced" }, "ai").quality).toBe("balanced");
  });
});

describe("groupStages", () => {
  it("shows four groups in the order the work happens", () => {
    expect(groupStages(all({})).map((g) => g.label)).toEqual(["Preparing video", "Tracking selected region", "Reconstructing video", "Finalizing export"]);
  });

  it("is active while any of its steps is active, and done once every step is", () => {
    const groups = groupStages(all({ analyze: "done", mask: "done", track: "skipped", preparing: "done", measuring: "active" }));
    expect(groups[0].state).toBe("active");
    expect(groups[0].detail).toContain("measuring");
    expect(groups[1].state).toBe("skipped");
    expect(groups[2].state).toBe("pending");
  });

  it("marks a group done only when everything in it finished", () => {
    const groups = groupStages(all({ analyze: "done", mask: "done", track: "done", preparing: "done", measuring: "done", reconstructing: "active" }));
    expect(groups.map((g) => g.state)).toEqual(["done", "done", "active", "pending"]);
  });

  it("carries a failure to its group, and the frame count to the reconstruction detail", () => {
    const groups = groupStages([...all({ reconstructing: "failed" }).filter((s) => s.key !== "reconstructing"), stage("reconstructing", "failed", "12 of 96 frames")]);
    expect(groups[2].state).toBe("failed");
    expect(groups[2].detail).toContain("12 of 96 frames");
  });

  it("is all done after a completed job", () => {
    const done = all({ analyze: "done", mask: "done", track: "skipped", preparing: "done", measuring: "skipped", reconstructing: "done", audio: "skipped", verifying: "done", saving: "done" });
    expect(groupStages(done).map((g) => g.state)).toEqual(["done", "skipped", "done", "done"]);
  });
});
