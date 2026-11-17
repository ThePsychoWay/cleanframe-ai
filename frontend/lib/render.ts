/**
 * Pure helpers for the render panel: what to offer, how to word progress, and how a stream of job updates is read.
 * Kept free of React and the network so they can be tested directly.
 */

import { OUTPUT_HEIGHTS, type OutputHeight } from "./resolution";
import type { EngineInfo, Job, RenderRequest, RenderResult, RenderSettings, Stage } from "../types/render";

export const DEFAULT_SETTINGS: RenderSettings = { quality: "balanced", container: "mp4", output_height: null, keep_audio: true };

export const QUALITY_LABELS: Record<RenderSettings["quality"], string> = {
  fast: "Fast",
  balanced: "Balanced",
  high: "High quality",
};

export const QUALITY_HINTS: Record<RenderSettings["quality"], string> = {
  fast: "Quickest, larger visible compression.",
  balanced: "A good default.",
  high: "Slowest, closest to the original.",
};

/** What the two quality choices in the editor do, in plain words (the numbers are the server's: 10 and 16 frames). */
export const QUALITY_EXPLANATION: Record<"balanced" | "high", string> = {
  balanced: "Studies about 10 neighbouring frames per frame. A good default.",
  high: "Studies about 16 neighbouring frames and encodes at higher quality. Slower.",
};

export type ProcessingMode = "ai" | "fast";

export const MODE_EXPLANATION: Record<ProcessingMode, string> = {
  ai: "Rebuilds what was behind the mark, using the quality above.",
  fast: "Studies about 6 frames and encodes faster. Leftover traces are easier to see.",
};

/** The settings actually sent: Fast processing is the server's "fast" quality, whatever quality was picked above. */
export function effectiveSettings(settings: RenderSettings, mode: ProcessingMode): RenderSettings {
  return mode === "fast" ? { ...settings, quality: "fast" } : { ...settings, quality: settings.quality === "fast" ? "balanced" : settings.quality };
}

export type GroupState = Stage["state"];

export interface StageGroup {
  key: "prepare" | "track" | "reconstruct" | "finalize";
  label: string;
  state: GroupState;
  detail: string | null;
}

const GROUPS: Array<{ key: StageGroup["key"]; label: string; members: string[] }> = [
  { key: "prepare", label: "Preparing video", members: ["analyze", "mask", "preparing", "measuring"] },
  { key: "track", label: "Tracking selected region", members: ["track"] },
  { key: "reconstruct", label: "Reconstructing video", members: ["reconstructing"] },
  { key: "finalize", label: "Finalizing export", members: ["audio", "verifying", "saving"] },
];

/**
 * The server's nine steps shown as four. Nothing is invented: a group is done when all its steps are done (or skipped),
 * active when one of them is, and the detail is the server's own wording for the step that is running.
 */
export function groupStages(stages: Stage[]): StageGroup[] {
  return GROUPS.map(({ key, label, members }) => {
    const own = stages.filter((s) => members.includes(s.key));
    const states = own.map((s) => s.state);
    let state: GroupState = "pending";
    if (own.length === 0) state = "pending";
    else if (states.includes("failed")) state = "failed";
    else if (states.includes("stopped")) state = "stopped";
    else if (states.includes("active")) state = "active";
    else if (states.every((x) => x === "skipped")) state = "skipped";
    else if (states.every((x) => x === "done" || x === "skipped")) state = "done";
    else if (states.includes("done")) state = "active"; // between two steps of the same group
    const current = own.find((s) => s.state === "active") ?? own.find((s) => s.state === "failed" || s.state === "stopped");
    const detail = current ? (current.detail ? `${current.label} · ${current.detail}` : current.label) : state === "skipped" ? (own[0]?.detail ?? null) : null;
    return { key, label, state, detail };
  });
}

export function isActive(job: Job | null): boolean {
  return job !== null && (job.status === "queued" || job.status === "running");
}

/**
 * Sizes worth offering: only ones smaller than the original's shorter side (so 1080×1920 counts as 1080), because the
 * server never enlarges. Largest first.
 */
export function heightOptions(originalWidth: number | null, originalHeight: number | null): OutputHeight[] {
  if (!originalWidth || !originalHeight) return [];
  const short = Math.min(originalWidth, originalHeight);
  return OUTPUT_HEIGHTS.filter((h) => h < short);
}

export function buildRequest(settings: RenderSettings, engine: EngineInfo | null, acknowledged: boolean): RenderRequest {
  return { ...settings, acknowledge_third_party: Boolean(engine?.sends_frames_to_third_party && acknowledged) };
}

export function formatRemaining(seconds: number | null): string | null {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 5) return "a few seconds";
  if (seconds < 60) return `about ${Math.round(seconds / 5) * 5} seconds`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `about ${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return `about ${hours} h${rest ? ` ${rest} min` : ""}`;
}

export function formatSpeed(fps: number | null): string | null {
  if (fps === null || !Number.isFinite(fps) || fps <= 0) return null;
  return fps >= 10 ? `${Math.round(fps)} frames/s` : `${fps.toFixed(1)} frames/s`;
}

export function queueMessage(job: Job): string | null {
  if (job.status !== "queued") return null;
  if (job.jobs_ahead === null || job.jobs_ahead === 0) return "Waiting for a free worker…";
  return `Waiting for a free worker. ${job.jobs_ahead} ${job.jobs_ahead === 1 ? "video is" : "videos are"} ahead of yours.`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

const AUDIO_TEXT: Record<RenderResult["audio"], string> = {
  none: "No audio (the original had none)",
  removed: "Audio removed, as you chose",
  copied: "Original audio kept unchanged",
  converted: "Original audio kept (converted once to fit the format)",
};

export function describeResult(result: RenderResult): string[] {
  return [
    `${result.width}×${result.height}, ${result.frames} frames at ${Number(result.fps.toFixed(2))} fps`,
    `${result.container.toUpperCase()}, ${formatBytes(result.size_bytes)}`,
    AUDIO_TEXT[result.audio],
    result.tracking_used ? "Followed the tracked marks" : result.automatic ? "Used the marks the detector placed" : "Used the marks where you placed them",
  ];
}

/** Reads one server-sent `job` event. Returns null for anything that isn't a well-formed job. */
export function parseJobEvent(data: string): Job | null {
  try {
    const value: unknown = JSON.parse(data);
    if (typeof value !== "object" || value === null) return null;
    const job = value as Partial<Job>;
    if (typeof job.id !== "string" || typeof job.status !== "string" || !Array.isArray(job.stages)) return null;
    return job as Job;
  } catch {
    return null;
  }
}

/** A newer view of the same job wins; a different job (a new render) replaces it; an older view never wins. */
export function mergeJob(current: Job | null, incoming: Job): Job {
  if (current === null || current.id !== incoming.id) return incoming;
  const terminal = (j: Job) => j.status === "completed" || j.status === "failed" || j.status === "cancelled";
  if (terminal(current) && !terminal(incoming)) return current;
  if (!terminal(current) && !terminal(incoming) && incoming.frames_done < current.frames_done && incoming.stage === current.stage) return current;
  return incoming;
}
