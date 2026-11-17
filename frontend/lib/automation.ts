/**
 * What the automatic pipeline tells the person, written from what the server recorded. Nothing here is estimated:
 * every sentence is built from a field the detector or the render filled in.
 */

import type { Automation, DetectedOverlay, Project } from "@/types/project";

export type Outcome =
  | { kind: "working"; step: "uploading" | "analysing" | "restoring" }
  | { kind: "done" }
  | { kind: "needs_help"; title: string; message: string }
  | { kind: "failed"; message: string };

/** Where an uploaded project is, as one of a few honest outcomes the page can show. */
export function outcomeOf(project: Project): Outcome {
  switch (project.status) {
    case "UPLOADING":
      return { kind: "working", step: "uploading" };
    case "UPLOADED":
    case "ANALYZING":
      return { kind: "working", step: "analysing" };
    case "PROCESSING":
    case "ENCODING":
    case "TRACKING":
      return { kind: "working", step: "restoring" };
    case "COMPLETED":
      return { kind: "done" };
    case "FAILED":
    case "EXPIRED":
      return { kind: "failed", message: project.error ?? "This video couldn't be processed." };
    case "MASKING":
      return needsHelp(project.automation ?? null);
    default:
      return { kind: "working", step: "analysing" };
  }
}

function needsHelp(automation: Automation | null): Outcome {
  switch (automation?.state) {
    case "nothing_found":
      return {
        kind: "needs_help",
        title: "No watermark could be found automatically",
        message:
          "The video was checked on every frame and nothing was steady enough on screen to be sure it is an overlay. Nothing was removed, and nothing was passed off as processed. You can mark the area yourself in the editor.",
      };
    case "detection_failed":
      return {
        kind: "needs_help",
        title: "Automatic detection couldn't run on this video",
        message: "Nothing was removed. You can mark the area yourself in the editor.",
      };
    case "engine_unavailable":
      return {
        kind: "needs_help",
        title: "Overlays were found, but the removal engine isn't ready",
        message: automation.state_reason ?? "The reconstruction engine isn't ready.",
      };
    case "ready":
      return {
        kind: "needs_help",
        title: "Overlays were found",
        message: "Automatic processing is switched off on this server, so nothing has been removed yet. Review the marks and start it from the editor.",
      };
    default:
      return {
        kind: "needs_help",
        title: "Ready to mark",
        message: "This video hasn't been processed. Open the editor to mark what to remove.",
      };
  }
}

const KIND: Record<DetectedOverlay["kind"], string> = { text: "Text", logo: "Logo or mark" };

/** "Logo or mark, frames 0-149 (150 of 150)". Frame numbers are the video's own. */
export function describeOverlay(overlay: DetectedOverlay, total: number | null): string {
  const span =
    overlay.first_frame === null || overlay.last_frame === null
      ? ""
      : overlay.first_frame === 0 && total !== null && overlay.last_frame >= total - 1
        ? ", on every frame"
        : `, frames ${overlay.first_frame} to ${overlay.last_frame}`;
  const moves = overlay.notes.includes("moves between positions") ? ", moves between positions" : "";
  return `${KIND[overlay.kind]}${span}${moves}`;
}

/** Short note on how sure the detector was, and what that rests on. */
export function basisNote(overlay: DetectedOverlay): string {
  if (overlay.basis === "appearance") return "Judged by its look and position only: nothing moved around it to prove it.";
  if (overlay.basis === "copy") return "Matched to another mark that was proven.";
  return "Stayed put while the picture changed around it.";
}

export function removed(automation: Automation | null | undefined): DetectedOverlay[] {
  return (automation?.overlays ?? []).filter((o) => o.decision === "remove");
}

export function left(automation: Automation | null | undefined): DetectedOverlay[] {
  return (automation?.overlays ?? []).filter((o) => o.decision === "left");
}
