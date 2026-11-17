import type { Project, ProjectStatus } from "../types/project";

export type BadgeTone = "working" | "done" | "failed" | "neutral";

/**
 * How a project's status is worded to the person. Three headline labels (Processing, Completed, Failed); the states that
 * need the person to do something, or that have ended, keep their own plain names.
 */
export const STATUS_BADGE: Record<ProjectStatus, { label: string; tone: BadgeTone }> = {
  UPLOADING: { label: "Upload unfinished", tone: "neutral" },
  UPLOADED: { label: "Processing", tone: "working" },
  ANALYZING: { label: "Processing", tone: "working" },
  MASKING: { label: "Ready to edit", tone: "neutral" },
  TRACKING: { label: "Processing", tone: "working" },
  PROCESSING: { label: "Processing", tone: "working" },
  ENCODING: { label: "Processing", tone: "working" },
  COMPLETED: { label: "Completed", tone: "done" },
  FAILED: { label: "Failed", tone: "failed" },
  EXPIRED: { label: "Expired", tone: "neutral" },
};

/** Statuses where the server is still working by itself, so the list keeps refreshing. */
export const POLL_WHILE: ReadonlySet<ProjectStatus> = new Set(["UPLOADED", "ANALYZING", "TRACKING", "PROCESSING", "ENCODING"]);

/** Statuses that have a preview picture to show (the editor's preview copy exists from MASKING on). */
export const HAS_PREVIEW: ReadonlySet<ProjectStatus> = new Set(["MASKING", "TRACKING", "PROCESSING", "ENCODING", "COMPLETED"]);

/** Where clicking a project goes, or null when there is nowhere to go yet (or any more). */
export function projectHref(project: Pick<Project, "id" | "status">): string | null {
  switch (project.status) {
    case "COMPLETED":
      return `/result/${project.id}`;
    case "MASKING":
    case "TRACKING":
    case "PROCESSING":
    case "ENCODING":
      return `/editor/${project.id}`;
    default:
      return null;
  }
}

export function filterProjects(projects: Project[], query: string): Project[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return projects;
  return projects.filter((p) => p.name.toLowerCase().includes(needle) || p.original_filename.toLowerCase().includes(needle));
}

/** Search is only worth showing once the list is long enough to need it. */
export const SEARCH_FROM = 6;

export function formatCreated(iso: string, locale?: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(locale, { month: "short", day: "numeric", year: "numeric" });
}
