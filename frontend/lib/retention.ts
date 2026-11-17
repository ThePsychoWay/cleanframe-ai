import type { Project } from "@/types/project";

/** "Nov 5, 2026, 10:00 AM" in the person's own language and time zone. */
export function formatWhen(iso: string, locale?: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(locale, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

/**
 * What to say about when a project's files go. The backend deletes them at `expires_at` (when somebody opens the
 * project, or when its periodic sweep runs), so a date in the future is a promise it keeps and a date in the past is
 * described as due, not as already done.
 */
export function describeRetention(project: Pick<Project, "status" | "expires_at">, now: Date = new Date(), locale?: string): string {
  if (project.status === "EXPIRED") {
    return "The files were deleted after the retention period. Upload the video again to start over.";
  }
  const when = formatWhen(project.expires_at, locale);
  if (when === "") return "";
  if (new Date(project.expires_at).getTime() <= now.getTime()) return "Past its retention period; the files are being deleted.";
  return `Files are deleted automatically on ${when}.`;
}
