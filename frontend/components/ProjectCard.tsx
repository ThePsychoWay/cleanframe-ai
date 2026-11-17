"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { HAS_PREVIEW, STATUS_BADGE, formatCreated, projectHref, type BadgeTone } from "@/lib/projects";
import { describeRetention } from "@/lib/retention";
import { formatDuration } from "@/lib/video";
import { projectsService } from "@/services/projects";
import type { Project } from "@/types/project";

const TONE: Record<BadgeTone, string> = {
  working: "bg-mask/15 text-mask",
  done: "bg-mask text-ink",
  failed: "bg-[#3a1d19] text-danger",
  neutral: "bg-raised text-dim",
};

interface ProjectCardProps {
  project: Project;
  busy: boolean;
  confirming: boolean;
  onAskDelete: () => void;
  onCancelDelete: () => void;
  onDelete: () => void;
  onRetry: () => void;
}

export function ProjectCard({ project, busy, confirming, onAskDelete, onCancelDelete, onDelete, onRetry }: ProjectCardProps) {
  const badge = STATUS_BADGE[project.status];
  const href = projectHref(project);
  const thumb = useThumbnail(project);
  const working = badge.tone === "working";

  const picture = (
    <div className="relative aspect-video w-full overflow-hidden bg-black">
      {thumb.url ? (
        // eslint-disable-next-line @next/next/no-img-element -- a short-lived signed URL; the image optimiser can't reach it
        <img src={thumb.url} alt="" className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-[1.03]" />
      ) : (
        <div className="grid h-full w-full place-items-center text-faint">
          <Icon name="film" size={28} />
        </div>
      )}
      <span className={`absolute left-2.5 top-2.5 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${TONE[badge.tone]}`}>
        {working && <span aria-hidden="true" className="size-2 animate-pulse rounded-full bg-current" />}
        {badge.label}
      </span>
    </div>
  );

  return (
    <li ref={thumb.ref} className="surface group flex flex-col overflow-hidden transition-colors duration-200 hover:border-[#3f3f43]">
      {href ? (
        <Link href={href} aria-label={`Open ${project.name}`} className="block">
          {picture}
        </Link>
      ) : (
        picture
      )}

      <div className="flex flex-1 flex-col p-4">
        <p className="truncate font-medium" title={project.original_filename}>
          {project.name}
        </p>
        <p className="tnum mt-1 text-sm text-dim">
          {project.duration != null ? formatDuration(project.duration) : "—"} · {formatCreated(project.created_at)}
        </p>
        {project.status === "FAILED" && project.error && <p className="mt-2 line-clamp-3 text-sm text-faint">{project.error}</p>}
        <p className="mt-2 text-xs text-faint">{describeRetention(project)}</p>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          {project.status === "COMPLETED" && (
            <Link href={`/result/${project.id}`} className="btn btn-primary !h-10 !px-4 !text-sm">
              Open result
            </Link>
          )}
          {project.status === "MASKING" && (
            <Link href={`/editor/${project.id}`} className="btn btn-primary !h-10 !px-4 !text-sm">
              Continue
            </Link>
          )}
          {(project.status === "PROCESSING" || project.status === "TRACKING" || project.status === "ENCODING") && (
            <Link href={`/editor/${project.id}`} className="btn btn-ghost !h-10 !px-4 !text-sm">
              View progress
            </Link>
          )}
          {project.status === "COMPLETED" && (
            <Link href={`/editor/${project.id}`} className="btn btn-ghost !h-10 !px-4 !text-sm">
              Edit selection
            </Link>
          )}
          {project.status === "UPLOADED" && (
            <button type="button" className="btn btn-ghost !h-10 !px-4 !text-sm" disabled={busy} onClick={onRetry}>
              Retry analysis
            </button>
          )}

          <span className="ml-auto flex items-center gap-2">
            {confirming ? (
              <>
                <span className="text-sm text-dim">Delete this video and its project?</span>
                <button type="button" className="btn btn-primary !h-10 !px-3 !text-sm" disabled={busy} onClick={onDelete}>
                  {busy ? "Deleting…" : "Delete"}
                </button>
                <button type="button" className="btn btn-ghost !h-10 !px-3 !text-sm" disabled={busy} onClick={onCancelDelete}>
                  Keep
                </button>
              </>
            ) : (
              <button type="button" className="btn btn-quiet btn-icon" aria-label={`Delete ${project.name}`} title="Delete project" onClick={onAskDelete}>
                <Icon name="trash" size={18} />
              </button>
            )}
          </span>
        </div>
      </div>
    </li>
  );
}

/**
 * The picture for a card comes from the project's own preview copy, asked for only once the card is on screen.
 * Projects without a preview yet (still uploading, failed, expired) show a plain film icon: nothing is invented.
 */
function useThumbnail(project: Project) {
  const ref = useRef<HTMLLIElement>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const eligible = HAS_PREVIEW.has(project.status);

  useEffect(() => {
    const node = ref.current;
    if (!node || !eligible) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [eligible]);

  useEffect(() => {
    if (!visible || !eligible || url) return;
    let cancelled = false;
    projectsService.media(project.id).then(
      (media) => {
        if (!cancelled) setUrl(media.thumbnail_url);
      },
      () => undefined, // no picture is fine; the card still works
    );
    return () => {
      cancelled = true;
    };
  }, [visible, eligible, url, project.id]);

  return { ref, url };
}
