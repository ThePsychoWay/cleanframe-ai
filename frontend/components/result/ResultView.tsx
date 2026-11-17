"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Navbar } from "@/components/Navbar";
import { useToast } from "@/components/Toast";
import { Icon } from "@/components/ui/Icon";
import { ApiError } from "@/lib/api";
import { linkNeedsRefresh } from "@/lib/compare";
import { OWNERSHIP_NOTICE } from "@/lib/site";
import { formatBytes } from "@/lib/video";
import { describeResult } from "@/lib/render";
import { projectsService } from "@/services/projects";
import type { Project } from "@/types/project";
import type { RenderResult } from "@/types/render";
import type { PreviewInfo } from "@/types/result";
import { formatContainer, formatSize, resolutionLabel } from "@/lib/resolution";
import { CompareView } from "./CompareView";
import { ExportPanel } from "./ExportPanel";

type Load =
  | { status: "loading" }
  | { status: "blocked"; title: string; message: string; editorId?: string }
  | { status: "ready"; project: Project; preview: PreviewInfo; fetchedAt: number; version: number };

type Action = { status: "idle" } | { status: "working"; what: "download" | "delete" } | { status: "error"; message: string };

function messageOf(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function ResultView() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const toast = useToast();
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [action, setAction] = useState<Action>({ status: "idle" });
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [linkProblem, setLinkProblem] = useState<string | null>(null);
  const refreshing = useRef(false);
  const attempts = useRef<number[]>([]);

  const fetchPreview = useCallback(async (): Promise<{ project: Project; preview: PreviewInfo } | Load> => {
    try {
      const project = await projectsService.get(id);
      if (project.status === "EXPIRED") {
        return {
          status: "blocked",
          title: "This project has expired",
          message: "Its files were deleted after the retention period. Upload the video again to start a new project.",
        };
      }
      if (project.status !== "COMPLETED" && project.status !== "MASKING" && project.status !== "PROCESSING") {
        return { status: "blocked", title: "No finished video here", message: "This project doesn't have a finished video." };
      }
      const preview = await projectsService.preview(id);
      return { project, preview };
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return { status: "blocked", title: "Project not found", message: "It may have been deleted, or it isn't yours." };
      }
      if (error instanceof ApiError && error.status === 409) {
        return {
          status: "blocked",
          title: "No finished video yet",
          message: "Process the video in the editor first. Nothing has been restored for this project yet.",
          editorId: id,
        };
      }
      if (error instanceof ApiError && error.status === 410) {
        return { status: "blocked", title: "This video is no longer available", message: messageOf(error, "Process the video again to make a new one."), editorId: id };
      }
      return { status: "blocked", title: "We couldn't open this video", message: messageOf(error, "Please try again.") };
    }
  }, [id]);

  useEffect(() => {
    let cancelled = false;
    void fetchPreview().then((result) => {
      if (cancelled) return;
      if ("preview" in result) setLoad({ status: "ready", project: result.project, preview: result.preview, fetchedAt: Date.now(), version: 0 });
      else setLoad(result);
    });
    return () => {
      cancelled = true;
    };
  }, [fetchPreview]);

  // Signed links expire. Ask for new ones when the old ones are nearly used up (checked when the tab comes back),
  // and when a video fails to load.
  const refreshLinks = useCallback(async () => {
    if (refreshing.current) return;
    // A video that keeps failing shouldn't keep asking forever: three tries in a minute, then say so.
    const now = Date.now();
    attempts.current = [...attempts.current.filter((t) => now - t < 60_000), now];
    if (attempts.current.length > 3) {
      setLinkProblem("The video couldn't be loaded. Check your connection and reload the page.");
      return;
    }
    refreshing.current = true;
    try {
      const preview = await projectsService.preview(id);
      setLinkProblem(null);
      setLoad((current) => (current.status === "ready" ? { ...current, preview, fetchedAt: Date.now(), version: current.version + 1 } : current));
    } catch (error) {
      if (error instanceof ApiError && (error.status === 404 || error.status === 409 || error.status === 410)) {
        setLoad({
          status: "blocked",
          title: "This video is no longer available",
          message: messageOf(error, "Process the video again to make a new one."),
          editorId: id,
        });
      } else {
        setLinkProblem("We couldn't refresh the video link. Reload the page to try again.");
      }
    } finally {
      refreshing.current = false;
    }
  }, [id]);

  useEffect(() => {
    if (load.status !== "ready") return;
    const { fetchedAt, preview } = load;
    const check = () => {
      if (document.visibilityState === "visible" && linkNeedsRefresh(fetchedAt, preview.expires_in, Date.now())) void refreshLinks();
    };
    document.addEventListener("visibilitychange", check);
    return () => document.removeEventListener("visibilitychange", check);
  }, [load, refreshLinks]);

  const download = async () => {
    setAction({ status: "working", what: "download" });
    try {
      // A fresh link is asked for every time: it lasts minutes, not hours.
      const link = await projectsService.exportResult(id);
      const anchor = document.createElement("a");
      anchor.href = link.url;
      anchor.download = link.filename;
      anchor.rel = "noopener";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setAction({ status: "idle" });
      toast({ kind: "success", message: "Your download is starting." });
    } catch (error) {
      const message = messageOf(error, "We couldn't prepare the download.");
      setAction({ status: "error", message });
      toast({ kind: "error", message });
    }
  };

  const remove = async () => {
    setAction({ status: "working", what: "delete" });
    try {
      await projectsService.remove(id);
      toast({ kind: "success", message: "Project deleted." });
      router.replace("/projects");
    } catch (error) {
      setAction({ status: "error", message: messageOf(error, "We couldn't delete this project.") });
    }
  };

  const header = () => <Navbar step={4} />;

  if (load.status === "loading") {
    return (
      <div className="min-h-screen">
        {header()}
        <p role="status" className="mx-auto max-w-3xl px-4 py-16 text-dim">
          Opening your video…
        </p>
      </div>
    );
  }

  if (load.status === "blocked") {
    return (
      <div className="min-h-screen">
        {header()}
        <main className="mx-auto max-w-md px-4 py-16">
          <h1 className="heading text-2xl">{load.title}</h1>
          <p className="mt-3 text-dim">{load.message}</p>
          <div className="mt-6 flex gap-3">
            {load.editorId && (
              <Link href={`/editor/${load.editorId}`} className="btn btn-primary">
                Open the editor
              </Link>
            )}
            <Link href="/projects" className="btn btn-ghost">
              Back to projects
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const { project, preview } = load;
  const facts = preview.facts as Partial<RenderResult>;
  const lines = facts.width !== undefined && facts.fps !== undefined && facts.frames !== undefined && facts.container !== undefined && facts.audio !== undefined
    ? describeResult(facts as RenderResult)
    : [`${preview.width}×${preview.height}, ${preview.frame_count} frames`, `${preview.container.toUpperCase()}${preview.size_bytes ? `, ${formatBytes(preview.size_bytes)}` : ""}`];
  const busy = action.status === "working";

  return (
    <div className="min-h-screen">
      {header()}
      <main id="main" className="mx-auto max-w-3xl space-y-6 px-4 py-8 sm:py-10">
        <div className="text-center">
          <h1 className="heading text-[clamp(1.75rem,4vw,2.5rem)]">Your video is ready.</h1>
          <p className="mt-2 text-dim" title={project.original_filename}>
            {project.name}
          </p>
        </div>

        {preview.development_only && !preview.warnings.some((w) => /development engine/i.test(w)) && (
          <p className="rounded-xl border border-line bg-raised px-4 py-3 text-sm text-paper">
            This was made with the development engine, which is not AI: it only blurs the marked areas. Use it to check the
            workflow, not for final results.
          </p>
        )}
        {preview.warnings.length > 0 && (
          <ul className="space-y-1 rounded-xl border border-line px-4 py-3 text-sm text-paper">
            {preview.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}

        {facts.automatic && (
          <section aria-label="Automatic result" className="rounded-xl border border-line px-4 py-3 text-sm text-dim">
            <p>
              The overlays in this video were found and removed automatically. Open <em>Edit selection</em> to adjust what was
              removed.
            </p>
            {facts.quality &&
              (facts.quality.checked ? (
                <ul className="mt-2 space-y-1 text-paper">
                  {facts.quality.regions.map((r) => (
                    <li key={r.name}>
                      {r.name}:{" "}
                      {r.verdict === "clean"
                        ? "checked after rendering, no leftover found"
                        : r.verdict === "not_masked"
                          ? "had no frames that needed removing"
                          : `worth a look${r.problems?.length ? ` (${r.problems.join("; ")})` : ""}`}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2">The result could not be compared with the original{facts.quality.reason ? `: ${facts.quality.reason}` : "."}</p>
              ))}
          </section>
        )}

        {linkProblem && (
          <p role="alert" className="text-sm text-paper">
            {linkProblem}
          </p>
        )}
        <CompareView preview={preview} onVideoError={() => void refreshLinks()} linksVersion={load.version} />

        <dl className="tnum flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-sm" aria-label="Output details">
          {project.width && project.height ? (
            <div className="flex items-baseline gap-2">
              <dt className="text-faint">Original resolution</dt>
              <dd className="font-medium">{formatSize(project.width, project.height)}</dd>
            </div>
          ) : null}
          <div className="flex items-baseline gap-2">
            <dt className="text-faint">Output resolution</dt>
            <dd className="font-medium" title={resolutionLabel(preview.width, preview.height)}>
              {formatSize(preview.width, preview.height)}
            </dd>
          </div>
          <div className="flex items-baseline gap-2">
            <dt className="text-faint">Quality</dt>
            <dd className="font-medium">{resolutionLabel(preview.width, preview.height)}</dd>
          </div>
          <div className="flex items-baseline gap-2">
            <dt className="text-faint">Format</dt>
            <dd className="font-medium">{formatContainer(preview.container, `x.${preview.container}`)}</dd>
          </div>
          {preview.size_bytes ? (
            <div className="flex items-baseline gap-2">
              <dt className="text-faint">Size</dt>
              <dd className="font-medium">{formatBytes(preview.size_bytes)}</dd>
            </div>
          ) : null}
        </dl>
        <ul className="space-y-0.5 text-center text-xs text-faint">
          {lines.slice(2).map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>

        <div className="flex flex-wrap items-center justify-center gap-3">
          <button type="button" className="btn btn-primary !h-12 !px-6" onClick={() => void download()} disabled={busy}>
            <Icon name="download" size={18} />
            {action.status === "working" && action.what === "download" ? "Preparing…" : "Download video"}
          </button>
          <Link href={`/editor/${project.id}`} className="btn btn-ghost !h-12 !px-6">
            Edit selection
          </Link>
          <Link href="/dashboard" className="btn btn-quiet !h-12 !px-4">
            Start another project
          </Link>
        </div>
        {action.status === "error" && (
          <p role="alert" className="text-center text-sm text-paper">
            {action.message}
          </p>
        )}

        <ExportPanel projectId={project.id} />

        <div className="flex flex-wrap items-center justify-center gap-3 border-t border-line pt-6">
          {confirmingDelete ? (
            <>
              <span className="text-sm text-dim">Delete this video, the restored one and the project?</span>
              <button type="button" className="btn btn-primary !h-10 !px-4 !text-sm" onClick={() => void remove()} disabled={busy}>
                {action.status === "working" && action.what === "delete" ? "Deleting…" : "Delete"}
              </button>
              <button type="button" className="btn btn-ghost !h-10 !px-4 !text-sm" onClick={() => setConfirmingDelete(false)} disabled={busy}>
                Keep
              </button>
            </>
          ) : (
            <button type="button" className="btn btn-quiet !h-10 !text-sm" onClick={() => setConfirmingDelete(true)} disabled={busy}>
              <Icon name="trash" size={16} />
              Delete project
            </button>
          )}
        </div>
        <p className="text-center text-xs text-faint">{OWNERSHIP_NOTICE}</p>
      </main>
    </div>
  );
}
