"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Navbar } from "@/components/Navbar";
import { ProcessingProgress } from "@/components/editor/ProcessingProgress";
import { ApiError } from "@/lib/api";
import { basisNote, describeOverlay, left, outcomeOf, removed } from "@/lib/automation";
import { OWNERSHIP_NOTICE } from "@/lib/site";
import { projectsService } from "@/services/projects";
import type { Project } from "@/types/project";
import type { Job } from "@/types/render";

const POLL_MS = 1000;

type Load = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; project: Project };

/**
 * The whole job for the person: upload, watch, download. Nothing to mark and nothing to confirm. It shows what the
 * server really reports (analysis, what was found, the render's own progress) and moves on to the result by itself.
 */
export function ProcessView() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [job, setJob] = useState<Job | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const stopped = useRef(false);

  const tick = useCallback(async () => {
    try {
      const project = await projectsService.get(id);
      if (stopped.current) return;
      setLoad({ status: "ready", project });
      if (project.status === "COMPLETED") {
        stopped.current = true;
        router.replace(`/result/${id}`);
        return;
      }
      if (project.status === "PROCESSING" || project.status === "ENCODING") {
        const { job: latest } = await projectsService.render(id);
        if (!stopped.current) setJob(latest);
      }
    } catch (error) {
      if (stopped.current) return;
      const message =
        error instanceof ApiError && error.status === 404 ? "This project wasn't found. It may have been deleted." : error instanceof Error ? error.message : "We couldn't reach the server.";
      setLoad({ status: "error", message });
    }
  }, [id, router]);

  useEffect(() => {
    stopped.current = false;
    void tick();
    const timer = setInterval(() => void tick(), POLL_MS);
    return () => {
      stopped.current = true;
      clearInterval(timer);
    };
  }, [tick]);

  const cancel = async () => {
    setCancelling(true);
    try {
      await projectsService.cancelRender(id);
    } finally {
      setCancelling(false);
    }
  };

  const shell = (body: React.ReactNode) => (
    <div className="min-h-screen">
      <Navbar step={3} />
      <main id="main" className="mx-auto max-w-2xl space-y-6 px-4 py-10">
        {body}
        <p className="text-center text-xs text-faint">{OWNERSHIP_NOTICE}</p>
      </main>
    </div>
  );

  if (load.status === "loading") {
    return shell(
      <p role="status" className="text-dim">
        Opening your video…
      </p>,
    );
  }
  if (load.status === "error") {
    return shell(
      <>
        <h1 className="heading text-2xl">We couldn&apos;t open this video</h1>
        <p className="text-dim">{load.message}</p>
        <Link href="/projects" className="btn btn-ghost">
          Back to projects
        </Link>
      </>,
    );
  }

  const { project } = load;
  const outcome = outcomeOf(project);
  const found = project.automation;
  const taken = removed(found);
  const spared = left(found);

  return shell(
    <>
      <div className="text-center">
        <h1 className="heading text-[clamp(1.6rem,4vw,2.25rem)]">
          {outcome.kind === "needs_help" ? outcome.title : outcome.kind === "failed" ? "This video couldn't be processed" : "Cleaning your video"}
        </h1>
        <p className="mt-2 text-dim" title={project.original_filename}>
          {project.name}
        </p>
      </div>

      {outcome.kind === "working" && outcome.step !== "restoring" && (
        <section aria-live="polite" className="surface p-5 sm:p-6">
          <p className="flex items-center gap-3 text-paper">
            <span className="size-2.5 animate-pulse rounded-full bg-mask" aria-hidden="true" />
            Reading every frame and looking for overlays…
          </p>
          <p className="mt-2 text-sm text-dim">
            Static, transparent and text watermarks are looked for across the whole video, including ones that move or come and go. You don&apos;t need to mark
            anything.
          </p>
        </section>
      )}

      {outcome.kind === "working" && outcome.step === "restoring" && (
        <>
          {job ? (
            <ProcessingProgress job={job} busy={cancelling} onCancel={() => void cancel()} />
          ) : (
            <p role="status" className="text-dim">
              Starting…
            </p>
          )}
        </>
      )}

      {outcome.kind === "needs_help" && (
        <section className="surface space-y-4 p-5 sm:p-6">
          <p className="text-dim">{outcome.message}</p>
          <div className="flex flex-wrap gap-3">
            <Link href={`/editor/${project.id}`} className="btn btn-primary">
              Open the editor
            </Link>
            <Link href="/dashboard" className="btn btn-ghost">
              Upload another video
            </Link>
          </div>
        </section>
      )}

      {outcome.kind === "failed" && (
        <section className="surface space-y-4 p-5 sm:p-6">
          <p role="alert" className="text-paper">
            {outcome.message}
          </p>
          <Link href="/dashboard" className="btn btn-primary">
            Upload another video
          </Link>
        </section>
      )}

      {found && (taken.length > 0 || spared.length > 0) && (
        <section aria-labelledby="found-heading" className="surface p-5 sm:p-6">
          <h2 id="found-heading" className="text-sm font-medium text-paper">
            What was found
          </h2>
          <ul className="mt-3 space-y-3 text-sm">
            {taken.map((o) => (
              <li key={o.id}>
                <span className="text-paper">Removing: {describeOverlay(o, project.frame_count)}</span>
                <span className="block text-xs text-dim">{o.reason || basisNote(o)}</span>
              </li>
            ))}
            {spared.map((o) => (
              <li key={o.id}>
                <span className="text-dim">Left alone: {describeOverlay(o, project.frame_count)}</span>
                <span className="block text-xs text-faint">{o.reason}</span>
              </li>
            ))}
          </ul>
          {found.notes.map((n) => (
            <p key={n} className="mt-3 text-xs text-dim">
              {n}
            </p>
          ))}
        </section>
      )}
    </>,
  );
}
