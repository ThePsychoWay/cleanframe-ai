"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { isActive, mergeJob } from "@/lib/render";
import { projectsService } from "@/services/projects";
import { followJob } from "@/services/renderEvents";
import type { Job, RenderRequest } from "@/types/render";

const POLL_MS = 2000;

interface Options {
  projectId: string;
  initial: Job | null;
  /** Called once each time a job reaches an end state (completed, failed or cancelled) while this page is open. */
  onFinished: (job: Job) => void;
}

/**
 * The project's latest render job, kept current while it is waiting or running: live from the server's event stream,
 * or by asking every two seconds when the stream can't be kept.
 */
export function useRenderJob({ projectId, initial, onFinished }: Options) {
  const [job, setJob] = useState<Job | null>(initial);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const finished = useRef(onFinished);
  finished.current = onFinished;
  const wasActive = useRef(isActive(initial));

  const accept = useCallback((incoming: Job) => {
    setJob((current) => mergeJob(current, incoming));
  }, []);

  useEffect(() => {
    const active = isActive(job);
    if (wasActive.current && !active && job) finished.current(job);
    wasActive.current = active;
  }, [job]);

  const jobId = job?.id ?? null;
  const following = isActive(job);
  useEffect(() => {
    if (!following) return;
    let stopped = false;
    let timer: number | undefined;
    const poll = () => {
      let failures = 0;
      timer = window.setInterval(async () => {
        try {
          const latest = await projectsService.render(projectId);
          if (stopped) return;
          failures = 0;
          if (latest.job) accept(latest.job);
        } catch {
          failures += 1;
          if (!stopped && failures === 5) setProblem("We lost contact with the server. Reload the page to see how the render went.");
        }
      }, POLL_MS);
    };
    const stream = followJob(projectId, accept, poll);
    return () => {
      stopped = true;
      stream.close();
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [following, jobId, projectId, accept]);

  const run = useCallback(async (action: () => Promise<{ job: Job | null }>, fallback: string): Promise<boolean> => {
    setBusy(true);
    setProblem(null);
    try {
      const response = await action();
      const next = response.job;
      if (next) setJob((current) => (current?.id === next.id ? mergeJob(current, next) : next));
      return true;
    } catch (error) {
      setProblem(error instanceof ApiError ? error.message : fallback);
      return false;
    } finally {
      setBusy(false);
    }
  }, []);

  const start = useCallback(
    (request: RenderRequest) => run(() => projectsService.startRender(projectId, request), "We couldn't start processing. Please try again."),
    [projectId, run],
  );
  const cancel = useCallback(() => run(() => projectsService.cancelRender(projectId), "We couldn't stop the job. Please try again."), [projectId, run]);
  const reopen = useCallback(() => run(() => projectsService.reopen(projectId), "We couldn't reopen this video for editing."), [projectId, run]);

  return { job, problem, busy, start, cancel, reopen, clearProblem: () => setProblem(null) };
}
