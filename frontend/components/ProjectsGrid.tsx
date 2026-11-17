"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { useToast } from "@/components/Toast";
import { POLL_WHILE, SEARCH_FROM, filterProjects } from "@/lib/projects";
import { projectsService } from "@/services/projects";
import type { Project } from "@/types/project";
import { ProjectCard } from "./ProjectCard";

type LoadState = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; projects: Project[] };

export function ProjectsGrid() {
  const toast = useToast();
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [query, setQuery] = useState("");
  const [confirming, setConfirming] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const projects = await projectsService.list();
      setState({ status: "ready", projects });
    } catch (error) {
      // Keep showing the last good list if a background refresh fails.
      setState((current) =>
        current.status === "ready" ? current : { status: "error", message: error instanceof Error ? error.message : "We couldn't load your projects." },
      );
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Analysis, tracking and rendering finish on their own, so keep an eye on anything still in progress.
  const waiting = state.status === "ready" && state.projects.some((p) => POLL_WHILE.has(p.status));
  useEffect(() => {
    if (!waiting) return;
    const timer = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(timer);
  }, [waiting, load]);

  const remove = async (project: Project) => {
    setBusyId(project.id);
    try {
      await projectsService.remove(project.id);
      setConfirming(null);
      toast({ kind: "success", message: `Deleted “${project.name}”.` });
      await load();
    } catch (error) {
      toast({ kind: "error", message: error instanceof Error ? error.message : "We couldn't delete that project." });
    } finally {
      setBusyId(null);
    }
  };

  const retry = async (project: Project) => {
    setBusyId(project.id);
    try {
      await projectsService.analyze(project.id);
      await load();
    } catch (error) {
      toast({ kind: "error", message: error instanceof Error ? error.message : "We couldn't restart the analysis." });
    } finally {
      setBusyId(null);
    }
  };

  const projects = state.status === "ready" ? state.projects : [];
  const shown = useMemo(() => filterProjects(projects, query), [projects, query]);

  return (
    <section aria-labelledby="projects-heading">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 id="projects-heading" className="heading text-3xl">
            My projects
          </h1>
          <p className="mt-1.5 text-dim">Your uploaded and restored videos.</p>
        </div>
        <div className="flex w-full items-center gap-3 sm:w-auto">
          {state.status === "ready" && projects.length >= SEARCH_FROM && (
            <label className="relative block min-w-0 flex-1 sm:w-64 sm:flex-none">
              <span className="sr-only">Search projects</span>
              <Icon name="search" size={18} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-faint" />
              <input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search projects"
                className="field h-11 w-full !pl-10"
              />
            </label>
          )}
          <Link href="/dashboard" className="btn btn-primary !h-11 shrink-0 !px-4 !text-sm">
            <Icon name="plus" size={16} />
            New project
          </Link>
        </div>
      </div>

      {state.status === "loading" && (
        <p role="status" className="mt-10 text-dim">
          Loading your projects…
        </p>
      )}

      {state.status === "error" && (
        <div className="surface mt-8 p-8">
          <p className="font-medium">We couldn&apos;t load your projects</p>
          <p className="mt-1 max-w-md text-dim">{state.message}</p>
          <button type="button" className="btn btn-ghost mt-4" onClick={() => void load()}>
            Try again
          </button>
        </div>
      )}

      {state.status === "ready" && projects.length === 0 && (
        <div className="surface mt-8 grid place-items-center px-6 py-16 text-center">
          <span className="grid size-14 place-items-center rounded-2xl bg-raised text-dim">
            <Icon name="film" size={26} />
          </span>
          <p className="mt-5 text-lg font-medium">No projects yet</p>
          <p className="mt-1 max-w-sm text-dim">Upload a video to start your first one. It will appear here.</p>
          <Link href="/dashboard" className="btn btn-primary mt-6">
            Upload a video
          </Link>
        </div>
      )}

      {state.status === "ready" && projects.length > 0 && shown.length === 0 && (
        <p role="status" className="mt-10 text-dim">
          No projects match “{query}”.
        </p>
      )}

      {shown.length > 0 && (
        <ul className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              busy={busyId === project.id}
              confirming={confirming === project.id}
              onAskDelete={() => setConfirming(project.id)}
              onCancelDelete={() => setConfirming(null)}
              onDelete={() => void remove(project)}
              onRetry={() => void retry(project)}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
