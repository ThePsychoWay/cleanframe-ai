"use client";

import { Icon } from "@/components/ui/Icon";
import { formatRemaining, formatSpeed, groupStages, queueMessage, type StageGroup } from "@/lib/render";
import type { Job } from "@/types/render";

interface ProcessingProgressProps {
  job: Job;
  busy: boolean;
  onCancel: () => void;
}

function GroupRow({ group }: { group: StageGroup }) {
  const active = group.state === "active";
  const dim = group.state === "pending" || group.state === "skipped";
  return (
    <li className="flex items-start gap-3" aria-current={active ? "step" : undefined}>
      <span
        aria-label={group.state}
        className={`mt-0.5 grid size-6 shrink-0 place-items-center rounded-full text-xs ${
          group.state === "done"
            ? "bg-mask text-ink"
            : active
              ? "bg-mask/15 text-mask ring-1 ring-mask"
              : group.state === "failed"
                ? "bg-[#3a1d19] text-danger"
                : "bg-raised text-faint ring-1 ring-line"
        }`}
      >
        {group.state === "done" ? (
          <Icon name="check" size={14} strokeWidth={2.5} />
        ) : active ? (
          <Icon name="spinner" size={14} strokeWidth={2.5} className="animate-spin-slow" />
        ) : group.state === "failed" ? (
          <Icon name="close" size={14} strokeWidth={2.5} />
        ) : group.state === "stopped" ? (
          <span className="size-2 rounded-sm bg-current" />
        ) : (
          <span className="size-1.5 rounded-full bg-current" />
        )}
      </span>
      <span className="min-w-0">
        <span className={`block text-sm font-medium ${dim ? "text-faint" : "text-paper"}`}>
          {group.label}
        </span>
        {group.detail && group.state !== "pending" && <span className="tnum block text-xs text-dim">{group.detail}</span>}
      </span>
    </li>
  );
}

/** What the server is really doing, in four steps, with its own progress numbers and nothing made up. */
export function ProcessingProgress({ job, busy, onCancel }: ProcessingProgressProps) {
  const groups = groupStages(job.stages);
  const percent = job.percent;
  const speed = formatSpeed(job.speed_fps);
  const remaining = formatRemaining(job.eta_seconds);
  const queued = job.status === "queued";

  return (
    <section aria-labelledby="processing-heading" className="surface p-5 sm:p-6">
      <h2 id="processing-heading" className="heading text-xl">
        {queued ? "Waiting to start" : "Removing the watermark"}
      </h2>
      <p className="mt-1 text-sm text-dim">You can leave this page. It keeps going and your video appears under My projects.</p>

      {queued && (
        <p role="status" className="mt-4 text-sm text-dim">
          {queueMessage(job)}
        </p>
      )}

      {/* A bar with a number only when the server has one; otherwise a bar that says it is working, without a figure. */}
      <div className="mt-5">
        {percent !== null ? (
          <>
            <div
              role="progressbar"
              aria-label="Reconstruction progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(percent)}
              className="h-2 overflow-hidden rounded-full bg-raised"
            >
              <div className="h-full rounded-full bg-mask transition-[width] duration-300" style={{ width: `${percent}%` }} />
            </div>
            <p className="tnum mt-2 text-sm text-dim">
              {Math.round(percent)}% · {job.frames_done} of {job.frames_total} frames{speed ? ` · ${speed}` : ""}
            </p>
            {remaining && job.status === "running" && <p className="text-xs text-faint">About {remaining.replace(/^about /, "")} left for this step.</p>}
          </>
        ) : (
          <div role="progressbar" aria-label="Working" aria-valuetext="Working" className="relative h-2 overflow-hidden rounded-full bg-raised">
            <div className="animate-indeterminate absolute inset-y-0 left-0 w-2/5 rounded-full bg-mask/70" />
          </div>
        )}
      </div>

      <ol className="mt-6 space-y-4" aria-label="Processing steps">
        {groups.map((g) => (
          <GroupRow key={g.key} group={g} />
        ))}
      </ol>

      <button type="button" className="btn btn-ghost mt-6" disabled={busy || job.stage === "saving"} onClick={onCancel}>
        {busy ? "Stopping…" : "Cancel"}
      </button>
      <p className="mt-3 text-xs text-faint">Your selection is locked while this runs.</p>
    </section>
  );
}
