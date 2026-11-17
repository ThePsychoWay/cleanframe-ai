"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { useToast } from "@/components/Toast";
import { Icon } from "@/components/ui/Icon";
import { SAME_SIZE, buildExportRequest, describeExport, formatKey, hasChoices, isRunning } from "@/lib/exports";
import { formatBytes } from "@/lib/video";
import { exportsService } from "@/services/exports";
import type { ExportJob, ExportOptions } from "@/types/exports";

function messageOf(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

/**
 * Additional copies of the finished video at a smaller size or in another format. Only options the server says are valid are
 * listed (it never enlarges), and nothing is rebuilt: each export is one re-encode of the finished file.
 */
export function ExportPanel({ projectId }: { projectId: string }) {
  const uid = useId();
  const toast = useToast();
  const [options, setOptions] = useState<ExportOptions | null>(null);
  const [jobs, setJobs] = useState<ExportJob[]>([]);
  const [size, setSize] = useState(SAME_SIZE);
  const [format, setFormat] = useState("");
  const [working, setWorking] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [loaded, list] = await Promise.all([exportsService.options(projectId), exportsService.list(projectId)]);
        if (cancelled) return;
        setOptions(loaded);
        setJobs(list.exports);
        setSize(loaded.sizes[0] ? String(loaded.sizes[0].height) : SAME_SIZE);
        const same = loaded.formats.find((f) => f.container === loaded.container && f.codec === loaded.codec) ?? loaded.formats[0];
        if (same) setFormat(formatKey(same));
      } catch {
        // No panel is better than a broken one: the main download above still works.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  const refresh = useCallback(async () => {
    try {
      const list = await exportsService.list(projectId);
      if (alive.current) setJobs(list.exports);
    } catch {
      /* try again on the next tick */
    }
  }, [projectId]);

  const anyRunning = jobs.some(isRunning);
  useEffect(() => {
    if (!anyRunning) return;
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [anyRunning, refresh]);

  if (!options || !hasChoices(options)) return null;

  const request = buildExportRequest(options, size, format);

  const create = async () => {
    if (!request) return;
    setProblem(null);
    setWorking("create");
    try {
      await exportsService.create(projectId, request);
      await refresh();
    } catch (error) {
      setProblem(messageOf(error, "We couldn't start that export."));
    } finally {
      setWorking(null);
    }
  };

  const download = async (job: ExportJob) => {
    setWorking(job.id);
    try {
      const link = await exportsService.link(projectId, job.id);
      const anchor = document.createElement("a");
      anchor.href = link.url;
      anchor.download = link.filename;
      anchor.rel = "noopener";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      toast({ kind: "success", message: "Your download is starting." });
    } catch (error) {
      setProblem(messageOf(error, "We couldn't prepare that download."));
    } finally {
      setWorking(null);
    }
  };

  return (
    <section aria-labelledby={`${uid}-heading`} className="rounded-xl border border-line p-4">
      <h2 id={`${uid}-heading`} className="heading text-base">
        More sizes and formats
      </h2>
      <p className="mt-1 text-xs leading-relaxed text-faint">
        Makes another copy of the finished video. Only smaller sizes are offered: enlarging would stretch the pixels that are there, not add detail.
        {options.hardware_encoder ? " Encoded on the GPU." : ""}
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-[1fr_1fr_auto] sm:items-end">
        <label className="block text-sm text-dim">
          Size
          <select className="field mt-1 w-full" value={size} onChange={(e) => setSize(e.target.value)} disabled={working !== null}>
            <option value={SAME_SIZE}>
              Same as the finished video ({options.output_width} × {options.output_height})
            </option>
            {options.sizes.map((s) => (
              <option key={s.height} value={s.height}>
                {s.label} ({s.width_px} × {s.height_px})
              </option>
            ))}
          </select>
        </label>
        <label className="block text-sm text-dim">
          Format
          <select className="field mt-1 w-full" value={format} onChange={(e) => setFormat(e.target.value)} disabled={working !== null}>
            {options.formats.map((f) => (
              <option key={formatKey(f)} value={formatKey(f)}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn btn-ghost !h-11 !px-4" onClick={() => void create()} disabled={!request || working !== null}>
          {working === "create" ? "Starting…" : "Make this copy"}
        </button>
      </div>
      {!request && <p className="mt-2 text-xs text-faint">That is the video you already have. Pick another size or format.</p>}
      {problem && (
        <p role="alert" className="mt-2 text-sm text-paper">
          {problem}
        </p>
      )}

      {jobs.length > 0 && (
        <ul className="mt-4 space-y-2" aria-label="Your exports">
          {jobs.map((job) => (
            <li key={job.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line px-3 py-2 text-sm">
              <span className="min-w-0">
                <span className="font-medium">{describeExport(job)}</span>
                {job.status === "completed" && job.result?.size_bytes ? <span className="ml-2 text-faint">{formatBytes(job.result.size_bytes)}</span> : null}
              </span>
              {isRunning(job) && (
                <span role="status" className="tnum text-dim">
                  {job.status === "queued" ? "Waiting…" : job.percent !== null ? `Encoding ${Math.round(job.percent)}%` : "Encoding…"}
                </span>
              )}
              {job.status === "failed" && <span className="text-paper">{job.error ?? "This export failed."}</span>}
              {job.status === "cancelled" && <span className="text-dim">Cancelled</span>}
              {job.status === "completed" && (
                <button type="button" className="btn btn-ghost !h-9 !px-3 !text-sm" onClick={() => void download(job)} disabled={working !== null}>
                  <Icon name="download" size={16} />
                  {working === job.id ? "Preparing…" : "Download"}
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
