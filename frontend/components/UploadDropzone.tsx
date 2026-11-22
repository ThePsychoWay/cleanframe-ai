"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useId, useRef, useState, type DragEvent } from "react";
import { Icon } from "@/components/ui/Icon";
import { useToast } from "@/components/Toast";
import { ApiError } from "@/lib/api";
import { SUPPORTED_RESOLUTIONS_TEXT, formatContainer, formatFrameRate, formatSize, resolutionLabel } from "@/lib/resolution";
import { OWNERSHIP_NOTICE, PRIVACY_NOTICE } from "@/lib/site";
import {
  AnalysisTimeoutError,
  UploadCancelledError,
  putPartWithXhr,
  uploadVideo,
  waitForAnalysis,
  type UploadProgress,
} from "@/lib/upload";
import {
  ACCEPT_ATTRIBUTE,
  aspectRatioLabel,
  formatBytes,
  formatDuration,
  getMaxUploadBytes,
  readVideoMetadata,
  validateVideoFile,
  type BrowserVideoMetadata,
} from "@/lib/video";
import { projectsService } from "@/services/projects";
import type { Project } from "@/types/project";

type MetaState =
  | { status: "reading" }
  | { status: "ready"; data: BrowserVideoMetadata }
  | { status: "unavailable" };

/** Where the selected file is in its journey. Every state is something that really happened. */
type Phase =
  | { name: "selected"; resumable: boolean }
  | { name: "uploading"; progress: UploadProgress }
  | { name: "analyzing" }
  | { name: "ready"; project: Project }
  | { name: "rejected"; project: Project };

interface UploadDropzoneProps {
  /** Called whenever the server-side list of projects may have changed. */
  onProjectsChanged?: () => void;
}

export function UploadDropzone({ onProjectsChanged }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const readToken = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [meta, setMeta] = useState<MetaState | null>(null);
  const [phase, setPhase] = useState<Phase>({ name: "selected", resumable: false });
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const toast = useToast();
  const router = useRouter();
  const inputId = useId();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // A local preview of the chosen file, so the person can see what they picked before and after it is uploaded.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const maxLabel = formatBytes(getMaxUploadBytes());
  const busy = phase.name === "uploading" || phase.name === "analyzing";

  // Leaving the page stops sending; the upload stays resumable on the server.
  useEffect(() => () => controller.current?.abort(), []);

  const handleFile = (candidate: File) => {
    if (busy) return;
    const result = validateVideoFile(candidate);
    if (!result.ok) {
      setError(result.message);
      setFile(null);
      setMeta(null);
      return;
    }

    setError(null);
    setFile(candidate);
    setPhase({ name: "selected", resumable: false });
    setMeta({ status: "reading" });

    const token = ++readToken.current;
    readVideoMetadata(candidate)
      .then((data) => {
        if (token === readToken.current) setMeta({ status: "ready", data });
      })
      .catch(() => {
        if (token === readToken.current) setMeta({ status: "unavailable" });
      });
  };

  const reset = () => {
    controller.current?.abort();
    readToken.current += 1;
    setFile(null);
    setMeta(null);
    setError(null);
    setPhase({ name: "selected", resumable: false });
    if (inputRef.current) inputRef.current.value = "";
  };

  const startUpload = async () => {
    if (!file || busy) return;
    const abort = new AbortController();
    controller.current = abort;
    setError(null);
    setPhase({ name: "uploading", progress: { uploadedBytes: 0, totalBytes: file.size, partsDone: 0, totalParts: 1 } });

    let created: Project | null = null;
    try {
      const uploaded = await uploadVideo(file, {
        api: projectsService,
        putPart: putPartWithXhr,
        signal: abort.signal,
        onProjectCreated: (project) => {
          created = project;
          onProjectsChanged?.();
        },
        onProgress: (progress) => setPhase({ name: "uploading", progress }),
      });

      setPhase({ name: "analyzing" });
      onProjectsChanged?.();
      const analysed = await waitForAnalysis(uploaded, {
        fetchProject: projectsService.get,
        signal: abort.signal,
      });
      onProjectsChanged?.();
      // Fully automatic: once the server has looked for overlays on its own (and started the render, or said why it didn't),
      // the person is taken to the page that follows it. Nothing to mark, nothing to confirm.
      if (analysed.status !== "FAILED" && analysed.automation) {
        router.push(`/process/${analysed.id}`);
        return;
      }
      setPhase(analysed.status === "FAILED" ? { name: "rejected", project: analysed } : { name: "ready", project: analysed });
      if (analysed.status !== "FAILED") toast({ kind: "success", message: "Video uploaded and checked." });
    } catch (caught) {
      onProjectsChanged?.();
      if (caught instanceof UploadCancelledError) {
        if (abort.signal.aborted && controller.current === abort) {
          setError("Upload cancelled. Resume to pick up where it left off, or choose another file.");
          setPhase({ name: "selected", resumable: created !== null });
        }
        return;
      }
      if (caught instanceof AnalysisTimeoutError) {
        setError(`${caught.message} Your project is listed below.`);
        setPhase({ name: "selected", resumable: false });
        return;
      }
      setError(caught instanceof ApiError || caught instanceof Error ? caught.message : "The upload failed. Please try again.");
      setPhase({ name: "selected", resumable: created !== null });
    } finally {
      if (controller.current === abort) controller.current = null;
    }
  };

  const onDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setDragging(false);
    const files = event.dataTransfer.files;
    if (files.length === 0) return;
    if (files.length > 1) {
      setError("Drop one video at a time.");
      return;
    }
    handleFile(files[0]);
  };

  const onDragLeave = (event: DragEvent<HTMLElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
  };

  const server = phase.name === "ready" || phase.name === "rejected" ? phase.project : null;

  return (
    <section aria-labelledby="upload-heading" className="mx-auto w-full max-w-3xl">
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        className="peer sr-only"
        disabled={busy}
        onChange={(e) => {
          const picked = e.target.files?.[0];
          if (picked) handleFile(picked);
          e.target.value = "";
        }}
      />

      {file === null ? (
        <label
          htmlFor={inputId}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={`group flex min-h-[22rem] cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed px-6 py-14 text-center transition-colors duration-200 peer-focus-visible:border-mask ${
            dragging ? "border-mask bg-mask/10" : "border-line bg-panel hover:border-[#3f3f43] hover:bg-[#18181a]"
          }`}
        >
          <span
            className={`grid size-16 place-items-center rounded-2xl border transition-all duration-200 ${
              dragging ? "scale-110 border-mask bg-mask/15 text-mask" : "border-line bg-raised text-dim group-hover:text-paper"
            }`}
          >
            <Icon name="upload" size={28} />
          </span>
          <h1 id="upload-heading" className="heading mt-6 text-[clamp(1.5rem,3.6vw,2rem)]">
            Drop your video here
          </h1>
          <p className="mt-2 text-dim">or browse files</p>
          <span className="btn btn-primary mt-7 !h-12 !px-6">Choose video</span>
          <p className="mt-6 text-sm text-dim">MP4, MOV, M4V, WebM, MKV or AVI</p>
          <p className="mt-1 text-sm text-dim">{SUPPORTED_RESOLUTIONS_TEXT}</p>
          <p className="mt-1 text-xs text-faint">Up to {maxLabel} per file</p>
        </label>
      ) : (
        <div className="surface animate-fade-up overflow-hidden">
          <div className="grid gap-6 p-5 sm:grid-cols-[minmax(0,13rem)_minmax(0,1fr)] sm:p-6">
            <Preview url={previewUrl} meta={meta} />

            <div className="min-w-0">
              <h1 id="upload-heading" className="heading text-xl">
                {headingFor(phase)}
              </h1>

              <dl className="tnum mt-5 grid gap-x-6 gap-y-4 sm:grid-cols-2">
                <Detail label="File name" value={file.name} truncate wide />
                <Detail label="Size" value={formatBytes(file.size)} />
                {/* Once the server has looked, show what it found; a rejected file has nothing to show. */}
                {phase.name === "ready" && server ? (
                  <ServerDetails project={server} />
                ) : phase.name === "rejected" ? null : (
                  <BrowserDetails meta={meta} />
                )}
              </dl>

              {(phase.name === "uploading" || phase.name === "analyzing") && <ProgressBlock phase={phase} />}

              {phase.name === "selected" && meta?.status === "unavailable" && (
                <p className="mt-5 text-sm text-dim">
                  This browser couldn&apos;t read the video&apos;s details. The file may use a codec the browser
                  can&apos;t play, or it may be damaged. The full check happens on the server after upload.
                </p>
              )}
              {phase.name === "selected" && (
                <p className="mt-5 text-sm text-faint">
                  Frame rate, codec and length are read on the server after upload, and the processing-time estimate
                  appears once the video has been analysed.
                </p>
              )}

              {phase.name === "rejected" && (
                <p role="alert" className="mt-5 rounded-xl border border-line bg-raised px-4 py-3 text-sm text-paper">
                  {phase.project.error ?? "We couldn't use this video."}
                </p>
              )}

              <div className="mt-6 flex flex-wrap items-center gap-3">
                {phase.name === "selected" && (
                  <button type="button" className="btn btn-primary" onClick={startUpload}>
                    {phase.resumable ? "Resume upload" : "Upload video"}
                  </button>
                )}
                {phase.name === "uploading" && (
                  <button type="button" className="btn btn-ghost" onClick={() => controller.current?.abort()}>
                    Cancel upload
                  </button>
                )}
                {phase.name === "ready" && (
                  <>
                    <Link href={`/editor/${phase.project.id}`} className="btn btn-primary">
                      Continue to selection
                      <Icon name="chevron" size={16} className="-rotate-90" />
                    </Link>
                    <button type="button" className="btn btn-ghost" onClick={reset}>
                      Upload another video
                    </button>
                  </>
                )}
                {(phase.name === "selected" || phase.name === "rejected") && (
                  <button type="button" className="btn btn-ghost" onClick={reset}>
                    Choose another file
                  </button>
                )}
              </div>
              {phase.name === "ready" && (
                <p className="mt-3 text-sm text-faint">
                  Your video is uploaded and checked. Next you mark what to remove. It&apos;s also saved under My projects.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {error && (
        <p role="alert" className="animate-fade-up mt-4 rounded-xl border border-line bg-raised px-4 py-3 text-sm text-paper">
          {error}
        </p>
      )}

      <p className="mt-6 text-sm text-dim">{OWNERSHIP_NOTICE}</p>
      <p className="mt-1.5 max-w-2xl text-sm text-faint">{PRIVACY_NOTICE}</p>
    </section>
  );
}

/** What the person picked, played from their own computer. Nothing here has been uploaded yet. */
function Preview({ url, meta }: { url: string | null; meta: MetaState | null }) {
  const [broken, setBroken] = useState(false);
  const ratio = meta?.status === "ready" && meta.data.height > 0 ? meta.data.width / meta.data.height : 9 / 16;
  return (
    <div
      className="relative mx-auto w-full max-w-[13rem] overflow-hidden rounded-xl border border-line bg-black sm:mx-0"
      style={{ aspectRatio: `${Math.min(Math.max(ratio, 0.4), 2.2)}` }}
    >
      {url && !broken ? (
        <video src={url} muted playsInline controls preload="metadata" aria-label="Preview of the chosen video" onError={() => setBroken(true)} className="h-full w-full object-contain" />
      ) : (
        <div className="grid h-full w-full place-items-center p-4 text-center text-xs text-faint">
          <span>
            <Icon name="film" size={22} className="mx-auto mb-2" />
            This browser can&apos;t preview the file.
          </span>
        </div>
      )}
    </div>
  );
}

function headingFor(phase: Phase): string {
  switch (phase.name) {
    case "uploading":
      return "Uploading";
    case "analyzing":
      return "Checking your video";
    case "ready":
      return "Video ready";
    case "rejected":
      return "We couldn’t use this video";
    default:
      return phase.resumable ? "Upload cancelled" : "Video selected";
  }
}

function ProgressBlock({ phase }: { phase: Extract<Phase, { name: "uploading" | "analyzing" }> }) {
  if (phase.name === "analyzing") {
    return (
      <p role="status" className="mt-5 text-sm text-dim">
        Upload complete. Reading the file on the server and preparing the editor preview.
      </p>
    );
  }
  const { uploadedBytes, totalBytes } = phase.progress;
  const percent = totalBytes > 0 ? Math.min(100, Math.floor((uploadedBytes / totalBytes) * 100)) : 0;
  return (
    <div className="mt-5">
      <div
        role="progressbar"
        aria-label="Upload progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className="h-2 w-full overflow-hidden rounded-full bg-raised"
      >
        <div className="h-full rounded-full bg-mask transition-[width] duration-200" style={{ width: `${percent}%` }} />
      </div>
      <p className="tnum mt-2 text-sm text-dim">
        {percent}% · {formatBytes(uploadedBytes)} of {formatBytes(totalBytes)}
      </p>
    </div>
  );
}

function BrowserDetails({ meta }: { meta: MetaState | null }) {
  const reading = meta?.status === "reading";
  const ready = meta?.status === "ready" ? meta.data : null;
  const fallback = reading ? "Reading…" : "Checked after upload";
  return (
    <>
      <Detail
        label="Duration"
        value={ready?.durationSeconds != null ? formatDuration(ready.durationSeconds) : fallback}
      />
      <Detail label="Resolution" value={ready ? formatSize(ready.width, ready.height) : fallback} />
      <Detail label="Aspect ratio" value={ready ? aspectRatioLabel(ready.width, ready.height) : fallback} />
    </>
  );
}

/** What the server found in the file itself, which is the source of truth. */
function ServerDetails({ project }: { project: Project }) {
  const { width, height, duration, fps, codec, audio_codec: audio } = project;
  return (
    <>
      <Detail label="Duration" value={duration != null ? formatDuration(duration) : "Unknown"} />
      <Detail label="Resolution" value={width && height ? formatSize(width, height) : "Unknown"} />
      <Detail label="Quality" value={width && height ? resolutionLabel(width, height) : "Unknown"} />
      <Detail label="Aspect ratio" value={width && height ? aspectRatioLabel(width, height) : "Unknown"} />
      <Detail label="Frame rate" value={fps != null ? formatFrameRate(fps) : "Unknown"} />
      <Detail label="Format" value={formatContainer(project.container, project.original_filename)} />
      <Detail label="Video codec" value={codec ? codec.toUpperCase() : "Unknown"} />
      <Detail label="Audio" value={audio ? audio.toUpperCase() : "No audio track"} />
    </>
  );
}

function Detail({ label, value, truncate = false, wide = false }: { label: string; value: string; truncate?: boolean; wide?: boolean }) {
  return (
    <div className={`min-w-0 ${wide ? "sm:col-span-2" : ""}`}>
      <dt className="text-xs uppercase tracking-wide text-faint">{label}</dt>
      <dd className={`mt-0.5 font-medium ${truncate ? "truncate" : ""}`} title={truncate ? value : undefined}>
        {value}
      </dd>
    </div>
  );
}
