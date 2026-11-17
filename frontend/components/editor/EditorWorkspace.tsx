"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { StepIndicator } from "@/components/StepIndicator";
import { Tip } from "@/components/ui/Tip";
import { useToast } from "@/components/Toast";
import { Logo } from "@/components/Logo";
import { ApiError } from "@/lib/api";
import { canRedo, canUndo, commit, createHistory, redo, undo, type History } from "@/lib/history";
import { DetectionUnavailableError, detectInVideo } from "@/lib/detectVideo";
import type { Tool } from "@/lib/maskRender";
import {
  EMPTY_DOCUMENT,
  MAX_REGIONS,
  addRegion,
  createRegion,
  findRegion,
  frameToTime,
  problemWithDocument,
  removeKeyframe,
  editShapeAt,
  removeRegion,
  renameRegion,
  shapeAt,
  timeToFrame,
} from "@/lib/masks";
import { DEFAULT_SETTINGS, buildRequest, effectiveSettings, heightOptions, isActive, type ProcessingMode } from "@/lib/render";
import { lostAreas, resolverFor } from "@/lib/tracking";
import { OWNERSHIP_NOTICE } from "@/lib/site";
import { useRenderJob } from "@/hooks/useRenderJob";
import { engineService } from "@/services/engine";
import { projectsService } from "@/services/projects";
import type { MaskDocument, MediaInfo } from "@/types/mask";
import type { Project, ProjectStatus } from "@/types/project";
import type { EngineInfo, Job, RenderSettings } from "@/types/render";
import type { TrackingResponse } from "@/types/tracking";
import { DetectionBanner, SuggestionBoxes, type DetectionState } from "./DetectionSuggestions";
import { HelpDialog } from "./HelpDialog";
import { MaskCanvas } from "./MaskCanvas";
import { ProcessingProgress } from "./ProcessingProgress";
import { RegionList } from "./RegionList";
import { RemovalSettings } from "./RemovalSettings";
import { SelectionToolbar, toolHint } from "./SelectionToolbar";
import { TrackingPanel } from "./TrackingPanel";
import { VideoTimeline } from "./VideoTimeline";

type Load =
  | { status: "loading" }
  | { status: "blocked"; title: string; message: string }
  | {
      status: "ready";
      project: Project;
      media: MediaInfo;
      tracking: TrackingResponse;
      job: Job | null;
      engine: EngineInfo | null;
      engineProblem: string | null;
    };

type SaveState = { status: "idle" } | { status: "saving" } | { status: "saved" } | { status: "error"; message: string };

const OPENABLE: ProjectStatus[] = ["MASKING", "TRACKING", "PROCESSING", "COMPLETED"];
const DEFAULT_BRUSH_RADIUS = 0.02;
const MEDIA_RELOAD_COOLDOWN_MS = 20_000;
const TRACKING_POLL_MS = 1000;
const NO_TRACKING: TrackingResponse = { status: "none", frames_done: 0, frames_total: 0, error: null, started_at: null, data: null };

export function EditorWorkspace() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [history, setHistory] = useState<History<MaskDocument>>(() => createHistory(EMPTY_DOCUMENT));
  const [savedDoc, setSavedDoc] = useState<MaskDocument>(EMPTY_DOCUMENT);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const project = await projectsService.get(id);
        // Projects being tracked or rendered, and finished ones, open too: the editor shows the progress or the result
        // and keeps the marks locked.
        if (!OPENABLE.includes(project.status)) {
          if (!cancelled) setLoad(blockedFor(project));
          return;
        }
        const [media, mask, tracking, render, engine] = await Promise.all([
          projectsService.media(id),
          projectsService.getMask(id),
          projectsService.tracking(id),
          projectsService.render(id),
          engineService.get().then(
            (info) => ({ info, problem: null as string | null }),
            () => ({ info: null, problem: "We couldn't check the reconstruction engine. Reload the page to try again." }),
          ),
        ]);
        if (cancelled) return;
        setHistory(createHistory(mask.mask_data));
        setSavedDoc(mask.mask_data);
        setLoad({ status: "ready", project, media, tracking, job: render.job, engine: engine.info, engineProblem: engine.problem });
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof ApiError ? error.message : "We couldn't open this project.";
        setLoad({ status: "blocked", title: "We couldn't open the editor", message });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (load.status === "loading") {
    return (
      <p role="status" className="px-5 py-24 text-center text-dim">
        Opening the editor…
      </p>
    );
  }
  if (load.status === "blocked") {
    return (
      <div className="mx-auto max-w-md px-5 py-24" role="alert">
        <h1 className="heading text-2xl">{load.title}</h1>
        <p className="mt-3 text-dim">{load.message}</p>
        <Link href="/projects" className="btn btn-primary mt-6">
          Back to projects
        </Link>
      </div>
    );
  }

  return (
    <Editor
      project={load.project}
      initialMedia={load.media}
      initialTracking={load.tracking}
      initialJob={load.job}
      engine={load.engine}
      engineProblem={load.engineProblem}
      history={history}
      setHistory={setHistory}
      savedDoc={savedDoc}
      setSavedDoc={setSavedDoc}
    />
  );
}

function blockedFor(project: Project): Load {
  if (project.status === "EXPIRED") {
    return {
      status: "blocked",
      title: "This project has expired",
      message: "Its files were deleted after the retention period. Upload the video again to start a new project.",
    };
  }
  if (project.status === "FAILED") {
    return { status: "blocked", title: "This video couldn't be used", message: project.error ?? "Please upload another copy." };
  }
  if (project.status === "UPLOADING" || project.status === "UPLOADED" || project.status === "ANALYZING") {
    return { status: "blocked", title: "Still getting ready", message: "This video is still being uploaded or checked. Try again in a moment." };
  }
  return { status: "blocked", title: "Not editable right now", message: "Masks can only be changed while a project is waiting for them." };
}

interface EditorProps {
  project: Project;
  initialMedia: MediaInfo;
  initialTracking: TrackingResponse;
  initialJob: Job | null;
  engine: EngineInfo | null;
  engineProblem: string | null;
  history: History<MaskDocument>;
  setHistory: React.Dispatch<React.SetStateAction<History<MaskDocument>>>;
  savedDoc: MaskDocument;
  setSavedDoc: (doc: MaskDocument) => void;
}

function Editor({ project, initialMedia, initialTracking, initialJob, engine, engineProblem, history, setHistory, savedDoc, setSavedDoc }: EditorProps) {
  const router = useRouter();
  const toast = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [media, setMedia] = useState(initialMedia);
  const [tool, setTool] = useState<Tool>("rectangle");
  const [rawSelectedId, setSelectedId] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [brushRadius, setBrushRadius] = useState(DEFAULT_BRUSH_RADIUS);
  const [notice, setNotice] = useState<string | null>(null);
  const [save, setSave] = useState<SaveState>({ status: "idle" });
  const [tracking, setTracking] = useState<TrackingResponse>(initialTracking);
  const [settings, setSettings] = useState<RenderSettings>(DEFAULT_SETTINGS);
  const [mode, setMode] = useState<ProcessingMode>("ai");
  const [acknowledged, setAcknowledged] = useState(false);
  const [detection, setDetection] = useState<DetectionState>({ status: "idle" });
  const [helpOpen, setHelpOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const detectAbort = useRef<AbortController | null>(null);
  const lastMediaReload = useRef(0);

  const doc = history.present;
  const lastFrame = Math.max(0, media.frame_count - 1);
  const dirty = doc !== savedDoc;
  // Undoing a region's creation must not leave a dangling selection.
  const selectedId = findRegion(doc, rawSelectedId) ? rawSelectedId : null;

  // The project's status as the server last told us. A render changes it by itself, so it is re-read when one ends.
  const [projectStatus, setProjectStatus] = useState<ProjectStatus>(project.status);
  const {
    job,
    problem: renderProblem,
    busy: renderBusy,
    start: startRender,
    cancel: cancelRender,
    reopen,
  } = useRenderJob({
    projectId: project.id,
    initial: initialJob,
    onFinished: (ended) => {
      projectsService.get(project.id).then(
        (latest) => setProjectStatus(latest.status),
        () => setNotice("The render ended, but we couldn't refresh this page. Reload to see the result."),
      );
      if (ended.status === "completed") {
        // The next step opens by itself, because the previous one succeeded.
        toast({ kind: "success", message: "Your video is ready." });
        router.push(`/result/${project.id}`);
      } else if (ended.status === "failed") {
        toast({ kind: "error", message: ended.error ?? "Processing couldn't finish. Your selection is unchanged." });
      } else {
        toast({ kind: "info", message: "Stopped. Nothing was kept from that attempt." });
      }
    },
  });
  const rendering = isActive(job);
  const finished = !rendering && projectStatus === "COMPLETED";

  // While tracking or a render runs, the saved marks are what it is following, and a finished video's marks
  // produced the result it shows: none of these can change until the person reopens the video for editing.
  const locked = tracking.status === "running" || rendering || finished;
  // Results describe the *saved* marks. Once the marks on screen differ, they would show the wrong thing.
  const resultsCurrent = tracking.status === "done" && tracking.data !== null && !dirty;
  const resolve = useMemo(() => (resultsCurrent ? resolverFor(tracking.data) : shapeAt), [resultsCurrent, tracking.data]);
  const lost = useMemo(() => (resultsCurrent ? lostAreas(tracking.data, doc) : []), [resultsCurrent, tracking.data, doc]);

  // --- the video ------------------------------------------------------------------------------

  const seek = useCallback(
    (target: number) => {
      const clamped = Math.min(lastFrame, Math.max(0, Math.round(target)));
      setFrame(clamped);
      const video = videoRef.current;
      if (video) video.currentTime = frameToTime(clamped, media.fps);
    },
    [lastFrame, media.fps],
  );

  const pause = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    setPlaying(false);
    // Land exactly on a frame so what's drawn over the picture belongs to the picture shown.
    const stopped = timeToFrame(video.currentTime, media.fps, media.frame_count);
    setFrame(stopped);
    video.currentTime = frameToTime(stopped, media.fps);
  }, [media.fps, media.frame_count]);

  const togglePlay = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    if (playing) {
      pause();
      return;
    }
    if (frame >= lastFrame) video.currentTime = 0;
    try {
      await video.play();
      setPlaying(true);
    } catch {
      setNotice("Your browser wouldn't start playback. Use the slider or the frame buttons instead.");
    }
  }, [frame, lastFrame, pause, playing]);

  useEffect(() => {
    if (!playing) return;
    const video = videoRef.current;
    if (!video) return;
    let raf = 0;
    const tick = () => {
      setFrame((current) => {
        const next = timeToFrame(video.currentTime, media.fps, media.frame_count);
        return next === current ? current : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const onEnded = () => {
      setPlaying(false);
      setFrame(lastFrame);
    };
    video.addEventListener("ended", onEnded);
    return () => {
      cancelAnimationFrame(raf);
      video.removeEventListener("ended", onEnded);
    };
  }, [playing, media.fps, media.frame_count, lastFrame]);

  // Preview links expire after an hour. If the video stops loading, fetch fresh ones, at most every 20 s.
  const onVideoError = useCallback(async () => {
    if (Date.now() - lastMediaReload.current < MEDIA_RELOAD_COOLDOWN_MS) return;
    lastMediaReload.current = Date.now();
    try {
      setMedia(await projectsService.media(project.id));
    } catch {
      setNotice("The preview video couldn't be loaded. Reload the page to try again.");
    }
  }, [project.id]);

  const toggleFullscreen = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.requestFullscreen?.().catch(() => setNotice("Your browser wouldn't open the preview in fullscreen."));
  }, []);
  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === stageRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  // --- editing --------------------------------------------------------------------------------

  useEffect(() => {
    if (tracking.status !== "running") return;
    let stopped = false;
    let failures = 0;
    const timer = window.setInterval(async () => {
      try {
        const next = await projectsService.tracking(project.id);
        if (stopped) return;
        failures = 0;
        setTracking((previous) => {
          if (previous.status === "running" && next.status === "done") toast({ kind: "success", message: "Tracking finished." });
          return next;
        });
      } catch {
        failures += 1;
        if (!stopped && failures === 5) setNotice("We lost contact with the server while tracking. Reload the page to see how it went.");
      }
    }, TRACKING_POLL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [tracking.status, project.id, toast]);

  const onChange = useCallback(
    (next: MaskDocument, selectId?: string | null) => {
      if (locked) return;
      setHistory((h) => commit(h, next));
      if (selectId !== undefined) setSelectedId(selectId);
      setSave((s) => (s.status === "idle" ? s : { status: "idle" }));
    },
    [setHistory, locked],
  );

  const onUndo = useCallback(() => setHistory((h) => undo(h)), [setHistory]);
  const onRedo = useCallback(() => setHistory((h) => redo(h)), [setHistory]);

  const deleteRegion = useCallback(
    (regionId: string) => {
      onChange(removeRegion(doc, regionId), null);
    },
    [doc, onChange],
  );

  const resetSelection = () => {
    if (doc.regions.length === 0) return;
    setDetection({ status: "idle" });
    onChange({ ...doc, regions: [] }, null); // one step in the history, so Undo brings everything back
  };

  const addKeyframeHere = () => {
    const region = findRegion(doc, selectedId);
    if (!region) return;
    // A keyframe that holds the region exactly where it is on this frame (tracked or blended).
    onChange(editShapeAt(doc, region.id, frame, (current) => current, resolve));
  };
  const removeKeyframeHere = () => {
    if (selectedId) onChange(removeKeyframe(doc, selectedId, frame));
  };

  // --- automatic detection (suggests only; the person confirms) -------------------------------

  useEffect(() => () => detectAbort.current?.abort(), []);

  const runDetect = useCallback(async () => {
    if (locked) return;
    pause();
    detectAbort.current?.abort();
    const abort = new AbortController();
    detectAbort.current = abort;
    setDetection({ status: "working" });
    try {
      const outcome = await detectInVideo(media.proxy_url, abort.signal);
      if (abort.signal.aborted) return;
      if (outcome.kind === "found") setDetection({ status: "found", boxes: outcome.boxes });
      else if (outcome.kind === "none") setDetection({ status: "none" });
      else setDetection({ status: "unknown", reason: outcome.reason });
    } catch (error) {
      if (abort.signal.aborted) return;
      setDetection({
        status: "unavailable",
        reason: error instanceof DetectionUnavailableError ? error.message : "Automatic detection couldn't run on this video.",
      });
    }
  }, [locked, media.proxy_url, pause]);

  const acceptSuggestions = (adjust: boolean) => {
    if (detection.status !== "found") return;
    let next = doc;
    let firstId: string | null = null;
    for (const rect of detection.boxes) {
      if (next.regions.length >= MAX_REGIONS) break;
      const region = createRegion(next, "rectangle", { frame, rect });
      firstId ??= region.id;
      next = addRegion(next, region);
    }
    setDetection({ status: "idle" });
    if (next === doc) return;
    onChange(next, firstId);
    if (adjust) {
      setTool("select");
      setNotice("Drag the selection to move it, or a handle to resize it, until it covers the overlay.");
    } else {
      toast({ kind: "success", message: "Selection added. Press Remove Watermark when it covers the overlay." });
    }
  };

  // --- saving ---------------------------------------------------------------------------------

  const saveMasks = useCallback(async (): Promise<boolean> => {
    if (save.status === "saving" || locked) return false;
    const toSave = doc;
    const problem = problemWithDocument(toSave, media.frame_count);
    if (problem) {
      setSave({ status: "error", message: problem });
      toast({ kind: "error", message: problem });
      return false;
    }
    setSave({ status: "saving" });
    try {
      await projectsService.saveMask(project.id, toSave);
      // Compared by reference: if more edits were made while this was saving, the page stays "unsaved".
      setSavedDoc(toSave);
      setSave({ status: "saved" });
      setTracking(NO_TRACKING); // the server drops tracking results when the marks are saved
      return true;
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Your selection couldn't be saved.";
      setSave({ status: "error", message });
      toast({ kind: "error", message });
      return false;
    }
  }, [doc, locked, media.frame_count, project.id, save.status, setSavedDoc, toast]);

  const startTracking = useCallback(async () => {
    if (locked || save.status === "saving" || doc.regions.length === 0) return;
    pause();
    if (dirty && !(await saveMasks())) return; // tracking follows the saved marks
    try {
      const started = await projectsService.startTracking(project.id);
      // Normally still running; if it already finished, fetch the result (the start response has no data).
      setTracking(started.status === "running" ? started : await projectsService.tracking(project.id));
      setNotice(null);
    } catch (error) {
      setNotice(error instanceof ApiError ? error.message : "Tracking couldn't be started.");
    }
  }, [dirty, doc.regions.length, locked, pause, project.id, save.status, saveMasks]);

  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const goBack = () => {
    if (dirty && !window.confirm("You have unsaved changes to your selection. Leave without saving?")) return;
    router.push("/projects");
  };

  // --- removing -------------------------------------------------------------------------------

  const validProblem = doc.regions.length === 0 ? "Select the area to remove first" : problemWithDocument(doc, media.frame_count);
  const engineReady = engine?.available ?? false;
  const needsConsent = Boolean(engine?.sends_frames_to_third_party) && !acknowledged;
  const startBlocked =
    validProblem ??
    (tracking.status === "running"
      ? "Wait for tracking to finish"
      : save.status === "saving"
        ? "Saving your selection"
        : !engineReady
          ? (engine?.reason ?? "The reconstruction engine isn't ready")
          : needsConsent
            ? "Confirm the notice in Removal settings first"
            : null);
  const canRemove = startBlocked === null && !renderBusy && !rendering && !finished;

  const removeWatermark = async () => {
    if (!canRemove) return;
    pause();
    if (dirty && !(await saveMasks())) return;
    const request = buildRequest(effectiveSettings(settings, mode), engine, acknowledged);
    const ok = await startRender(request);
    if (ok) {
      setProjectStatus("PROCESSING");
      setSettingsOpen(false);
    }
  };

  // --- keyboard -------------------------------------------------------------------------------

  const keys = useRef({ togglePlay, seek, frame, onUndo, onRedo, saveMasks, deleteRegion, selectedId, tool, pause, locked });
  keys.current = { togglePlay, seek, frame, onUndo, onRedo, saveMasks, deleteRegion, selectedId, tool, pause, locked };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const k = keys.current;
      const target = event.target as HTMLElement | null;
      if (target && (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable)) return;
      if (target?.closest("dialog")) return;
      const modifier = event.ctrlKey || event.metaKey;

      if (modifier && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (k.locked) return;
        if (event.shiftKey) k.onRedo();
        else k.onUndo();
      } else if (modifier && event.key.toLowerCase() === "y") {
        event.preventDefault();
        if (!k.locked) k.onRedo();
      } else if (modifier && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void k.saveMasks();
      } else if (modifier || event.altKey) {
        return;
      } else if (event.key === " " && target?.tagName !== "BUTTON") {
        event.preventDefault();
        void k.togglePlay();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        k.pause();
        k.seek(k.frame + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 10 : 1));
      } else if (k.locked) {
        return; // the remaining keys edit marks or change tools
      } else if (event.key === "Delete" && k.selectedId && k.tool !== "polygon") {
        event.preventDefault();
        k.deleteRegion(k.selectedId);
      } else if (event.key === "[" || event.key === "]") {
        setBrushRadius((r) => Math.min(0.1, Math.max(0.002, r + (event.key === "]" ? 0.004 : -0.004))));
      } else {
        const tools: Record<string, Tool> = { v: "select", r: "rectangle", p: "polygon", b: "brush", e: "eraser" };
        const next = tools[event.key.toLowerCase()];
        if (next) setTool(next);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const status = useMemo(() => {
    if (rendering) return { text: "Processing…", tone: "text-dim" };
    if (finished) return { text: "Finished. Selection is locked", tone: "text-dim" };
    if (locked) return { text: "Tracking…", tone: "text-dim" };
    if (save.status === "saving") return { text: "Saving…", tone: "text-dim" };
    if (save.status === "error") return { text: "Couldn't save", tone: "text-danger" };
    if (dirty) return { text: "Unsaved changes", tone: "text-paper" };
    if (save.status === "saved") return { text: "All changes saved", tone: "text-dim" };
    return { text: doc.regions.length === 0 ? "No selection yet" : "Saved", tone: "text-dim" };
  }, [save, dirty, locked, rendering, finished, doc.regions.length]);

  const step = rendering || finished ? 3 : 2;
  const heights = heightOptions(project.width, project.height);
  const failedJob = !rendering && job && (job.status === "failed" || job.status === "cancelled") ? job : null;

  const actionButton = finished ? (
    <Link href={`/result/${project.id}`} className="btn btn-primary !h-10 !rounded-xl !px-4 !text-sm">
      View result
    </Link>
  ) : (
    <button
      type="button"
      className="btn btn-primary !h-10 !rounded-xl !px-4 !text-sm"
      disabled={!canRemove}
      title={startBlocked ?? undefined}
      onClick={() => void removeWatermark()}
    >
      {renderBusy || rendering ? "Processing…" : "Remove Watermark"}
    </button>
  );

  return (
    <div className="flex min-h-dvh flex-col lg:h-dvh lg:overflow-hidden">
      <header className="z-20 shrink-0 border-b border-line bg-ink/90 backdrop-blur">
        <div className="flex h-14 items-center gap-1.5 px-2 sm:gap-2 sm:px-4">
          <Tip label="Back to projects" side="bottom">
            <button type="button" onClick={goBack} aria-label="Back to projects" className="btn btn-quiet btn-icon">
              <Icon name="back" size={20} />
            </button>
          </Tip>
          <span className="hidden sm:block">
            <Logo />
          </span>
          <h1 className="min-w-0 flex-1 truncate px-1 text-sm font-semibold sm:flex-none sm:max-w-[16rem] sm:px-3 sm:text-base" title={project.original_filename}>
            {project.name}
          </h1>
          <StepIndicator current={step} className="hidden flex-1 justify-center xl:flex" />
          <div className="ml-auto flex items-center gap-0.5">
            <Tip label="Undo (Ctrl+Z)" side="bottom">
              <button type="button" className="btn btn-quiet btn-icon" onClick={onUndo} disabled={locked || !canUndo(history)} aria-label="Undo" aria-keyshortcuts="Control+Z">
                <Icon name="undo" size={19} />
              </button>
            </Tip>
            <Tip label="Redo (Ctrl+Shift+Z)" side="bottom">
              <button type="button" className="btn btn-quiet btn-icon" onClick={onRedo} disabled={locked || !canRedo(history)} aria-label="Redo" aria-keyshortcuts="Control+Shift+Z">
                <Icon name="redo" size={19} />
              </button>
            </Tip>
            <Tip label="Clear every selection (you can undo this)" side="bottom">
              <button type="button" className="btn btn-quiet btn-icon sm:!w-auto sm:gap-2 sm:!px-3" onClick={resetSelection} disabled={locked || doc.regions.length === 0} aria-label="Reset selection">
                <Icon name="reset" size={19} />
                <span className="hidden text-sm sm:inline">Reset selection</span>
              </button>
            </Tip>
            <Tip label="Help and keyboard shortcuts" side="bottom">
              <button type="button" className="btn btn-quiet btn-icon" onClick={() => setHelpOpen(true)} aria-label="Help">
                <Icon name="help" size={19} />
              </button>
            </Tip>
          </div>
          <p role="status" className={`ml-1 hidden max-w-[10rem] truncate text-xs lg:block ${status.tone}`}>
            {status.text}
          </p>
          <button
            type="button"
            className="btn btn-ghost !hidden !h-10 !rounded-xl !px-3 !text-sm lg:!inline-flex"
            onClick={() => void saveMasks()}
            disabled={locked || !dirty || save.status === "saving"}
            aria-keyshortcuts="Control+S"
          >
            Save
          </button>
          <span className="hidden lg:block">{actionButton}</span>
        </div>
        <div className="border-t border-line px-3 py-2 xl:hidden">
          <StepIndicator current={step} />
        </div>
      </header>

      <main id="main" className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)] lg:grid-cols-[14.5rem_minmax(0,1fr)_21rem] lg:overflow-hidden">
        {/* Left: how to select */}
        <aside aria-label="Selection tools" className="order-2 min-w-0 border-line px-3 py-3 lg:order-none lg:overflow-y-auto lg:border-r lg:p-3">
          <SelectionToolbar
            tool={tool}
            onTool={(t) => {
              setTool(t);
              setDetection((d) => (d.status === "found" ? d : { status: "idle" }));
            }}
            brushRadius={brushRadius}
            onBrushRadius={setBrushRadius}
            locked={locked}
            detecting={detection.status === "working"}
            onDetect={() => void runDetect()}
          />
          <p className="mt-3 hidden text-xs leading-relaxed text-faint lg:block">{toolHint(tool)}</p>
        </aside>

        {/* Center: the picture, then the timeline */}
        <section aria-label="Preview" className="order-1 flex min-h-0 min-w-0 flex-col gap-3 px-3 py-3 sm:px-4 lg:order-none lg:overflow-y-auto">
          <p className="min-h-5 text-center text-sm text-dim" aria-live="polite">
            {notice ?? (doc.regions.length === 0 && !locked ? "Draw over the logo or text you want to remove." : "")}
          </p>

          <div
            ref={stageRef}
            className={`relative flex flex-1 items-center justify-center [--stage-h:58vh] lg:[--stage-h:calc(100dvh-20rem)] ${
              fullscreen ? "!bg-black p-4 [--stage-h:92vh]" : ""
            }`}
          >
            <MaskCanvas
              media={media}
              doc={doc}
              resolve={resolve}
              locked={locked}
              frame={frame}
              tool={tool}
              selectedId={selectedId}
              brushRadius={brushRadius}
              videoRef={videoRef}
              onChange={onChange}
              onSelect={setSelectedId}
              onToolChange={setTool}
              onNotice={setNotice}
              onInteract={pause}
              onVideoError={() => void onVideoError()}
            >
              {detection.status === "found" && <SuggestionBoxes boxes={detection.boxes} />}
            </MaskCanvas>
          </div>

          <DetectionBanner
            state={detection}
            onUse={() => acceptSuggestions(false)}
            onAdjust={() => acceptSuggestions(true)}
            onDismiss={() => setDetection({ status: "idle" })}
          />

          <VideoTimeline
            frame={frame}
            frameCount={media.frame_count}
            fps={media.fps}
            playing={playing}
            regions={doc.regions}
            selectedId={selectedId}
            lost={lost}
            tracked={resultsCurrent}
            locked={locked}
            onSeek={(f) => {
              pause();
              seek(f);
            }}
            onTogglePlay={() => void togglePlay()}
            onStep={(delta) => {
              pause();
              seek(frame + delta);
            }}
            onAddKeyframe={addKeyframeHere}
            onRemoveKeyframe={removeKeyframeHere}
            onFullscreen={toggleFullscreen}
          />
        </section>

        {/* Right: what was selected, and how to remove it */}
        <aside aria-label="Selection and removal settings" className="order-3 min-w-0 space-y-4 px-3 pb-28 pt-3 lg:overflow-y-auto lg:border-l lg:border-line lg:p-4 lg:pb-4">
          {rendering && job ? (
            <ProcessingProgress job={job} busy={renderBusy} onCancel={() => void cancelRender()} />
          ) : (
            <>
              {finished && (
                <section className="surface p-4" aria-label="Finished">
                  <p className="font-medium">This video has been processed.</p>
                  <p className="mt-1 text-sm text-dim">Your selection is locked so the result still matches it. Edit the selection to make a new version.</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Link href={`/result/${project.id}`} className="btn btn-primary !h-10 !px-4 !text-sm">
                      View result
                    </Link>
                    <button
                      type="button"
                      className="btn btn-ghost !h-10 !px-4 !text-sm"
                      disabled={renderBusy}
                      onClick={() => void reopen().then((ok) => ok && setProjectStatus("MASKING"))}
                    >
                      Edit selection
                    </button>
                  </div>
                </section>
              )}

              {failedJob && (
                <p role="alert" className="rounded-xl border border-line bg-raised px-4 py-3 text-sm text-paper">
                  {failedJob.status === "failed" ? `${failedJob.error ?? "Processing couldn't finish."} Your selection is unchanged.` : "Stopped. Nothing was kept from that attempt."}
                </p>
              )}
              {renderProblem && (
                <p role="alert" className="rounded-xl border border-line bg-raised px-4 py-3 text-sm text-paper">
                  {renderProblem}
                </p>
              )}

              <RegionList
                doc={doc}
                selectedId={selectedId}
                onSelect={(regionId) => {
                  setSelectedId(regionId);
                  setTool("select");
                }}
                onRename={(regionId, name) => onChange(renameRegion(doc, regionId, name))}
                onDelete={deleteRegion}
                locked={locked}
              />

              {doc.regions.length > 0 && (
                <TrackingPanel
                  tracking={tracking}
                  fps={media.fps}
                  frameCount={media.frame_count}
                  regionCount={doc.regions.length}
                  unsaved={dirty}
                  saving={save.status === "saving"}
                  lost={lost}
                  resultsCurrent={resultsCurrent}
                  blockedReason={finished ? "Choose Edit selection first" : null}
                  onTrack={() => void startTracking()}
                  onGoToFrame={(f) => {
                    pause();
                    seek(f);
                  }}
                />
              )}

              {/* On a phone the settings fold away under the picture; on a desktop they are always open. */}
              <section className="surface p-4" aria-label="Removal settings panel">
                <button
                  type="button"
                  aria-expanded={settingsOpen}
                  aria-controls="removal-settings-body"
                  onClick={() => setSettingsOpen((v) => !v)}
                  className="flex min-h-11 w-full items-center justify-between text-left lg:hidden"
                >
                  <span className="heading text-lg">Removal settings</span>
                  <Icon name="chevron" size={18} className={`text-dim transition-transform duration-200 ${settingsOpen ? "rotate-180" : ""}`} />
                </button>
                <div id="removal-settings-body" className={`${settingsOpen ? "mt-3 block" : "hidden"} lg:mt-0 lg:block`}>
                  <RemovalSettings
                    settings={settings}
                    mode={mode}
                    onSettings={setSettings}
                    onMode={setMode}
                    engine={engine}
                    engineProblem={engineProblem}
                    heights={heights}
                    hasAudio={Boolean(project.audio_codec)}
                    acknowledged={acknowledged}
                    onAcknowledged={setAcknowledged}
                    disabled={locked || renderBusy}
                  />
                </div>
              </section>
            </>
          )}

          {!rendering && !finished && startBlocked && (
            <p className="hidden text-xs text-faint lg:block">{startBlocked}.</p>
          )}
          <p className="text-xs leading-relaxed text-dim">{OWNERSHIP_NOTICE}</p>
          <p className="text-xs leading-relaxed text-faint">
            You&apos;re marking areas on a lower-resolution preview. Your selection applies to your original video, which is never changed: the result is a
            new file.
          </p>
        </aside>
      </main>

      {/* Phone: the main action stays within reach. */}
      <div className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-ink/95 px-3 pb-[calc(env(safe-area-inset-bottom)+0.75rem)] pt-3 backdrop-blur lg:hidden">
        <div className="flex items-center gap-3">
          <p className="min-w-0 flex-1 text-xs text-dim" role="status">
            {rendering ? "Processing…" : (startBlocked ?? status.text)}
          </p>
          <span className="shrink-0 [&>*]:!h-11 [&>*]:!px-5">{actionButton}</span>
        </div>
      </div>

      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}
