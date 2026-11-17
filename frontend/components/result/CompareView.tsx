"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ZOOM_LEVELS,
  clampPan,
  dividerFromKey,
  dividerFromPointer,
  needsResync,
  nextZoom,
  panForZoom,
  type Offset,
  type Zoom,
} from "@/lib/compare";
import { Icon } from "@/components/ui/Icon";
import { Tip } from "@/components/ui/Tip";
import { frameToTime, timeToFrame } from "@/lib/masks";
import { formatDuration } from "@/lib/video";
import type { PreviewInfo } from "@/types/result";

type Mode = "compare" | "original" | "result";

interface CompareViewProps {
  preview: PreviewInfo;
  /** Called when a video can't load (usually because its signed link expired). */
  onVideoError: () => void;
  /** Goes up each time fresh links were fetched, so a video that failed is loaded again even if the URL text is the same. */
  linksVersion: number;
}

const MODES: Array<{ key: Mode; label: string }> = [
  { key: "compare", label: "Compare" },
  { key: "original", label: "Original" },
  { key: "result", label: "Restored" },
];

export function CompareView({ preview, onVideoError, linksVersion }: CompareViewProps) {
  const resultRef = useRef<HTMLVideoElement>(null); // the master: it has the sound and sets the time
  const originalRef = useRef<HTMLVideoElement>(null); // follows the master (a silent, lower-resolution copy)
  const frameRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<Mode>("compare");
  const [position, setPosition] = useState(50);
  const [zoom, setZoom] = useState<Zoom>(1);
  const [pan, setPan] = useState<Offset>({ x: 0, y: 0 });
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const shellRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const { fps, frame_count: frameCount } = preview;
  const hasSound = preview.audio === "copied" || preview.audio === "converted";
  const lastFrame = Math.max(0, frameCount - 1);

  const box = useCallback(() => {
    const rect = frameRef.current?.getBoundingClientRect();
    return { width: rect?.width ?? 0, height: rect?.height ?? 0 };
  }, []);

  // --- keeping the two videos on the same frame ----------------------------------------------

  const follow = useCallback(
    (force = false) => {
      const master = resultRef.current;
      const other = originalRef.current;
      if (!master || !other) return;
      if (force || needsResync(master.currentTime, other.currentTime, fps, !master.paused)) other.currentTime = master.currentTime;
    },
    [fps],
  );

  useEffect(() => {
    const master = resultRef.current;
    const other = originalRef.current;
    if (!master || !other) return;
    let handle = 0;
    const tick = () => {
      follow();
      setFrame(timeToFrame(master.currentTime, fps, frameCount));
      if (!master.paused && !master.ended) handle = schedule();
    };
    const schedule = (): number => {
      if ("requestVideoFrameCallback" in master) return master.requestVideoFrameCallback(tick);
      return window.requestAnimationFrame(tick);
    };
    const cancel = () => {
      if ("cancelVideoFrameCallback" in master) master.cancelVideoFrameCallback(handle);
      else window.cancelAnimationFrame(handle);
    };
    const onPlay = () => {
      setPlaying(true);
      void other.play().catch(() => undefined);
      cancel();
      handle = schedule();
    };
    const onPause = () => {
      setPlaying(false);
      other.pause();
      cancel();
      follow(true);
      setFrame(timeToFrame(master.currentTime, fps, frameCount));
    };
    const onSeeked = () => {
      follow(true);
      setFrame(timeToFrame(master.currentTime, fps, frameCount));
    };
    const onRate = () => {
      other.playbackRate = master.playbackRate;
    };
    const onReady = () => setReady(true);
    master.addEventListener("play", onPlay);
    master.addEventListener("pause", onPause);
    master.addEventListener("ended", onPause);
    master.addEventListener("seeked", onSeeked);
    master.addEventListener("ratechange", onRate);
    master.addEventListener("loadeddata", onReady);
    if (master.readyState >= 2) setReady(true);
    return () => {
      cancel();
      master.removeEventListener("play", onPlay);
      master.removeEventListener("pause", onPause);
      master.removeEventListener("ended", onPause);
      master.removeEventListener("seeked", onSeeked);
      master.removeEventListener("ratechange", onRate);
      master.removeEventListener("loadeddata", onReady);
    };
  }, [follow, fps, frameCount]);

  // Fresh links: load both videos again and go back to where they were.
  const firstLinks = useRef(true);
  useEffect(() => {
    if (firstLinks.current) {
      firstLinks.current = false;
      return;
    }
    const master = resultRef.current;
    const other = originalRef.current;
    if (!master || !other) return;
    const at = master.currentTime;
    for (const video of [master, other]) {
      const restore = () => {
        video.currentTime = at;
      };
      video.addEventListener("loadedmetadata", restore, { once: true });
      video.load();
    }
  }, [linksVersion, preview.result_url, preview.original_url]);

  // --- controls -------------------------------------------------------------------------------

  const seek = useCallback(
    (target: number) => {
      const master = resultRef.current;
      if (!master) return;
      const clamped = Math.min(lastFrame, Math.max(0, Math.round(target)));
      master.pause();
      master.currentTime = frameToTime(clamped, fps);
      setFrame(clamped);
    },
    [fps, lastFrame],
  );

  const toggle = useCallback(async () => {
    const master = resultRef.current;
    if (!master) return;
    if (!master.paused) {
      master.pause();
      return;
    }
    if (master.ended || frame >= lastFrame) master.currentTime = 0;
    try {
      await master.play();
    } catch {
      /* the browser refused to start playback; the button stays on "Play" */
    }
  }, [frame, lastFrame]);

  useEffect(() => {
    const master = resultRef.current;
    if (master) master.volume = volume;
  }, [volume]);

  const toggleFullscreen = () => {
    const shell = shellRef.current;
    if (!shell) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void shell.requestFullscreen?.().catch(() => undefined);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.closest("input, select, textarea") || target.getAttribute("role") === "slider")) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === " " && !(target instanceof HTMLButtonElement)) {
        event.preventDefault();
        void toggle();
      } else if (event.key === "ArrowLeft") seek(frame - 1);
      else if (event.key === "ArrowRight") seek(frame + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frame, seek, toggle]);

  const changeZoom = (direction: 1 | -1) => {
    const next = nextZoom(zoom, direction);
    setPan((current) => panForZoom(current, zoom, next, box()));
    setZoom(next);
  };

  // --- dragging the divider, and dragging the picture when zoomed -----------------------------

  const dragDivider = (event: React.PointerEvent<HTMLDivElement>) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const move = (e: PointerEvent) => {
      const rect = frameRef.current?.getBoundingClientRect();
      if (rect) setPosition(dividerFromPointer(e.clientX, rect.left, rect.width));
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
    move(event.nativeEvent);
    setMode("compare");
  };

  const dragPicture = (event: React.PointerEvent<HTMLDivElement>) => {
    if (zoom <= 1) return;
    const start = { x: event.clientX, y: event.clientY, pan };
    const move = (e: PointerEvent) =>
      setPan(clampPan({ x: start.pan.x + e.clientX - start.x, y: start.pan.y + e.clientY - start.y }, zoom, box()));
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  };

  const shown = mode === "original" ? 100 : mode === "result" ? 0 : position;
  const transform = { transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "center" };
  const seconds = frame / fps;
  const ratio = preview.width / preview.height;
  const soundOn = hasSound && !muted && volume > 0;

  return (
    <section aria-label="Before and after" className="space-y-3">
      <div ref={shellRef} className="flex items-center justify-center [&:fullscreen]:bg-black [&:fullscreen]:[--frame-h:100vh]">
        <div
          ref={frameRef}
          onPointerDown={dragPicture}
          className={`relative w-full select-none overflow-hidden rounded-2xl border border-line bg-black shadow-[var(--shadow-soft)] ${zoom > 1 ? "cursor-grab active:cursor-grabbing" : ""}`}
          style={{
            aspectRatio: `${preview.width} / ${preview.height}`,
            maxWidth: `min(100%, calc(var(--frame-h, 68vh) * ${ratio}))`,
            touchAction: zoom > 1 ? "none" : "pan-y",
          }}
        >
          <div className="absolute inset-0">
            <div className="h-full w-full" style={transform}>
              <video
                ref={resultRef}
                src={preview.result_url}
                playsInline
                preload="auto"
                muted={muted || !hasSound}
                onError={onVideoError}
                className="h-full w-full object-fill"
                aria-label="Restored video"
              />
            </div>
          </div>
          {/* The original sits above and is cut off at the divider. The cut is made on this fixed layer, so it stays
              put on screen while the picture inside is zoomed and moved. */}
          <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - shown}% 0 0)` }} aria-hidden={shown === 0}>
            <div className="h-full w-full" style={transform}>
              <video ref={originalRef} src={preview.original_url} playsInline muted preload="auto" onError={onVideoError} className="h-full w-full object-fill" aria-label="Original video" />
            </div>
          </div>

          {mode === "compare" && (
            <>
              <span className="pointer-events-none absolute left-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-xs font-medium text-paper backdrop-blur">Original</span>
              <span className="pointer-events-none absolute right-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-xs font-medium text-paper backdrop-blur">Restored</span>
              <div
                role="slider"
                tabIndex={0}
                aria-label="Comparison divider"
                aria-orientation="horizontal"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(position)}
                aria-valuetext={`${Math.round(position)}% original, ${Math.round(100 - position)}% restored`}
                onPointerDown={dragDivider}
                onKeyDown={(e) => {
                  const next = dividerFromKey(position, e.key, e.shiftKey);
                  if (next !== null) {
                    e.preventDefault();
                    setPosition(next);
                  }
                }}
                className="group absolute inset-y-0 z-10 w-10 -translate-x-1/2 cursor-ew-resize touch-none outline-none"
                style={{ left: `${position}%` }}
              >
                <span className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-paper/90 shadow" />
                <span className="absolute left-1/2 top-1/2 grid size-11 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-paper text-ink shadow-[0_2px_14px_rgba(0,0,0,0.5)] transition-transform duration-150 group-focus-visible:ring-2 group-focus-visible:ring-mask group-focus-visible:ring-offset-2 group-focus-visible:ring-offset-ink group-active:scale-95">
                  <Icon name="back" size={14} className="-mr-0.5" />
                  <Icon name="back" size={14} className="-ml-0.5 rotate-180" />
                </span>
              </div>
            </>
          )}
          {mode === "original" && <span className="pointer-events-none absolute left-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-xs font-medium text-paper backdrop-blur">Original</span>}
          {mode === "result" && <span className="pointer-events-none absolute right-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-xs font-medium text-paper backdrop-blur">Restored</span>}
          {!ready && (
            <p role="status" className="absolute inset-0 flex items-center justify-center bg-black/60 text-sm text-dim">
              Loading the videos…
            </p>
          )}
        </div>
      </div>

      <div className="surface p-3 sm:p-4">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <button type="button" className="btn btn-primary btn-icon !rounded-full" onClick={() => void toggle()} aria-keyshortcuts="Space" aria-label={playing ? "Pause" : "Play"}>
            <Icon name={playing ? "pause" : "play"} size={18} />
          </button>
          <Tip label="Back one frame (←)">
            <button type="button" className="btn btn-quiet btn-icon" onClick={() => seek(frame - 1)} aria-label="Back one frame">
              <Icon name="frameBack" size={18} />
            </button>
          </Tip>
          <Tip label="Forward one frame (→)">
            <button type="button" className="btn btn-quiet btn-icon" onClick={() => seek(frame + 1)} aria-label="Forward one frame">
              <Icon name="frameForward" size={18} />
            </button>
          </Tip>
          <p className="tnum text-sm">
            <span className="text-paper">
              {formatDuration(seconds)}.{String(Math.floor((seconds % 1) * 100)).padStart(2, "0")}
            </span>
            <span className="ml-2 hidden text-xs text-faint sm:inline">
              frame {frame + 1} of {frameCount}
            </span>
          </p>

          <div className="ml-auto flex items-center gap-1">
            <div role="group" aria-label="What to show" className="seg mr-1 !grid-flow-col">
              {MODES.map(({ key, label }) => (
                <button
                  key={key}
                  type="button"
                  aria-pressed={mode === key}
                  onClick={() => setMode(key)}
                  className={`min-h-9 rounded-lg px-3 text-sm font-medium transition-colors duration-150 ${mode === key ? "bg-raised text-paper shadow-[0_0_0_1px_#3a3a3d_inset]" : "text-dim hover:text-paper"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div role="group" aria-label="Zoom" className="flex items-center">
              <button type="button" className="btn btn-quiet btn-icon" onClick={() => changeZoom(-1)} disabled={zoom === ZOOM_LEVELS[0]} aria-label="Zoom out">
                <Icon name="zoomOut" size={18} />
              </button>
              <span className="tnum w-8 text-center text-xs text-dim" aria-live="polite">
                {zoom}×
              </span>
              <button type="button" className="btn btn-quiet btn-icon" onClick={() => changeZoom(1)} disabled={zoom === ZOOM_LEVELS[ZOOM_LEVELS.length - 1]} aria-label="Zoom in">
                <Icon name="zoomIn" size={18} />
              </button>
            </div>
            <Tip label={hasSound ? (soundOn ? "Mute" : "Unmute") : "This video has no audio"}>
              <button
                type="button"
                className="btn btn-quiet btn-icon"
                disabled={!hasSound}
                onClick={() => setMuted((m) => !m)}
                aria-pressed={hasSound ? muted : undefined}
                aria-label={hasSound ? "Mute" : "No audio"}
              >
                <Icon name={soundOn ? "volume" : "volumeOff"} size={18} />
              </button>
            </Tip>
            {hasSound && (
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={muted ? 0 : volume}
                onChange={(e) => {
                  const next = Number(e.target.value);
                  setVolume(next);
                  setMuted(next === 0);
                }}
                aria-label="Volume"
                className="range hidden w-20 sm:block"
              />
            )}
            <Tip label="Fullscreen">
              <button type="button" className="btn btn-quiet btn-icon" onClick={toggleFullscreen} aria-label="Fullscreen">
                <Icon name="fullscreen" size={18} />
              </button>
            </Tip>
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(1, lastFrame)}
          step={1}
          value={frame}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Video position"
          aria-valuetext={`Frame ${frame + 1} of ${frameCount}`}
          className="range mt-3 block h-6 w-full cursor-pointer"
        />
        <p className="mt-2 text-xs text-faint">
          The original here is the lower-resolution preview copy, so it looks softer than the restored video, which is shown at its real size.
          {hasSound ? "" : " This video has no audio."}
        </p>
      </div>
    </section>
  );
}
