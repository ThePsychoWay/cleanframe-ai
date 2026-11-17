"use client";

import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { VideoComparisonSlider } from "./VideoComparisonSlider";

/** Where the two sample clips live. See docs/ARCHITECTURE.md ("Landing sample") for how they were made. */
export const SAMPLE = {
  before: "/samples/before.mp4",
  after: "/samples/after.mp4",
  width: 540,
  height: 960,
} as const;

/**
 * The hero demo: two real video files that play in step behind a draggable divider. "Original" is a generated clip with a
 * semi-transparent logo added; "Restored" is that same clip after it went through CleanFrame's own overlay removal.
 */
export function SampleCompare({ className = "" }: { className?: string }) {
  const beforeRef = useRef<HTMLVideoElement>(null);
  const afterRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState(false);

  // Keep the two clips on the same frame: the second follows the first and is nudged back when it drifts.
  useEffect(() => {
    const a = beforeRef.current;
    const b = afterRef.current;
    if (!a || !b) return;
    const drift = () => {
      if (Math.abs(a.currentTime - b.currentTime) > 0.12) b.currentTime = a.currentTime;
    };
    const onPlay = () => {
      setPlaying(true);
      void b.play().catch(() => undefined);
    };
    const onPause = () => {
      setPlaying(false);
      b.pause();
      b.currentTime = a.currentTime;
    };
    a.addEventListener("timeupdate", drift);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!reduce) void a.play().catch(() => undefined);
    return () => {
      a.removeEventListener("timeupdate", drift);
      a.removeEventListener("play", onPlay);
      a.removeEventListener("pause", onPause);
    };
  }, []);

  const toggle = () => {
    const a = beforeRef.current;
    if (!a) return;
    if (a.paused) void a.play().catch(() => undefined);
    else a.pause();
  };

  const clip = (ref: React.RefObject<HTMLVideoElement | null>, src: string, label: string) => (
    <video
      ref={ref}
      src={src}
      muted
      loop
      playsInline
      preload="auto"
      aria-label={label}
      onError={() => setFailed(true)}
      className="absolute inset-0 h-full w-full object-cover"
    />
  );

  if (failed) {
    return (
      <p role="status" className={`surface p-6 text-sm text-dim ${className}`}>
        The sample clips couldn&apos;t be loaded. The tool itself isn&apos;t affected.
      </p>
    );
  }

  return (
    <div className={className}>
      <VideoComparisonSlider
        aspect={`${SAMPLE.width} / ${SAMPLE.height}`}
        autoSweep
        label="Before and after sample"
        before={clip(beforeRef, SAMPLE.before, "Sample clip with a logo on it")}
        after={clip(afterRef, SAMPLE.after, "The same clip after the logo was removed")}
      />
      <div className="mt-3 flex items-start justify-between gap-3">
        <p className="max-w-[17rem] text-xs leading-relaxed text-faint">
          A generated sample clip with a semi-transparent logo, restored by CleanFrame. Drag the divider to compare.
        </p>
        <button
          type="button"
          onClick={toggle}
          aria-label={playing ? "Pause the sample" : "Play the sample"}
          className="btn btn-quiet btn-icon !size-10 shrink-0"
        >
          <Icon name={playing ? "pause" : "play"} size={16} />
        </button>
      </div>
    </div>
  );
}
