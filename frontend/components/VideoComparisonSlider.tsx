"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Icon } from "@/components/ui/Icon";

interface VideoComparisonSliderProps {
  /** Shown on the left of the divider. */
  before: ReactNode;
  /** Shown on the right of the divider. */
  after: ReactNode;
  /** CSS aspect-ratio value, e.g. "9 / 16". */
  aspect: string;
  beforeLabel?: string;
  afterLabel?: string;
  /** Sweep the divider once on load, then hand control to the person. */
  autoSweep?: boolean;
  className?: string;
  /** Accessible name for the divider. */
  label?: string;
}

const SWEEP_FROM = 96;
const SWEEP_TO = 50;

/**
 * A before/after reveal. The divider is a real range input laid over the frame, so mouse, touch, pen and keyboard
 * (arrow keys, Home, End) all work without custom pointer code.
 */
export function VideoComparisonSlider({
  before,
  after,
  aspect,
  beforeLabel = "Original",
  afterLabel = "Restored",
  autoSweep = false,
  className = "",
  label = "Comparison divider",
}: VideoComparisonSliderProps) {
  const [position, setPosition] = useState(autoSweep ? SWEEP_FROM : 50);
  const interacted = useRef(false);

  useEffect(() => {
    if (!autoSweep) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setPosition(SWEEP_TO);
      return;
    }
    const delay = 600;
    const duration = 1500;
    const startedAt = performance.now() + delay;
    let frame = 0;
    const tick = (now: number) => {
      if (interacted.current) return;
      const t = Math.min(Math.max((now - startedAt) / duration, 0), 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setPosition(SWEEP_FROM + (SWEEP_TO - SWEEP_FROM) * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [autoSweep]);

  const take = () => {
    interacted.current = true;
  };

  return (
    <div className={`relative w-full select-none overflow-hidden rounded-2xl border border-line bg-black shadow-[var(--shadow-soft)] ${className}`} style={{ aspectRatio: aspect, containerType: "inline-size" }}>
      <div className="absolute inset-0">{after}</div>
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
        {before}
      </div>

      <span className="pointer-events-none absolute left-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-xs font-medium text-paper backdrop-blur">{beforeLabel}</span>
      <span className="pointer-events-none absolute right-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-xs font-medium text-paper backdrop-blur">{afterLabel}</span>

      <input
        type="range"
        min={0}
        max={100}
        step={0.5}
        value={position}
        onChange={(e) => {
          take();
          setPosition(Number(e.target.value));
        }}
        onPointerDown={take}
        onKeyDown={take}
        aria-label={`${label}. Left of the divider shows the ${beforeLabel.toLowerCase()}, right shows the ${afterLabel.toLowerCase()}.`}
        aria-valuetext={`${Math.round(position)}% ${beforeLabel.toLowerCase()}`}
        className="peer absolute inset-0 z-10 h-full w-full cursor-ew-resize opacity-0"
        style={{ touchAction: "pan-y" }}
      />
      <div className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-paper/90" style={{ left: `${position}%` }} aria-hidden="true" />
      <span
        className="pointer-events-none absolute top-1/2 grid size-11 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-paper text-ink shadow-[0_2px_14px_rgba(0,0,0,0.5)] transition-transform duration-150 peer-active:scale-95 peer-focus-visible:ring-2 peer-focus-visible:ring-mask peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-ink"
        style={{ left: `${position}%` }}
        aria-hidden="true"
      >
        <Icon name="back" size={14} className="-mr-0.5" />
        <Icon name="back" size={14} className="-ml-0.5 rotate-180" />
      </span>
    </div>
  );
}
