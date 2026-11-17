import type { ReactNode } from "react";

interface TipProps {
  /** What the control does. Icon-only controls must also carry this as their aria-label. */
  label: string;
  children: ReactNode;
  side?: "top" | "bottom" | "right";
  className?: string;
}

const SIDE = {
  top: "bottom-full left-1/2 mb-2 -translate-x-1/2",
  bottom: "top-full left-1/2 mt-2 -translate-x-1/2",
  right: "left-full top-1/2 ml-2 -translate-y-1/2",
} as const;

/** A small tooltip shown on hover and on keyboard focus. Touch devices rely on the control's own label. */
export function Tip({ label, children, side = "top", className = "" }: TipProps) {
  return (
    <span className={`group/tip relative inline-flex ${className}`}>
      {children}
      <span
        role="tooltip"
        className={`pointer-events-none absolute z-40 hidden w-max max-w-[16rem] rounded-lg border border-line bg-raised px-2.5 py-1.5 text-xs font-medium text-paper opacity-0 shadow-[var(--shadow-soft)] transition-opacity duration-150 [@media(hover:hover)]:block group-focus-within/tip:opacity-100 group-hover/tip:opacity-100 ${SIDE[side]}`}
      >
        {label}
      </span>
    </span>
  );
}
