"use client";

import { Icon } from "@/components/ui/Icon";
import type { Rect } from "@/types/mask";

/** The suggested boxes, drawn over the picture. They are only highlights until the person accepts them. */
export function SuggestionBoxes({ boxes }: { boxes: Rect[] }) {
  return (
    <>
      {boxes.map((b, i) => (
        <div
          key={`${b.x}-${b.y}-${i}`}
          aria-hidden="true"
          className="animate-fade-up pointer-events-none absolute rounded-sm border-2 border-dashed border-mask bg-mask/20"
          style={{ left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` }}
        />
      ))}
    </>
  );
}

export type DetectionState =
  | { status: "idle" }
  | { status: "working" }
  | { status: "found"; boxes: Rect[] }
  | { status: "none" }
  | { status: "unknown"; reason: string }
  | { status: "unavailable"; reason: string };

interface DetectionBannerProps {
  state: DetectionState;
  onUse: () => void;
  onAdjust: () => void;
  onDismiss: () => void;
}

/** What automatic detection found, and the two ways to go on. Nothing is selected until the person chooses. */
export function DetectionBanner({ state, onUse, onAdjust, onDismiss }: DetectionBannerProps) {
  if (state.status === "idle") return null;

  if (state.status === "working") {
    return (
      <p role="status" className="animate-fade-up flex items-center gap-2.5 rounded-xl border border-line bg-panel px-4 py-3 text-sm text-dim">
        <Icon name="spinner" size={16} className="animate-spin-slow text-mask" />
        Finding unwanted overlays...
      </p>
    );
  }

  if (state.status === "found") {
    return (
      <div role="status" className="animate-fade-up rounded-xl border border-mask/40 bg-mask/10 p-4">
        <p className="flex items-center gap-2 font-medium">
          <Icon name="wand" size={18} className="text-mask" />
          Possible overlay detected
        </p>
        <p className="mt-1 text-sm text-dim">
          {state.boxes.length === 1 ? "One area looks" : `${state.boxes.length} areas look`} like a logo or text that stays still while the picture moves. {state.boxes.length === 1 ? "It is" : "They are"}{" "}
          highlighted in the preview. Nothing is selected yet.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <button type="button" className="btn btn-primary !h-10 !px-4 !text-sm" onClick={onUse}>
            Use selection
          </button>
          <button type="button" className="btn btn-ghost !h-10 !px-4 !text-sm" onClick={onAdjust}>
            Adjust manually
          </button>
        </div>
      </div>
    );
  }

  const message =
    state.status === "none"
      ? "No overlay found. Nothing in this video stays perfectly still while the picture moves. Try selecting the area yourself with the Rectangle or Brush tool."
      : `${state.reason} Try selecting the area yourself with the Rectangle or Brush tool.`;
  return (
    <div role="status" className="animate-fade-up rounded-xl border border-line bg-panel p-4">
      <p className="text-sm text-dim">{message}</p>
      <button type="button" className="btn btn-ghost mt-3 !h-10 !px-4 !text-sm" onClick={onDismiss}>
        Got it
      </button>
    </div>
  );
}
