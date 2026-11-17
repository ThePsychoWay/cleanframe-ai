"use client";

import { Icon, type IconName } from "@/components/ui/Icon";
import { Tip } from "@/components/ui/Tip";
import type { Tool } from "@/lib/maskRender";
import { MAX_RADIUS, MIN_RADIUS } from "@/lib/masks";

interface ToolDef {
  id: Tool;
  label: string;
  key: string;
  icon: IconName;
  hint: string;
}

/** The four tools people need, then two for adjusting what they drew. */
const MAIN: readonly ToolDef[] = [
  { id: "rectangle", label: "Rectangle", key: "R", icon: "rectangle", hint: "Drag across the area you want to remove." },
  { id: "brush", label: "Brush", key: "B", icon: "brush", hint: "Paint over the area. Strokes are added to the selected brush region, or start a new one." },
  { id: "eraser", label: "Eraser", key: "E", icon: "eraser", hint: "Paint over part of the selected region to take it out of the mask." },
];
const MORE: readonly ToolDef[] = [
  { id: "select", label: "Adjust", key: "V", icon: "cursor", hint: "Click a region to select it. Drag to move it; drag a handle to reshape it." },
  { id: "polygon", label: "Polygon", key: "P", icon: "polygon", hint: "Click to place points. Click the first point, press Enter, or double-click to finish. Backspace removes the last point." },
];
const ALL = [...MAIN, ...MORE];

export function toolHint(tool: Tool): string {
  return ALL.find((t) => t.id === tool)?.hint ?? "";
}

interface SelectionToolbarProps {
  tool: Tool;
  onTool: (tool: Tool) => void;
  brushRadius: number;
  onBrushRadius: (radius: number) => void;
  /** Tools are off while tracking or processing runs. */
  locked?: boolean;
  detecting: boolean;
  onDetect: () => void;
}

export function SelectionToolbar({ tool, onTool, brushRadius, onBrushRadius, locked = false, detecting, onDetect }: SelectionToolbarProps) {
  const showSize = tool === "brush" || tool === "eraser";

  const toolButton = (t: ToolDef) => {
    const active = tool === t.id;
    return (
      <Tip key={t.id} label={`${t.label} (${t.key}): ${t.hint}`} side="right">
        <button
          type="button"
          aria-label={t.label}
          aria-pressed={active}
          aria-keyshortcuts={t.key}
          disabled={locked}
          onClick={() => onTool(t.id)}
          className={`flex h-11 min-w-11 shrink-0 items-center gap-2.5 rounded-xl border px-3 text-sm font-medium transition-all duration-150 active:scale-[0.97] lg:w-full ${
            active ? "border-mask bg-mask/15 text-paper shadow-[0_0_0_1px_var(--color-mask)_inset]" : "border-transparent text-dim hover:bg-raised hover:text-paper"
          } disabled:cursor-not-allowed disabled:opacity-50`}
        >
          <Icon name={t.icon} size={20} className={active ? "text-mask" : ""} />
          <span>{t.label}</span>
          <kbd className="ml-auto hidden rounded border border-line px-1.5 text-[0.6875rem] font-normal text-faint lg:inline">{t.key}</kbd>
        </button>
      </Tip>
    );
  };

  return (
    <div className="space-y-3">
      <div role="toolbar" aria-label="Selection tools" aria-orientation="vertical" className="flex gap-1 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
        {MAIN.map(toolButton)}
        <Tip label="Automatic detection: look for a still logo or text over the moving picture. You confirm before anything is selected." side="right">
          <button
            type="button"
            aria-label="Automatic detection"
            onClick={onDetect}
            disabled={locked || detecting}
            aria-busy={detecting}
            className="flex h-11 min-w-11 shrink-0 items-center gap-2.5 rounded-xl border border-transparent px-3 text-sm font-medium text-dim transition-all duration-150 hover:bg-raised hover:text-paper active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 lg:w-full"
          >
            <Icon name={detecting ? "spinner" : "wand"} size={20} className={detecting ? "animate-spin-slow text-mask" : ""} />
            <span>Automatic detection</span>
          </button>
        </Tip>
        <span aria-hidden="true" className="mx-1 my-2 w-px shrink-0 self-stretch bg-line lg:mx-2 lg:my-1 lg:h-px lg:w-auto lg:self-auto" />
        {MORE.map(toolButton)}
      </div>

      {showSize && (
        <label className="block rounded-xl border border-line bg-ink px-3 py-2.5 text-sm text-dim">
          <span className="flex items-center justify-between">
            <span>Size</span>
            <span className="tnum text-xs text-faint">[ and ] to change</span>
          </span>
          <input
            type="range"
            min={MIN_RADIUS * 1000}
            max={MAX_RADIUS * 1000 * 0.4}
            step={1}
            value={Math.round(brushRadius * 1000)}
            onChange={(e) => onBrushRadius(Number(e.target.value) / 1000)}
            aria-label="Brush size"
            className="range mt-2 w-full"
          />
        </label>
      )}
    </div>
  );
}
