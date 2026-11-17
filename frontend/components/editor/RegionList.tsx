"use client";

import { useState } from "react";
import type { MaskDocument, RegionType } from "@/types/mask";

const TYPE_LABEL: Record<RegionType, string> = { rectangle: "Rectangle", polygon: "Polygon", brush: "Brush" };

interface RegionListProps {
  doc: MaskDocument;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  /** Names and regions can't change while tracking runs. */
  locked?: boolean;
}

export function RegionList({ doc, selectedId, onSelect, onRename, onDelete, locked = false }: RegionListProps) {
  return (
    <section aria-labelledby="regions-heading" className="surface p-4">
      <h2 id="regions-heading" className="heading text-lg">
        Selected regions
      </h2>

      {doc.regions.length === 0 ? (
        <p className="mt-3 text-sm text-dim">
          Nothing marked yet. Pick a tool and draw over what you want removed. You can mark several areas; they are all
          removed in one job.
        </p>
      ) : (
        <ul className="mt-3 space-y-2">
          {doc.regions.map((region) => {
            const selected = region.id === selectedId;
            return (
              <li key={region.id} className={`rounded-xl border p-2.5 transition-colors duration-150 ${selected ? "border-mask/60 bg-mask/10" : "border-line hover:border-[#3f3f43]"}`}>
                <button type="button" onClick={() => onSelect(region.id)} aria-pressed={selected} className="flex w-full items-baseline justify-between gap-3 text-left">
                  <span className="truncate font-medium" title={region.name}>
                    {region.name}
                  </span>
                  <span className="shrink-0 text-xs text-dim">
                    {TYPE_LABEL[region.type]} · {region.keyframes.length} keyframe{region.keyframes.length === 1 ? "" : "s"}
                  </span>
                </button>

                {selected && (
                  <div className="mt-2.5 space-y-2.5 border-t border-line pt-2.5">
                    <p className="text-xs font-medium uppercase tracking-wide text-faint">Region settings</p>
                    <NameField key={region.id} name={region.name} disabled={locked} onCommit={(name) => onRename(region.id, name)} />
                    <button type="button" className="btn btn-ghost !h-10 !px-4 !text-sm" disabled={locked} onClick={() => onDelete(region.id)} aria-keyshortcuts="Delete">
                      Clear region
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

/** Edits locally and commits once, on Enter or when focus leaves, so typing doesn't fill the undo history. */
function NameField({ name, onCommit, disabled }: { name: string; onCommit: (name: string) => void; disabled: boolean }) {
  const [value, setValue] = useState(name);
  const commit = () => {
    const trimmed = value.trim();
    if (trimmed === "") setValue(name); // a region always has a name
    else if (trimmed !== name) onCommit(trimmed);
  };
  return (
    <label className="block text-sm text-dim">
      <span className="sr-only">Region name</span>
      <input
        type="text"
        value={value}
        maxLength={60}
        disabled={disabled}
        onChange={(e) => setValue(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
          if (e.key === "Escape") setValue(name);
        }}
        className="field !h-10 w-full"
      />
    </label>
  );
}
