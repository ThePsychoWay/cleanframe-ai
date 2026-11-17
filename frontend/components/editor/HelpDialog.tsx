"use client";

import { useEffect, useRef } from "react";
import { Icon } from "@/components/ui/Icon";

const SHORTCUTS: ReadonlyArray<[string, string]> = [
  ["R / B / E / P / V", "Rectangle, Brush, Eraser, Polygon, Adjust"],
  ["Ctrl+Z / Ctrl+Shift+Z", "Undo / redo"],
  ["Space", "Play or pause"],
  ["← / →", "One frame back / forward (Shift: ten)"],
  ["Delete", "Remove the selected region"],
  ["[ / ]", "Smaller / larger brush"],
  ["Ctrl+S", "Save your selection"],
];

/** A short guide, as a native dialog so focus is trapped and Escape closes it without any extra code. */
export function HelpDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      aria-labelledby="help-heading"
      className="m-auto w-[min(32rem,calc(100vw-2rem))] rounded-2xl border border-line bg-panel p-0 text-paper shadow-[var(--shadow-soft)] backdrop:bg-black/70"
    >
      <div className="p-5 sm:p-6">
        <div className="flex items-start justify-between gap-4">
          <h2 id="help-heading" className="heading text-xl">
            How to remove a watermark
          </h2>
          <button type="button" onClick={onClose} aria-label="Close help" className="btn btn-quiet btn-icon -mr-2 -mt-2">
            <Icon name="close" size={18} />
          </button>
        </div>
        <ol className="mt-4 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-dim marker:text-faint">
          <li>
            Draw over the logo or text with <span className="text-paper">Rectangle</span> or <span className="text-paper">Brush</span>, or try{" "}
            <span className="text-paper">Automatic detection</span> and confirm what it finds.
          </li>
          <li>Drag the selection to move it, or its handles to resize it. Add more selections for more marks.</li>
          <li>If the mark moves, scrub to another frame and move the selection there, or use Follow movement.</li>
          <li>
            Press <span className="text-paper">Remove Watermark</span>, then compare the result with your original and download it.
          </li>
        </ol>
        <h3 className="mt-5 text-sm font-semibold">Keyboard</h3>
        <dl className="mt-2 divide-y divide-line text-sm">
          {SHORTCUTS.map(([keys, what]) => (
            <div key={keys} className="flex justify-between gap-4 py-1.5">
              <dt className="tnum shrink-0 text-paper">{keys}</dt>
              <dd className="text-right text-dim">{what}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-5 text-xs text-faint">
          You&apos;re marking areas on a lower-resolution preview. Saved selections apply to your original video, which is never changed: the result is a
          new file.
        </p>
      </div>
    </dialog>
  );
}
