"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { USE_CASES, type OverlayKey } from "@/lib/overlays";
import { CompareFrame } from "./CompareFrame";

export function UseCases() {
  const [active, setActive] = useState<OverlayKey>("logo");
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const current = USE_CASES.find((c) => c.key === active) ?? USE_CASES[0];

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = USE_CASES.findIndex((c) => c.key === active);
    let next = index;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") next = (index + 1) % USE_CASES.length;
    else if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = (index - 1 + USE_CASES.length) % USE_CASES.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = USE_CASES.length - 1;
    else return;
    event.preventDefault();
    const key = USE_CASES[next].key;
    setActive(key);
    tabRefs.current[key]?.focus();
  };

  return (
    <section id="use-cases" aria-labelledby="use-heading" className="scroll-mt-16 border-t border-line">
      <div className="mx-auto max-w-6xl px-5 py-24">
        <h2 id="use-heading" className="heading max-w-2xl text-[clamp(1.75rem,3.6vw,2.75rem)]">
          Built for the overlays you put there yourself
        </h2>
        <p className="mt-4 max-w-xl text-dim">
          Pick a case, then drag the divider to compare the original with the restored frame.
        </p>

        <div className="mt-12 grid gap-10 lg:grid-cols-[16rem_minmax(0,1fr)] lg:gap-14">
          <div
            role="tablist"
            aria-label="Overlay types"
            aria-orientation="vertical"
            onKeyDown={onKeyDown}
            className="-mx-5 flex gap-1 overflow-x-auto px-5 lg:mx-0 lg:flex-col lg:overflow-visible lg:px-0"
          >
            {USE_CASES.map((c) => {
              const selected = c.key === active;
              return (
                <button
                  key={c.key}
                  ref={(el) => {
                    tabRefs.current[c.key] = el;
                  }}
                  role="tab"
                  id={`tab-${c.key}`}
                  aria-selected={selected}
                  aria-controls="use-case-panel"
                  tabIndex={selected ? 0 : -1}
                  onClick={() => setActive(c.key)}
                  className={`shrink-0 rounded-lg px-4 py-3 text-left text-[0.9375rem] font-medium transition-colors lg:border-l-2 lg:rounded-none lg:rounded-r-lg ${
                    selected
                      ? "bg-raised text-paper lg:border-mask"
                      : "text-dim hover:text-paper lg:border-line"
                  }`}
                >
                  {c.label}
                </button>
              );
            })}
          </div>

          <div id="use-case-panel" role="tabpanel" aria-labelledby={`tab-${active}`}>
            <CompareFrame aspect="landscape" overlay={active} />
            <p className="mt-5 max-w-xl leading-relaxed text-dim">{current.description}</p>
            <p className="mt-3 text-xs text-faint">Illustration of the workflow, not a real result.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
