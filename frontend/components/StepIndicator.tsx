import { Icon } from "@/components/ui/Icon";

export const STEPS = ["Upload", "Select", "Restore", "Export"] as const;
export type StepNumber = 1 | 2 | 3 | 4;

/** Where the person is in Upload → Select → Restore → Export. Steps before the current one show as done. */
export function StepIndicator({ current, className = "" }: { current: StepNumber; className?: string }) {
  return (
    <nav aria-label="Progress" className={className}>
      <ol className="flex items-center gap-1.5 sm:gap-2">
        {STEPS.map((label, index) => {
          const number = (index + 1) as StepNumber;
          const state = number < current ? "done" : number === current ? "current" : "todo";
          return (
            <li key={label} className="flex items-center gap-1.5 sm:gap-2" aria-current={state === "current" ? "step" : undefined}>
              <span
                className={`grid size-6 place-items-center rounded-full text-xs font-semibold transition-colors duration-200 ${
                  state === "current"
                    ? "bg-mask text-ink"
                    : state === "done"
                      ? "bg-raised text-mask ring-1 ring-mask/40"
                      : "bg-raised text-faint ring-1 ring-line"
                }`}
              >
                {state === "done" ? <Icon name="check" size={13} strokeWidth={2.5} /> : number}
              </span>
              <span
                className={`text-sm font-medium ${state === "current" ? "text-paper" : "hidden text-dim sm:inline"} ${state === "current" ? "" : ""}`}
              >
                <span className="sr-only">{state === "done" ? "Done: " : state === "current" ? "Current step: " : "Next: "}</span>
                {label}
              </span>
              {number < STEPS.length && <span aria-hidden="true" className={`h-px w-4 sm:w-8 ${state === "done" ? "bg-mask/50" : "bg-line"}`} />}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
