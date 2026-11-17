"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Icon } from "@/components/ui/Icon";

export type ToastKind = "success" | "error" | "info";

export interface ToastInput {
  kind?: ToastKind;
  message: string;
  /** Milliseconds before it goes away. Errors stay longer. 0 keeps it until dismissed. */
  duration?: number;
}

interface ToastItem extends Required<Omit<ToastInput, "duration">> {
  id: number;
  duration: number;
}

const ToastContext = createContext<(input: ToastInput) => void>(() => undefined);

/** Shows a short message that goes away on its own. Safe to call anywhere: without a provider it does nothing. */
export function useToast() {
  return useContext(ToastContext);
}

const ICON = { success: "check", error: "alert", info: "info" } as const;
const TONE = { success: "text-mask", error: "text-danger", info: "text-dim" } as const;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const next = useRef(1);

  const dismiss = useCallback((id: number) => setItems((current) => current.filter((t) => t.id !== id)), []);
  const push = useCallback((input: ToastInput) => {
    const kind = input.kind ?? "info";
    const id = next.current++;
    const duration = input.duration ?? (kind === "error" ? 8000 : 4500);
    // Identical messages replace each other instead of piling up.
    setItems((current) => [...current.filter((t) => t.message !== input.message).slice(-3), { id, kind, message: input.message, duration }]);
  }, []);

  const value = useMemo(() => push, [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        role="region"
        aria-label="Notifications"
        aria-live="polite"
        className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex flex-col items-center gap-2 px-4 pb-[calc(env(safe-area-inset-bottom)+5.5rem)] sm:items-end sm:px-6 sm:pb-6"
      >
        {items.map((item) => (
          <ToastCard key={item.id} item={item} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}

function ToastCard({ item, onDismiss }: { item: ToastItem; onDismiss: (id: number) => void }) {
  useEffect(() => {
    if (item.duration === 0) return;
    const timer = window.setTimeout(() => onDismiss(item.id), item.duration);
    return () => window.clearTimeout(timer);
  }, [item.duration, item.id, onDismiss]);

  return (
    <div
      role={item.kind === "error" ? "alert" : "status"}
      className="animate-fade-up pointer-events-auto flex w-full max-w-md items-start gap-3 rounded-xl border border-line bg-raised px-4 py-3 text-sm shadow-[var(--shadow-soft)]"
    >
      <Icon name={ICON[item.kind]} size={18} className={`mt-0.5 ${TONE[item.kind]}`} />
      <p className="min-w-0 flex-1 break-words text-paper">{item.message}</p>
      <button
        type="button"
        onClick={() => onDismiss(item.id)}
        aria-label="Dismiss notification"
        className="-m-2 grid size-9 shrink-0 place-items-center rounded-lg text-dim transition-colors hover:text-paper"
      >
        <Icon name="close" size={16} />
      </button>
    </div>
  );
}
