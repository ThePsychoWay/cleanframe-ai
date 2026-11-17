import { API_URL } from "@/lib/api";
import { parseJobEvent } from "@/lib/render";
import type { Job } from "@/types/render";

export interface JobStream {
  close: () => void;
}

/**
 * Follows a project's render over server-sent events. `onJob` gets every state the server reports (the first is the
 * current one, so reconnecting catches up). `onBroken` is called once if the stream can't be kept; the caller then
 * falls back to asking on a timer.
 */
export function followJob(projectId: string, onJob: (job: Job) => void, onBroken: () => void): JobStream {
  if (typeof EventSource === "undefined") {
    queueMicrotask(onBroken);
    return { close: () => undefined };
  }
  const source = new EventSource(`${API_URL}/api/projects/${projectId}/render/events`, { withCredentials: true });
  let failures = 0;
  let broken = false;
  source.addEventListener("job", (event) => {
    failures = 0;
    const job = parseJobEvent((event as MessageEvent<string>).data);
    if (job) onJob(job);
  });
  source.onerror = () => {
    failures += 1;
    // The browser reconnects by itself after a dropped connection; give up after repeated failures or a refusal.
    if (!broken && (failures >= 3 || source.readyState === EventSource.CLOSED)) {
      broken = true;
      source.close();
      onBroken();
    }
  };
  return { close: () => source.close() };
}
