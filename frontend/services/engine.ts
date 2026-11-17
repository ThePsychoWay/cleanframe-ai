import { apiFetch } from "@/lib/api";
import type { EngineInfo } from "@/types/render";

export const engineService = {
  /** What will rebuild the picture: its name, whether it is ready, and whether frames leave the server. */
  get: () => apiFetch<EngineInfo>("/api/engine"),
};
