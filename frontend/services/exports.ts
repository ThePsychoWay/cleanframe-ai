import { apiFetch } from "@/lib/api";
import type { ExportJob, ExportList, ExportOptions, ExportRequest } from "@/types/exports";
import type { ExportInfo } from "@/types/result";

export const exportsService = {
  /** Sizes and formats the finished video can really be exported in. */
  options: (id: string) => apiFetch<ExportOptions>(`/api/projects/${id}/exports/options`),
  list: (id: string) => apiFetch<ExportList>(`/api/projects/${id}/exports`),
  /** Queues one more copy of the finished video (no AI re-run). Returns at once; poll `list`. */
  create: (id: string, request: ExportRequest) => apiFetch<ExportJob>(`/api/projects/${id}/exports`, { method: "POST", json: request }),
  /** A short-lived link that downloads a finished export. */
  link: (id: string, exportId: string) => apiFetch<ExportInfo>(`/api/projects/${id}/exports/${exportId}/link`, { method: "POST" }),
};
