import { apiFetch } from "@/lib/api";
import type { MaskDocument, MaskResponse, MediaInfo } from "@/types/mask";
import type { CompletedPart, PartUrl, Project, ProjectUpload } from "@/types/project";
import type { RenderRequest, RenderStatusResponse } from "@/types/render";
import type { ExportInfo, PreviewInfo } from "@/types/result";
import type { TrackingResponse } from "@/types/tracking";

export const projectsService = {
  list: () => apiFetch<Project[]>("/api/projects"),
  get: (id: string) => apiFetch<Project>(`/api/projects/${id}`),
  /** Creates the project, or resumes an unfinished upload of the same file. */
  create: (file: Pick<File, "name" | "size">) =>
    apiFetch<ProjectUpload>("/api/projects", { method: "POST", json: { filename: file.name, size_bytes: file.size } }),
  signParts: async (id: string, partNumbers: number[]) =>
    (await apiFetch<{ urls: PartUrl[] }>(`/api/projects/${id}/upload/urls`, { method: "POST", json: { part_numbers: partNumbers } }))
      .urls,
  complete: (id: string, parts: CompletedPart[]) =>
    apiFetch<Project>(`/api/projects/${id}/upload/complete`, { method: "POST", json: { parts } }),
  /** Signed links to the editor's preview video and poster. They expire; ask again when they do. */
  media: (id: string) => apiFetch<MediaInfo>(`/api/projects/${id}/media`),
  getMask: (id: string) => apiFetch<MaskResponse>(`/api/projects/${id}/mask`),
  saveMask: (id: string, document: MaskDocument) =>
    apiFetch<MaskResponse>(`/api/projects/${id}/mask`, { method: "PUT", json: document }),
  /** Where tracking stands, with the tracked motion once it has finished. */
  tracking: (id: string) => apiFetch<TrackingResponse>(`/api/projects/${id}/tracking`),
  /** Starts following the saved marks through the video. Returns at once; poll `tracking` for progress. */
  startTracking: (id: string) => apiFetch<TrackingResponse>(`/api/projects/${id}/track`, { method: "POST" }),
  /** The latest render job for the project, or `{ job: null }` if none was ever started. */
  render: (id: string) => apiFetch<RenderStatusResponse>(`/api/projects/${id}/render`),
  /** Queues the final render of the saved marks. Returns at once; follow it with `renderEventsUrl` or `render`. */
  startRender: (id: string, request: RenderRequest) =>
    apiFetch<RenderStatusResponse>(`/api/projects/${id}/render`, { method: "POST", json: request }),
  cancelRender: (id: string) => apiFetch<RenderStatusResponse>(`/api/projects/${id}/render/cancel`, { method: "POST" }),
  /** Lets a finished video's marks be changed again. The finished file is kept until a new render replaces it. */
  reopen: (id: string) => apiFetch<RenderStatusResponse>(`/api/projects/${id}/reopen`, { method: "POST" }),
  /** Signed links to the restored video and to the preview copy of the original, for the compare view. */
  preview: (id: string) => apiFetch<PreviewInfo>(`/api/projects/${id}/preview`),
  /** A short-lived link that downloads the restored video. */
  exportResult: (id: string) => apiFetch<ExportInfo>(`/api/projects/${id}/export`, { method: "POST" }),
  analyze: (id: string) => apiFetch<Project>(`/api/projects/${id}/analyze`, { method: "POST" }),
  remove: (id: string) => apiFetch<void>(`/api/projects/${id}`, { method: "DELETE" }),
};
