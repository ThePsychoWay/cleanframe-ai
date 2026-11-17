/** Mirrors the backend's project schemas (backend/app/schemas/project.py). */

export type ProjectStatus =
  | "UPLOADING"
  | "UPLOADED"
  | "ANALYZING"
  | "MASKING"
  | "TRACKING"
  | "PROCESSING"
  | "ENCODING"
  | "COMPLETED"
  | "FAILED"
  | "EXPIRED";

export interface Project {
  id: string;
  name: string;
  original_filename: string;
  size_bytes: number;
  status: ProjectStatus;
  error: string | null;
  container: string | null;
  codec: string | null;
  audio_codec: string | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  duration: number | null;
  bitrate: number | null;
  frame_count: number | null;
  /** What the automatic pipeline found and did, from the server. Absent for projects made before it existed. */
  automation?: Automation | null;
  created_at: string;
  updated_at: string;
  expires_at: string;
}

export interface UploadedPart {
  part_number: number;
  etag: string;
  size: number;
}

export interface UploadState {
  part_size: number;
  total_parts: number;
  uploaded_parts: UploadedPart[];
}

export interface ProjectUpload {
  project: Project;
  upload: UploadState;
}

export interface PartUrl {
  part_number: number;
  url: string;
}

export interface CompletedPart {
  part_number: number;
  etag: string;
}

/** One thing the detector found. `decision` says whether it was removed or deliberately left, and `reason` says why. */
export interface DetectedOverlay {
  id: number;
  kind: "text" | "logo";
  decision: "remove" | "left";
  reason: string;
  basis: "motion" | "appearance" | "copy";
  confidence: number;
  first_frame: number | null;
  last_frame: number | null;
  frames_present: number;
  notes: string[];
}

export type AutomationState = "rendering" | "ready" | "nothing_found" | "detection_failed" | "engine_unavailable";

export interface Automation {
  state?: AutomationState;
  state_reason?: string;
  found: number;
  removing: number;
  overlays: DetectedOverlay[];
  notes: string[];
  /** Seconds spent on each detection step, measured on the server. */
  timings?: Record<string, number>;
  seconds?: number;
}
