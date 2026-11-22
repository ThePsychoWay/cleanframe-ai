/** Mirrors backend/app/schemas/render.py and backend/app/schemas/engine.py. */

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type StageState = "done" | "active" | "pending" | "skipped" | "failed" | "stopped";

export interface Stage {
  key: string;
  label: string;
  state: StageState;
  detail: string | null;
}

export interface RenderSettings {
  quality: "fast" | "balanced" | "high";
  container: "mp4" | "mov" | "webm" | "mkv";
  /** Shorter side of the picture. null keeps the original size. The server only ever scales down. */
  output_height: 720 | 1080 | 1440 | 2160 | null;
  keep_audio: boolean;
}

export interface RenderRequest extends RenderSettings {
  /** Needed when the engine sends frames to an outside service. */
  acknowledge_third_party: boolean;
}

export interface RenderResult {
  width: number;
  height: number;
  frames: number;
  fps: number;
  container: string;
  /** What the picture was encoded with ("h264", "hevc", "vp9", "av1") and whether the GPU encoder did it. Older results may lack them. */
  codec?: string;
  encoder?: string;
  hardware_encoder?: boolean;
  source_width?: number;
  source_height?: number;
  size_bytes: number;
  audio: "none" | "removed" | "copied" | "converted";
  audio_codec: string | null;
  tracking_used: boolean;
  render_seconds: number;
  /** True when the marks were found by the detector and nobody drew them. */
  automatic?: boolean;
  /** The result looked at after the render: see backend/app/services/quality.py. */
  quality?: QualityReport;
}

export interface QualityRegion {
  name: string;
  frames_masked: number;
  of_frames?: number;
  residual: number | null;
  flicker: number | null;
  verdict: "clean" | "check" | "not_masked";
  problems?: string[];
}

export interface QualityReport {
  checked: boolean;
  reason?: string;
  regions: QualityRegion[];
  seconds?: number;
}

export interface Job {
  id: string;
  project_id: string;
  status: JobStatus;
  stage: string;
  stages: Stage[];
  frames_done: number;
  frames_total: number;
  /** Only while frames are being rebuilt (and 100 when complete); null where there is no honest fraction. */
  percent: number | null;
  speed_fps: number | null;
  eta_seconds: number | null;
  jobs_ahead: number | null;
  error: string | null;
  warnings: string[];
  engine: { provider: string; development_only: boolean };
  settings: Partial<RenderSettings>;
  result: RenderResult | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface RenderStatusResponse {
  job: Job | null;
}

export interface EngineInfo {
  provider: string;
  name: string;
  available: boolean;
  development_only: boolean;
  sends_frames_to_third_party: boolean;
  device: { kind: string; name: string; note: string } | null;
  reason: string | null;
  expectation: string;
}
