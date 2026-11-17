/**
 * Chunked, resumable upload straight to object storage.
 *
 * The API never sees video bytes: it hands out presigned URLs for each part, the browser PUTs the
 * parts to storage, and the API verifies the total before completing. Everything that touches the
 * network is injected, so the logic here (splitting, resuming, retrying, progress) is unit-tested
 * without a browser.
 */

import { ApiError } from "./api";
import type { CompletedPart, PartUrl, Project, ProjectStatus, ProjectUpload } from "@/types/project";

export interface UploadApi {
  create(file: Pick<File, "name" | "size">): Promise<ProjectUpload>;
  signParts(projectId: string, partNumbers: number[]): Promise<PartUrl[]>;
  complete(projectId: string, parts: CompletedPart[]): Promise<Project>;
}

export interface PutPartOptions {
  signal: AbortSignal;
  /** Bytes of this part sent so far. */
  onProgress: (loaded: number) => void;
}

/** Sends one part to its presigned URL and resolves with the ETag storage returned. */
export type PutPart = (url: string, body: Blob, options: PutPartOptions) => Promise<string>;

export interface UploadProgress {
  uploadedBytes: number;
  totalBytes: number;
  /** Parts fully stored, including ones found from an earlier attempt. */
  partsDone: number;
  totalParts: number;
}

export interface UploadOptions {
  api: UploadApi;
  putPart: PutPart;
  onProgress?: (progress: UploadProgress) => void;
  onProjectCreated?: (project: Project, resumed: boolean) => void;
  signal?: AbortSignal;
  /** Parts in flight at once. */
  concurrency?: number;
  /** Attempts per part before giving up. */
  maxAttempts?: number;
  sleep?: (ms: number) => Promise<void>;
}

export class UploadCancelledError extends Error {
  constructor() {
    super("The upload was cancelled.");
    this.name = "UploadCancelledError";
  }
}

/** A failed PUT to storage. `status` is 0 for network failures. */
export class PartUploadError extends Error {
  readonly status: number;
  readonly retryable: boolean;

  constructor(status: number, message: string, retryable = isRetryableStatus(status)) {
    super(message);
    this.name = "PartUploadError";
    this.status = status;
    this.retryable = retryable;
  }
}

/**
 * Network trouble, throttling and server errors are worth retrying. 403 is included because a
 * presigned URL may have expired mid-upload; every attempt signs a fresh one.
 */
export function isRetryableStatus(status: number): boolean {
  return status === 0 || status === 403 || status === 408 || status === 429 || status >= 500;
}

function isRetryable(error: unknown): boolean {
  if (error instanceof PartUploadError) return error.retryable;
  // The API itself: only transient failures. 401/404/422 will not fix themselves.
  if (error instanceof ApiError) return error.status === 0 || error.status === 429 || error.status >= 500;
  return false;
}

export function partRange(partNumber: number, partSize: number, totalSize: number): { start: number; end: number } {
  const start = (partNumber - 1) * partSize;
  return { start, end: Math.min(start + partSize, totalSize) };
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export function backoffMs(attempt: number): number {
  return Math.min(500 * 2 ** (attempt - 1), 8000);
}

export async function uploadVideo(file: File, options: UploadOptions): Promise<Project> {
  const { api, putPart, signal, onProgress, onProjectCreated } = options;
  const concurrency = Math.max(1, options.concurrency ?? 3);
  const maxAttempts = Math.max(1, options.maxAttempts ?? 4);
  const sleep = options.sleep ?? defaultSleep;

  const throwIfCancelled = () => {
    if (signal?.aborted) throw new UploadCancelledError();
  };
  throwIfCancelled();

  const created = await api.create(file);
  throwIfCancelled();
  const { project, upload } = created;
  onProjectCreated?.(project, created.upload.uploaded_parts.length > 0);

  const totalBytes = file.size;
  const completed = new Map<number, string>(); // part number -> etag
  const inFlight = new Map<number, number>(); // part number -> bytes sent so far

  // Only trust a stored part if it has exactly the size we would send now.
  for (const part of upload.uploaded_parts) {
    const { start, end } = partRange(part.part_number, upload.part_size, totalBytes);
    if (part.part_number >= 1 && part.part_number <= upload.total_parts && part.size === end - start) {
      completed.set(part.part_number, part.etag);
    }
  }

  const report = () => {
    if (!onProgress) return;
    let uploadedBytes = 0;
    for (const n of completed.keys()) {
      const { start, end } = partRange(n, upload.part_size, totalBytes);
      uploadedBytes += end - start;
    }
    for (const loaded of inFlight.values()) uploadedBytes += loaded;
    onProgress({
      uploadedBytes: Math.min(uploadedBytes, totalBytes),
      totalBytes,
      partsDone: completed.size,
      totalParts: upload.total_parts,
    });
  };
  report();

  const pending: number[] = [];
  for (let n = 1; n <= upload.total_parts; n += 1) if (!completed.has(n)) pending.push(n);

  // One failure stops the other workers instead of letting them run on.
  const stop = new AbortController();
  const onOuterAbort = () => stop.abort();
  signal?.addEventListener("abort", onOuterAbort);

  const sendPart = async (n: number): Promise<void> => {
    const { start, end } = partRange(n, upload.part_size, totalBytes);
    const body = file.slice(start, end);

    for (let attempt = 1; ; attempt += 1) {
      if (stop.signal.aborted) throw new UploadCancelledError();
      try {
        const [signed] = await api.signParts(project.id, [n]);
        if (!signed) throw new PartUploadError(0, "The server didn't return an upload address.", false);
        inFlight.set(n, 0);
        const etag = await putPart(signed.url, body, {
          signal: stop.signal,
          onProgress: (loaded) => {
            inFlight.set(n, Math.min(loaded, end - start));
            report();
          },
        });
        inFlight.delete(n);
        completed.set(n, etag);
        report();
        return;
      } catch (error) {
        inFlight.delete(n);
        report();
        if (stop.signal.aborted) throw new UploadCancelledError();
        if (attempt >= maxAttempts || !isRetryable(error)) throw error;
        await sleep(backoffMs(attempt));
      }
    }
  };

  const worker = async () => {
    for (;;) {
      const n = pending.shift();
      if (n === undefined || stop.signal.aborted) return;
      try {
        await sendPart(n);
      } catch (error) {
        stop.abort();
        throw error;
      }
    }
  };

  try {
    const results = await Promise.allSettled(Array.from({ length: Math.min(concurrency, pending.length) }, worker));
    // Report the real cause, not the cancellations it triggered in the other workers.
    const failures = results.filter((r): r is PromiseRejectedResult => r.status === "rejected").map((r) => r.reason as unknown);
    if (signal?.aborted) throw new UploadCancelledError();
    const cause = failures.find((e) => !(e instanceof UploadCancelledError)) ?? failures[0];
    if (cause !== undefined) throw cause;
  } finally {
    signal?.removeEventListener("abort", onOuterAbort);
  }

  const parts: CompletedPart[] = [...completed.entries()]
    .sort(([a], [b]) => a - b)
    .map(([part_number, etag]) => ({ part_number, etag }));
  return api.complete(project.id, parts);
}

// --- waiting for the server-side analysis ---------------------------------------------------------

const ANALYSIS_IN_PROGRESS: ReadonlySet<ProjectStatus> = new Set(["UPLOADED", "ANALYZING"]);

export interface WaitOptions {
  fetchProject: (id: string) => Promise<Project>;
  signal?: AbortSignal;
  intervalMs?: number;
  timeoutMs?: number;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
}

export class AnalysisTimeoutError extends Error {
  constructor() {
    super("Analysis is taking longer than expected. It will keep running; check back in a moment.");
    this.name = "AnalysisTimeoutError";
  }
}

/** Polls until analysis has finished, either way (ready for masking, or failed with a reason). */
export async function waitForAnalysis(initial: Project, options: WaitOptions): Promise<Project> {
  const { fetchProject, signal } = options;
  const intervalMs = options.intervalMs ?? 1000;
  const timeoutMs = options.timeoutMs ?? 120_000;
  const sleep = options.sleep ?? defaultSleep;
  const now = options.now ?? Date.now;
  const deadline = now() + timeoutMs;

  let project = initial;
  while (ANALYSIS_IN_PROGRESS.has(project.status)) {
    if (signal?.aborted) throw new UploadCancelledError();
    if (now() >= deadline) throw new AnalysisTimeoutError();
    await sleep(intervalMs);
    if (signal?.aborted) throw new UploadCancelledError();
    project = await fetchProject(project.id);
  }
  return project;
}

// --- the browser's side of a part upload ----------------------------------------------------------

/** XMLHttpRequest rather than fetch: it is the only way to get upload progress. */
export const putPartWithXhr: PutPart = (url, body, { signal, onProgress }) =>
  new Promise<string>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);

    xhr.upload.onprogress = (event) => onProgress(event.loaded);
    xhr.onerror = () => reject(new PartUploadError(0, "The connection dropped while uploading."));
    xhr.ontimeout = () => reject(new PartUploadError(0, "The upload timed out."));
    xhr.onabort = () => reject(new UploadCancelledError());
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new PartUploadError(xhr.status, `Storage rejected part of the upload (${xhr.status}).`));
        return;
      }
      const etag = xhr.getResponseHeader("ETag");
      if (!etag) {
        // The part is stored, but the browser can't read the receipt: a storage CORS setting
        // (ExposeHeaders: ETag) is missing. Retrying cannot fix that.
        reject(
          new PartUploadError(
            xhr.status,
            "Storage didn't expose the upload receipt. Its CORS settings need to allow the ETag header.",
            false,
          ),
        );
        return;
      }
      resolve(etag);
    };

    const abort = () => xhr.abort();
    if (signal.aborted) {
      reject(new UploadCancelledError());
      return;
    }
    signal.addEventListener("abort", abort, { once: true });
    xhr.onloadend = () => signal.removeEventListener("abort", abort);
    xhr.send(body);
  });
