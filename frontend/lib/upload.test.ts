import { describe, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import {
  AnalysisTimeoutError,
  PartUploadError,
  UploadCancelledError,
  backoffMs,
  isRetryableStatus,
  partRange,
  uploadVideo,
  waitForAnalysis,
  type PutPart,
  type UploadApi,
  type UploadProgress,
} from "./upload";
import type { CompletedPart, Project, ProjectStatus, ProjectUpload, UploadedPart } from "../types/project";

const PART = 10; // bytes; tiny parts keep the tests readable
const noSleep = () => Promise.resolve();

function project(status: ProjectStatus = "UPLOADING", extra: Partial<Project> = {}): Project {
  return {
    id: "p1",
    name: "clip",
    original_filename: "clip.mp4",
    size_bytes: 25,
    status,
    error: null,
    container: null,
    codec: null,
    audio_codec: null,
    width: null,
    height: null,
    fps: null,
    duration: null,
    bitrate: null,
    frame_count: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    expires_at: "2026-01-02T00:00:00Z",
    ...extra,
  };
}

function fakeFile(size: number): File {
  return new File([new Uint8Array(size)], "clip.mp4", { type: "video/mp4" });
}

function harness(opts: { size?: number; existing?: UploadedPart[] } = {}) {
  const size = opts.size ?? 25;
  const totalParts = Math.ceil(size / PART);
  const calls = { signed: [] as number[], completed: [] as unknown[] };
  const api: UploadApi = {
    create: vi.fn(
      async (): Promise<ProjectUpload> => ({
        project: project(),
        upload: { part_size: PART, total_parts: totalParts, uploaded_parts: opts.existing ?? [] },
      }),
    ),
    signParts: vi.fn(async (_id: string, numbers: number[]) => {
      calls.signed.push(...numbers);
      return numbers.map((n) => ({ part_number: n, url: `https://storage.test/part/${n}` }));
    }),
    complete: vi.fn(async (_id: string, parts: CompletedPart[]) => {
      calls.completed.push(parts);
      return project("UPLOADED");
    }),
  };
  return { api, calls, file: fakeFile(size), totalParts };
}

const partNumberOf = (url: string) => Number(url.split("/").pop());

describe("partRange", () => {
  it("splits a file into full parts and a shorter last part", () => {
    expect(partRange(1, 10, 25)).toEqual({ start: 0, end: 10 });
    expect(partRange(2, 10, 25)).toEqual({ start: 10, end: 20 });
    expect(partRange(3, 10, 25)).toEqual({ start: 20, end: 25 });
  });
});

describe("retry policy", () => {
  it("retries trouble that can clear up, not mistakes", () => {
    for (const status of [0, 403, 408, 429, 500, 503]) expect(isRetryableStatus(status)).toBe(true);
    for (const status of [400, 401, 404, 413, 422]) expect(isRetryableStatus(status)).toBe(false);
  });

  it("backs off exponentially with a ceiling", () => {
    expect([1, 2, 3, 4, 5, 6, 7].map(backoffMs)).toEqual([500, 1000, 2000, 4000, 8000, 8000, 8000]);
  });
});

describe("uploadVideo", () => {
  it("sends every part with the right bytes and completes with the receipts in order", async () => {
    const { api, calls, file } = harness();
    const sizes = new Map<number, number>();
    const putPart: PutPart = async (url, body) => {
      const n = partNumberOf(url);
      sizes.set(n, body.size);
      return `"etag-${n}"`;
    };

    const result = await uploadVideo(file, { api, putPart, sleep: noSleep });

    expect(result.status).toBe("UPLOADED");
    expect([...sizes.entries()].sort()).toEqual([[1, 10], [2, 10], [3, 5]]);
    expect(calls.completed).toEqual([
      [
        { part_number: 1, etag: '"etag-1"' },
        { part_number: 2, etag: '"etag-2"' },
        { part_number: 3, etag: '"etag-3"' },
      ],
    ]);
  });

  it("resumes: parts already in storage are not sent again", async () => {
    const { api, calls, file } = harness({
      existing: [
        { part_number: 1, etag: '"old-1"', size: 10 },
        { part_number: 3, etag: '"old-3"', size: 5 },
      ],
    });
    const sent: number[] = [];
    const putPart: PutPart = async (url) => {
      sent.push(partNumberOf(url));
      return '"new-2"';
    };
    const created = vi.fn();

    await uploadVideo(file, { api, putPart, sleep: noSleep, onProjectCreated: created });

    expect(sent).toEqual([2]);
    expect(created).toHaveBeenCalledWith(expect.objectContaining({ id: "p1" }), true);
    expect(calls.completed[0]).toEqual([
      { part_number: 1, etag: '"old-1"' },
      { part_number: 2, etag: '"new-2"' },
      { part_number: 3, etag: '"old-3"' },
    ]);
  });

  it("does not trust a stored part whose size is wrong", async () => {
    const { api, file } = harness({ existing: [{ part_number: 1, etag: '"short"', size: 4 }] });
    const sent: number[] = [];
    const putPart: PutPart = async (url) => {
      sent.push(partNumberOf(url));
      return '"fresh"';
    };
    await uploadVideo(file, { api, putPart, sleep: noSleep });
    expect(sent.sort()).toEqual([1, 2, 3]);
  });

  it("retries a flaky part, signing a fresh URL each time", async () => {
    const { api, calls, file } = harness({ size: 10 });
    let attempts = 0;
    const sleeps: number[] = [];
    const putPart: PutPart = async () => {
      attempts += 1;
      if (attempts < 3) throw new PartUploadError(0, "dropped");
      return '"ok"';
    };

    await uploadVideo(file, { api, putPart, sleep: async (ms) => void sleeps.push(ms) });

    expect(attempts).toBe(3);
    expect(calls.signed).toEqual([1, 1, 1]);
    expect(sleeps).toEqual([500, 1000]);
  });

  it("treats an expired signed URL (403) as retryable", async () => {
    const { api, file } = harness({ size: 10 });
    let attempts = 0;
    const putPart: PutPart = async () => {
      attempts += 1;
      if (attempts === 1) throw new PartUploadError(403, "expired");
      return '"ok"';
    };
    await expect(uploadVideo(file, { api, putPart, sleep: noSleep })).resolves.toBeDefined();
    expect(attempts).toBe(2);
  });

  it("gives up after the maximum attempts and surfaces the real error", async () => {
    const { api, calls, file } = harness({ size: 10 });
    const putPart: PutPart = async () => {
      throw new PartUploadError(503, "storage down");
    };
    await expect(uploadVideo(file, { api, putPart, sleep: noSleep, maxAttempts: 3 })).rejects.toThrow("storage down");
    expect(calls.signed).toHaveLength(3);
    expect(api.complete).not.toHaveBeenCalled();
  });

  it("does not retry errors that retrying cannot fix", async () => {
    const { api, calls, file } = harness({ size: 10 });
    const missingCors: PutPart = async () => {
      throw new PartUploadError(200, "CORS", false);
    };
    await expect(uploadVideo(file, { api, putPart: missingCors, sleep: noSleep })).rejects.toThrow("CORS");
    expect(calls.signed).toHaveLength(1);

    const { api: api2, file: file2 } = harness({ size: 10 });
    vi.mocked(api2.signParts).mockRejectedValue(new ApiError(404, "We couldn't find that project."));
    await expect(uploadVideo(file2, { api: api2, putPart: missingCors, sleep: noSleep })).rejects.toThrow("couldn't find");
    expect(api2.signParts).toHaveBeenCalledTimes(1);
  });

  it("retries transient API failures while signing", async () => {
    const { api, file } = harness({ size: 10 });
    vi.mocked(api.signParts)
      .mockRejectedValueOnce(new ApiError(0, "offline"))
      .mockResolvedValue([{ part_number: 1, url: "https://storage.test/part/1" }]);
    await expect(uploadVideo(file, { api, putPart: async () => '"ok"', sleep: noSleep })).resolves.toBeDefined();
    expect(api.signParts).toHaveBeenCalledTimes(2);
  });

  it("stops the other parts when one fails for good", async () => {
    const { api, file } = harness({ size: 50 });
    const started: number[] = [];
    const putPart: PutPart = async (url, _body, { signal }) => {
      const n = partNumberOf(url);
      started.push(n);
      if (n === 1) throw new PartUploadError(400, "bad request");
      await new Promise<void>((_, reject) => signal.addEventListener("abort", () => reject(new UploadCancelledError())));
      return '"never"';
    };
    await expect(uploadVideo(file, { api, putPart, sleep: noSleep, concurrency: 2 })).rejects.toThrow("bad request");
    expect(started).not.toContain(4);
    expect(started).not.toContain(5);
  });

  it("never has more than the allowed number of parts in flight", async () => {
    const { api, file } = harness({ size: 80 });
    let active = 0;
    let peak = 0;
    const putPart: PutPart = async (url) => {
      active += 1;
      peak = Math.max(peak, active);
      await new Promise((r) => setTimeout(r, 5));
      active -= 1;
      return `"${partNumberOf(url)}"`;
    };
    await uploadVideo(file, { api, putPart, sleep: noSleep, concurrency: 3 });
    expect(peak).toBe(3);
  });

  it("cancels cleanly and never completes", async () => {
    const { api, file } = harness({ size: 30 });
    const controller = new AbortController();
    const putPart: PutPart = (_url, _body, { signal }) =>
      new Promise((_, reject) => {
        controller.abort();
        signal.addEventListener("abort", () => reject(new UploadCancelledError()));
        if (signal.aborted) reject(new UploadCancelledError());
      });
    await expect(uploadVideo(file, { api, putPart, sleep: noSleep, signal: controller.signal })).rejects.toBeInstanceOf(
      UploadCancelledError,
    );
    expect(api.complete).not.toHaveBeenCalled();
  });

  it("does nothing at all if already cancelled", async () => {
    const { api, file } = harness();
    const controller = new AbortController();
    controller.abort();
    await expect(
      uploadVideo(file, { api, putPart: async () => '"x"', signal: controller.signal }),
    ).rejects.toBeInstanceOf(UploadCancelledError);
    expect(api.create).not.toHaveBeenCalled();
  });

  it("reports honest, monotonic progress that ends at the full size", async () => {
    const { api, file } = harness({ size: 25 });
    const seen: UploadProgress[] = [];
    const putPart: PutPart = async (url, body, { onProgress }) => {
      onProgress(Math.floor(body.size / 2));
      onProgress(body.size);
      return `"${partNumberOf(url)}"`;
    };
    await uploadVideo(file, { api, putPart, sleep: noSleep, concurrency: 1, onProgress: (p) => seen.push(p) });

    const bytes = seen.map((p) => p.uploadedBytes);
    expect(bytes[0]).toBe(0);
    expect(bytes).toEqual([...bytes].sort((a, b) => a - b));
    expect(bytes.at(-1)).toBe(25);
    expect(seen.at(-1)).toMatchObject({ partsDone: 3, totalParts: 3, totalBytes: 25 });
  });

  it("starts progress from what is already stored when resuming", async () => {
    const { api, file } = harness({ existing: [{ part_number: 1, etag: '"a"', size: 10 }, { part_number: 2, etag: '"b"', size: 10 }] });
    const seen: UploadProgress[] = [];
    await uploadVideo(file, { api, putPart: async () => '"c"', sleep: noSleep, onProgress: (p) => seen.push(p) });
    expect(seen[0]).toMatchObject({ uploadedBytes: 20, partsDone: 2 });
  });
});

describe("waitForAnalysis", () => {
  it("polls until the server has finished, then returns the analysed project", async () => {
    const fetchProject = vi
      .fn<(id: string) => Promise<Project>>()
      .mockResolvedValueOnce(project("ANALYZING"))
      .mockResolvedValueOnce(project("MASKING", { width: 320 }));
    const result = await waitForAnalysis(project("UPLOADED"), { fetchProject, sleep: noSleep });
    expect(result.status).toBe("MASKING");
    expect(result.width).toBe(320);
    expect(fetchProject).toHaveBeenCalledTimes(2);
  });

  it("returns a failed analysis as-is so the reason can be shown", async () => {
    const failed = project("FAILED", { error: "No video track was found." });
    const result = await waitForAnalysis(project("ANALYZING"), { fetchProject: async () => failed, sleep: noSleep });
    expect(result.error).toBe("No video track was found.");
  });

  it("does not poll if the project is already past analysis", async () => {
    const fetchProject = vi.fn();
    await waitForAnalysis(project("MASKING"), { fetchProject, sleep: noSleep });
    expect(fetchProject).not.toHaveBeenCalled();
  });

  it("stops waiting after the timeout instead of polling forever", async () => {
    let clock = 0;
    await expect(
      waitForAnalysis(project("ANALYZING"), {
        fetchProject: async () => project("ANALYZING"),
        sleep: async (ms) => void (clock += ms),
        now: () => clock,
        intervalMs: 1000,
        timeoutMs: 5000,
      }),
    ).rejects.toBeInstanceOf(AnalysisTimeoutError);
  });

  it("stops when cancelled", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      waitForAnalysis(project("ANALYZING"), { fetchProject: vi.fn(), signal: controller.signal, sleep: noSleep }),
    ).rejects.toBeInstanceOf(UploadCancelledError);
  });
});
