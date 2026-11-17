// @vitest-environment jsdom
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ExportPanel } from "@/components/result/ExportPanel";
import { ResultView } from "@/components/result/ResultView";
import { ToastProvider } from "@/components/Toast";
import type { ExportJob, ExportOptions } from "@/types/exports";

const mocks = vi.hoisted(() => ({
  options: vi.fn(),
  list: vi.fn(),
  create: vi.fn(),
  link: vi.fn(),
  get: vi.fn(),
  preview: vi.fn(),
}));

vi.mock("next/link", () => ({ default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => <a href={href} {...rest}>{children}</a> }));
vi.mock("next/navigation", () => ({ useParams: () => ({ id: "p1" }), useRouter: () => ({ replace: vi.fn() }), usePathname: () => "/result/p1" }));
vi.mock("@/components/Navbar", () => ({ Navbar: () => null }));
vi.mock("@/components/result/CompareView", () => ({ CompareView: () => <div data-testid="compare" /> }));
vi.mock("@/services/exports", () => ({ exportsService: { options: mocks.options, list: mocks.list, create: mocks.create, link: mocks.link } }));
vi.mock("@/services/projects", () => ({ projectsService: { get: mocks.get, preview: mocks.preview, exportResult: vi.fn(), remove: vi.fn() } }));

const sizes = (heights: number[]) => heights.map((h) => ({ height: h, label: `${h}p`, width_px: Math.round((h * 16) / 9), height_px: h }));

function options(over: Partial<ExportOptions> = {}): ExportOptions {
  return {
    original_width: 3840, original_height: 2160, original_label: "4K Ultra HD",
    output_width: 3840, output_height: 2160, output_label: "4K Ultra HD",
    codec: "h264", container: "mp4", sizes: sizes([1440, 1080, 720]), hardware_encoder: false,
    formats: [
      { container: "mp4", codec: "h264", label: "H.264 in MP4" },
      { container: "webm", codec: "vp9", label: "VP9 in WebM" },
    ],
    ...over,
  };
}

const job = (over: Partial<ExportJob> = {}): ExportJob => ({
  id: "e1", status: "queued", stage: "queued", frames_done: 0, frames_total: 60, percent: null, error: null, warnings: [],
  settings: { height: 1080, container: "mp4", codec: "h264" }, result: null, created_at: "", completed_at: null, ...over,
});

const panel = () => render(<ToastProvider><ExportPanel projectId="p1" /></ToastProvider>);

beforeEach(() => {
  Object.values(mocks).forEach((m) => m.mockReset());
  mocks.list.mockResolvedValue({ exports: [] });
});

describe("ExportPanel", () => {
  it("offers only the sizes the server says are valid, never the original's own size or larger", async () => {
    mocks.options.mockResolvedValue(options());
    panel();
    const size = await screen.findByLabelText("Size");
    const labels = within(size).getAllByRole("option").map((o) => o.textContent);
    expect(labels).toEqual(["Same as the finished video (3840 × 2160)", "1440p (2560 × 1440)", "1080p (1920 × 1080)", "720p (1280 × 720)"]);
    expect(labels.join()).not.toMatch(/4K|2160p/);
  });

  it("is not shown when there is nothing to choose (a 720p result in its only format)", async () => {
    mocks.options.mockResolvedValue(options({ sizes: [], output_width: 1280, output_height: 720, formats: [{ container: "mp4", codec: "h264", label: "H.264 in MP4" }] }));
    panel();
    await waitFor(() => expect(mocks.options).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByText("More sizes and formats")).not.toBeInTheDocument());
  });

  it("asks for the chosen size and format, then lists the export", async () => {
    mocks.options.mockResolvedValue(options());
    mocks.create.mockResolvedValue(job());
    panel();
    await screen.findByLabelText("Size");
    mocks.list.mockResolvedValue({ exports: [job()] });
    await userEvent.click(screen.getByRole("button", { name: "Make this copy" }));
    expect(mocks.create).toHaveBeenCalledWith("p1", { height: 1440, container: "mp4", codec: "h264", quality: "balanced" });
    expect(await screen.findByText("Waiting…")).toBeInTheDocument();
  });

  it("won't start an export of the video you already have", async () => {
    mocks.options.mockResolvedValue(options());
    panel();
    const size = await screen.findByLabelText("Size");
    await userEvent.selectOptions(size, "same");
    expect(screen.getByRole("button", { name: "Make this copy" })).toBeDisabled();
    expect(screen.getByText(/That is the video you already have/)).toBeInTheDocument();
  });

  it("downloads a finished export through a fresh link", async () => {
    mocks.options.mockResolvedValue(options());
    mocks.list.mockResolvedValue({ exports: [job({ status: "completed", percent: 100, result: { width: 1920, height: 1080, size_bytes: 5_000_000, container: "mp4", codec: "h264" } })] });
    mocks.link.mockResolvedValue({ url: "http://localhost/x.mp4", filename: "clip-restored-1080p.mp4", expires_in: 300, size_bytes: 5_000_000 });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    panel();
    expect(await screen.findByText("1920 × 1080 · H.264 in MP4")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Download" }));
    await waitFor(() => expect(mocks.link).toHaveBeenCalledWith("p1", "e1"));
    expect(click).toHaveBeenCalled();
    click.mockRestore();
  });

  it("shows the server's reason when an export fails", async () => {
    mocks.options.mockResolvedValue(options());
    mocks.list.mockResolvedValue({ exports: [job({ status: "failed", error: "The video could not be encoded: not enough disk space." })] });
    panel();
    expect(await screen.findByText("The video could not be encoded: not enough disk space.")).toBeInTheDocument();
  });
});

describe("finished video screen", () => {
  it("shows the original and the output resolution, read from the files themselves", async () => {
    mocks.get.mockResolvedValue({ id: "p1", name: "Drone", original_filename: "drone.mp4", status: "COMPLETED", width: 3840, height: 2160 });
    mocks.preview.mockResolvedValue({
      result_url: "u", original_url: "o", expires_in: 300, width: 3840, height: 2160, fps: 30, frame_count: 60, container: "mp4", size_bytes: 40_000_000,
      audio: "copied", development_only: false, warnings: [], finished_at: null, facts: {},
    });
    mocks.options.mockResolvedValue(options({ sizes: [], formats: [{ container: "mp4", codec: "h264", label: "H.264 in MP4" }] }));
    render(<ToastProvider><ResultView /></ToastProvider>);
    const list = (await screen.findByLabelText("Output details")) as HTMLElement;
    expect(list).toHaveTextContent("Original resolution3840 × 2160");
    expect(list).toHaveTextContent("Output resolution3840 × 2160");
    expect(list).toHaveTextContent("Quality4K Ultra HD");
    expect(list).toHaveTextContent("FormatMP4");
  });

  it("shows a smaller output next to the larger original when a smaller size was chosen", async () => {
    mocks.get.mockResolvedValue({ id: "p1", name: "Drone", original_filename: "drone.mp4", status: "COMPLETED", width: 3840, height: 2160 });
    mocks.preview.mockResolvedValue({
      result_url: "u", original_url: "o", expires_in: 300, width: 1920, height: 1080, fps: 30, frame_count: 60, container: "mp4", size_bytes: 9_000_000,
      audio: "copied", development_only: false, warnings: [], finished_at: null, facts: {},
    });
    mocks.options.mockResolvedValue(options({ output_width: 1920, output_height: 1080, sizes: sizes([720]) }));
    render(<ToastProvider><ResultView /></ToastProvider>);
    const list = (await screen.findByLabelText("Output details")) as HTMLElement;
    expect(list).toHaveTextContent("Original resolution3840 × 2160");
    expect(list).toHaveTextContent("Output resolution1920 × 1080");
  });
});
