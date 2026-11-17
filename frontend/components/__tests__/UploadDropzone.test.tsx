// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UploadDropzone } from "@/components/UploadDropzone";
import { OWNERSHIP_NOTICE, PRIVACY_NOTICE } from "@/lib/site";

const { uploadVideo } = vi.hoisted(() => ({ uploadVideo: vi.fn() }));

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push, replace: vi.fn() }) }));
vi.mock("next/link", () => ({ default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => <a href={href} {...rest}>{children}</a> }));
vi.mock("@/lib/video", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/video")>()),
  readVideoMetadata: vi.fn().mockResolvedValue({ duration: 12.5, width: 1920, height: 1080 }),
}));
vi.mock("@/lib/upload", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/upload")>()),
  uploadVideo,
}));
vi.mock("@/services/projects", () => ({ projectsService: { get: vi.fn() } }));

function pick(file: File) {
  const input = document.querySelector<HTMLInputElement>("input[type=file]")!;
  fireEvent.change(input, { target: { files: [file] } });
}
const video = (name = "clip.mp4", type = "video/mp4", bytes = 2048) => new File([new Uint8Array(bytes)], name, { type });

describe("upload dropzone", () => {
  beforeEach(() => {
    uploadVideo.mockReset(); // braces matter: a function returned from beforeEach is run as a teardown
  });

  it("shows the ownership and privacy notices before anything is chosen", () => {
    render(<UploadDropzone />);
    expect(screen.getByRole("heading", { name: "Drop your video here" })).toBeInTheDocument();
    expect(screen.getByText(OWNERSHIP_NOTICE)).toBeInTheDocument();
    expect(screen.getByText(PRIVACY_NOTICE)).toBeInTheDocument();
    expect(screen.getByText("MP4, MOV, WebM or AVI")).toBeInTheDocument();
    expect(screen.getByText("Supported resolutions: 720p, 1080p, 1440p and 4K.")).toBeInTheDocument();
    expect(screen.getByText("or browse files")).toBeInTheDocument();
    expect(screen.getByText("Choose video")).toBeInTheDocument();
  });

  it.each([
    ["a document", "notes.txt", "text/plain", /Unsupported video format/],
    ["a disguised executable", "clip.mp4.exe", "application/octet-stream", /Unsupported video format/],
    ["an image", "photo.png", "image/png", /Unsupported video format/],
    ["an empty file", "empty.mp4", "video/mp4", /This file is empty/],
  ])("refuses %s in the browser and never starts an upload", (_label, name, type, message) => {
    render(<UploadDropzone />);
    pick(new File([new Uint8Array(name === "empty.mp4" ? 0 : 10)], name, { type }));
    expect(screen.getByRole("alert")).toHaveTextContent(message);
    expect(screen.queryByRole("button", { name: "Upload video" })).not.toBeInTheDocument();
    expect(uploadVideo).not.toHaveBeenCalled();
  });

  it("takes a 600 MB 4K file: the default limit is 2 GiB, not 500 MB", () => {
    render(<UploadDropzone />);
    const file = video("4k.mp4");
    Object.defineProperty(file, "size", { value: 600 * 1024 * 1024 });
    pick(file);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload video" })).toBeInTheDocument();
  });

  it("refuses a file over the size limit and says what the limit is", () => {
    render(<UploadDropzone />);
    const big = video("huge.mp4");
    Object.defineProperty(big, "size", { value: 2.5 * 1024 * 1024 * 1024 });
    pick(big);
    expect(screen.getByRole("alert")).toHaveTextContent(/over the 2\.0 GB limit/);
    expect(uploadVideo).not.toHaveBeenCalled();
  });

  it("refuses several dropped files at once", () => {
    render(<UploadDropzone />);
    const zone = screen.getByRole("heading", { name: "Drop your video here" }).parentElement!;
    fireEvent.drop(zone, { dataTransfer: { files: [video("a.mp4"), video("b.mp4")] } });
    expect(screen.getByRole("alert")).toHaveTextContent("Drop one video at a time.");
  });

  it("shows the chosen file and what the browser could read, but no server-side facts yet", async () => {
    render(<UploadDropzone />);
    pick(video("holiday.mp4"));
    expect(screen.getByText("holiday.mp4")).toBeInTheDocument();
    expect(await screen.findByText(/1920/)).toBeInTheDocument();
    expect(screen.getByText(/read on the server after upload/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload video" })).toBeEnabled();
  });

  it("reports a failed upload in words and offers to resume once the project exists", async () => {
    uploadVideo.mockImplementation(async (_file: File, options: { onProjectCreated: (p: unknown) => void }) => {
      options.onProjectCreated({ id: "p1" });
      throw new Error("The connection dropped while sending part 2.");
    });
    render(<UploadDropzone />);
    pick(video());
    await userEvent.click(screen.getByRole("button", { name: "Upload video" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The connection dropped while sending part 2.");
    expect(screen.getByRole("button", { name: "Resume upload" })).toBeInTheDocument();
  });

  it("does not offer to resume when the project was never created", async () => {
    uploadVideo.mockRejectedValue(new Error("Too many requests. Please wait 4 minutes and try again."));
    render(<UploadDropzone />);
    pick(video());
    await userEvent.click(screen.getByRole("button", { name: "Upload video" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/Please wait 4 minutes/);
    await waitFor(() => expect(screen.getByRole("button", { name: "Upload video" })).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Resume upload" })).not.toBeInTheDocument();
  });

  it("shows the server's reason when the uploaded file turns out not to be a usable video", async () => {
    const failed = { id: "p2", status: "FAILED", error: "We couldn't read this video. The file may be corrupted. Please upload another copy." };
    uploadVideo.mockResolvedValue(failed);
    const { projectsService } = await import("@/services/projects");
    vi.mocked(projectsService.get).mockResolvedValue(failed as never);
    render(<UploadDropzone />);
    pick(video());
    await userEvent.click(screen.getByRole("button", { name: "Upload video" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("We couldn't read this video.");
    expect(screen.queryByRole("link", { name: "Continue to selection" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose another file" })).toBeInTheDocument();
  });
  it("shows what the server found in a 4K file: resolution, quality class, frame rate and format", async () => {
    const analysed = {
      id: "p3", status: "MASKING", error: null, name: "Drone", original_filename: "drone.mp4", size_bytes: 900_000_000,
      container: "mov", codec: "h264", audio_codec: "aac", width: 3840, height: 2160, fps: 30, duration: 12.5, bitrate: 50_000_000,
      frame_count: 375, created_at: "", updated_at: "", expires_at: "",
    };
    uploadVideo.mockResolvedValue(analysed);
    const { projectsService } = await import("@/services/projects");
    vi.mocked(projectsService.get).mockResolvedValue(analysed as never);
    render(<UploadDropzone />);
    pick(video("drone.mp4"));
    await userEvent.click(screen.getByRole("button", { name: "Upload video" }));
    const details = await screen.findByText("3840 × 2160");
    const list = details.closest("dl") as HTMLElement;
    expect(list).toHaveTextContent("Resolution3840 × 2160");
    expect(list).toHaveTextContent("Quality4K Ultra HD");
    expect(list).toHaveTextContent("Frame rate30 FPS");
    expect(list).toHaveTextContent("FormatMP4");
  });
});
