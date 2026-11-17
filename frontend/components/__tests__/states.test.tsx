// @vitest-environment jsdom
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthForm } from "@/components/AuthForm";
import { ProjectsGrid } from "@/components/ProjectsGrid";
import { ToastProvider } from "@/components/Toast";
import { ApiError } from "@/lib/api";
import type { Project } from "@/types/project";

const { list, remove, login, register, replace } = vi.hoisted(() => ({
  list: vi.fn(),
  remove: vi.fn(),
  login: vi.fn(),
  register: vi.fn(),
  replace: vi.fn(),
}));

vi.mock("next/link", () => ({ default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => <a href={href} {...rest}>{children}</a> }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }), useSearchParams: () => new URLSearchParams() }));
vi.mock("@/components/AuthProvider", () => ({ useAuth: () => ({ status: "anonymous", login, register }) }));
vi.mock("@/services/projects", () => ({ projectsService: { list, remove, analyze: vi.fn(), media: vi.fn().mockRejectedValue(new Error("no preview")) } }));

const inHours = (hours: number) => new Date(Date.now() + hours * 3600_000).toISOString();
function project(overrides: Partial<Project>): Project {
  return {
    id: "p1",
    name: "Beach day",
    original_filename: "beach.mp4",
    size_bytes: 5_000_000,
    status: "MASKING",
    error: null,
    container: "mp4",
    codec: "h264",
    audio_codec: "aac",
    width: 1280,
    height: 720,
    fps: 30,
    duration: 12,
    bitrate: 1000,
    frame_count: 360,
    created_at: inHours(-1),
    updated_at: inHours(-1),
    expires_at: inHours(47),
    ...overrides,
  };
}

beforeEach(() => {
  list.mockReset();
  remove.mockReset();
  login.mockReset();
  register.mockReset();
  replace.mockReset();
});

describe("project list states", () => {
  it("says what is happening while loading, then that nothing exists yet", async () => {
    list.mockResolvedValue([]);
    render(<ToastProvider><ProjectsGrid /></ToastProvider>);
    expect(screen.getByRole("status")).toHaveTextContent("Loading your projects");
    expect(await screen.findByText("No projects yet")).toBeInTheDocument();
  });

  it("explains a load failure and lets the person try again", async () => {
    list.mockRejectedValueOnce(new ApiError(503, "We can't reach our database right now. Please try again in a moment."));
    render(<ToastProvider><ProjectsGrid /></ToastProvider>);
    expect(await screen.findByText("We couldn't load your projects")).toBeInTheDocument();
    expect(screen.getByText(/can't reach our database/)).toBeInTheDocument();
    list.mockResolvedValueOnce([project({})]);
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Beach day")).toBeInTheDocument();
  });

  it.each([
    ["UPLOADING", "Upload unfinished"],
    ["ANALYZING", "Processing"],
    ["MASKING", "Ready to edit"],
    ["PROCESSING", "Processing"],
    ["ENCODING", "Processing"],
    ["COMPLETED", "Completed"],
    ["FAILED", "Failed"],
    ["EXPIRED", "Expired"],
  ] as const)("shows %s as '%s'", async (status, label) => {
    list.mockResolvedValue([project({ status })]);
    render(<ToastProvider><ProjectsGrid /></ToastProvider>);
    expect(await screen.findByText(label)).toBeInTheDocument();
  });

  it("links only to what a project in that state can actually do", async () => {
    list.mockResolvedValue([
      project({ id: "a", name: "Done", status: "COMPLETED" }),
      project({ id: "b", name: "Waiting", status: "UPLOADING" }),
      project({ id: "c", name: "Gone", status: "EXPIRED" }),
    ]);
    render(<ToastProvider><ProjectsGrid /></ToastProvider>);
    await screen.findByText("Done");
    const rows = screen.getAllByRole("listitem");
    const links = (row: HTMLElement) => within(row).queryAllByRole("link").map((a) => a.getAttribute("href"));
    expect(links(rows[0])).toEqual(["/result/a", "/result/a", "/editor/a"]); // picture, Open result, Edit selection
    expect(links(rows[1])).toEqual([]);
    expect(links(rows[2])).toEqual([]);
  });

  it("shows why a project failed, and when an expired one was deleted instead of a date", async () => {
    list.mockResolvedValue([
      project({ id: "f", name: "Broken", status: "FAILED", error: "We couldn't read this video. The file may be corrupted. Please upload another copy." }),
      project({ id: "e", name: "Old", status: "EXPIRED" }),
      project({ id: "s", name: "Fresh", status: "MASKING" }),
    ]);
    render(<ToastProvider><ProjectsGrid /></ToastProvider>);
    expect(await screen.findByText(/We couldn't read this video/)).toBeInTheDocument();
    expect(screen.getByText(/The files were deleted after the retention period/)).toBeInTheDocument();
    expect(screen.getAllByText(/Files are deleted automatically on /)).toHaveLength(2); // the failed and the fresh project; the expired one has no date
  });

  it("needs a confirmation before deleting and reports a failed delete without removing the row", async () => {
    list.mockResolvedValue([project({})]);
    remove.mockRejectedValue(new ApiError(500, "Something went wrong on our side. Please try again in a moment. If it keeps happening, quote reference abc123."));
    render(<ToastProvider><ProjectsGrid /></ToastProvider>);
    await userEvent.click(await screen.findByRole("button", { name: "Delete Beach day" }));
    expect(remove).not.toHaveBeenCalled();
    expect(screen.getByText("Delete this video and its project?")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/quote reference abc123/);
    expect(screen.getByText("Beach day")).toBeInTheDocument();
  });
});

describe("sign-in and sign-up error states", () => {
  it("shows the server's lockout message and re-enables the form", async () => {
    login.mockRejectedValue(new ApiError(429, "Too many failed sign-in attempts for this account. Please wait 15 minutes and try again."));
    render(<AuthForm mode="login" />);
    await userEvent.type(screen.getByLabelText(/Email/i), "ada@example.com");
    await userEvent.type(screen.getByLabelText(/Password/i), "wrong password here");
    await userEvent.click(screen.getByRole("button", { name: /Log in/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Please wait 15 minutes");
    expect(screen.getByRole("button", { name: /Log in/i })).toBeEnabled();
  });

  it("refuses a short password before contacting the server", async () => {
    render(<AuthForm mode="signup" />);
    await userEvent.type(screen.getByLabelText(/Name/i), "Ada");
    await userEvent.type(screen.getByLabelText(/Email/i), "ada@example.com");
    await userEvent.type(screen.getByLabelText(/Password/i), "short");
    await userEvent.click(screen.getByRole("button", { name: /Create account/i }));
    expect(screen.getByRole("alert")).toHaveTextContent("at least 10 characters");
    expect(register).not.toHaveBeenCalled();
  });

  it("tells a person whose email is taken to log in instead", async () => {
    register.mockRejectedValue(new ApiError(409, "An account with this email already exists. Log in instead."));
    render(<AuthForm mode="signup" />);
    await userEvent.type(screen.getByLabelText(/Name/i), "Ada");
    await userEvent.type(screen.getByLabelText(/Email/i), "ada@example.com");
    await userEvent.type(screen.getByLabelText(/Password/i), "correct horse battery");
    await userEvent.click(screen.getByRole("button", { name: /Create account/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Log in instead");
  });

  it("falls back to a plain message for an error that isn't from the API", async () => {
    login.mockRejectedValue(new TypeError("Failed to fetch"));
    render(<AuthForm mode="login" />);
    await userEvent.type(screen.getByLabelText(/Email/i), "ada@example.com");
    await userEvent.type(screen.getByLabelText(/Password/i), "whatever works");
    await userEvent.click(screen.getByRole("button", { name: /Log in/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong. Please try again.");
  });
});
