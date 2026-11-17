// @vitest-environment jsdom
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DetectionBanner } from "@/components/editor/DetectionSuggestions";
import { ProcessingProgress } from "@/components/editor/ProcessingProgress";
import { RemovalSettings } from "@/components/editor/RemovalSettings";
import { StepIndicator } from "@/components/StepIndicator";
import { ToastProvider, useToast } from "@/components/Toast";
import { VideoComparisonSlider } from "@/components/VideoComparisonSlider";
import { DEFAULT_SETTINGS } from "@/lib/render";
import { SEARCH_FROM, STATUS_BADGE, filterProjects, projectHref } from "@/lib/projects";
import type { Project } from "@/types/project";
import type { EngineInfo, Job, Stage } from "@/types/render";

describe("step indicator", () => {
  it("marks the current step and the ones already done", () => {
    render(<StepIndicator current={3} />);
    const items = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(items.map((li) => li.textContent?.replace(/^\d?/, "").replace(/^(Done: |Current step: |Next: )/, "").trim())).toEqual(["Upload", "Select", "Restore", "Export"]);
    expect(items[2]).toHaveAttribute("aria-current", "step");
    expect(items[0]).toHaveTextContent("Done: Upload");
    expect(items[3]).toHaveTextContent("Next: Export");
  });
});

describe("toasts", () => {
  function Trigger() {
    const toast = useToast();
    return (
      <>
        <button onClick={() => toast({ kind: "error", message: "Nope" })}>fail</button>
        <button onClick={() => toast({ kind: "success", message: "Done", duration: 1000 })}>ok</button>
      </>
    );
  }

  it("announces errors as alerts, and successes politely, and lets the person dismiss them", async () => {
    render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    );
    await userEvent.click(screen.getByText("fail"));
    expect(screen.getByRole("alert")).toHaveTextContent("Nope");
    await userEvent.click(screen.getByText("ok"));
    expect(screen.getByRole("status")).toHaveTextContent("Done");
    await userEvent.click(within(screen.getByRole("alert")).getByRole("button", { name: "Dismiss notification" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("goes away by itself", () => {
    vi.useFakeTimers();
    render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    );
    fireEvent.click(screen.getByText("ok"));
    expect(screen.getByText("Done")).toBeInTheDocument();
    act(() => void vi.advanceTimersByTime(1200));
    expect(screen.queryByText("Done")).not.toBeInTheDocument();
    vi.useRealTimers();
  });

  it("is harmless without a provider", () => {
    render(<Trigger />);
    fireEvent.click(screen.getByText("fail"));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("comparison slider", () => {
  it("reveals the original up to the divider and moves with the keyboard", () => {
    render(<VideoComparisonSlider aspect="9 / 16" before={<i data-testid="before" />} after={<i data-testid="after" />} />);
    const slider = screen.getByRole("slider");
    expect(slider).toHaveAttribute("aria-valuetext", "50% original");
    fireEvent.change(slider, { target: { value: "80" } });
    expect(slider).toHaveAttribute("aria-valuetext", "80% original");
    expect(screen.getByTestId("before").parentElement).toHaveStyle({ clipPath: "inset(0 20% 0 0)" });
    expect(screen.getByText("Original")).toBeInTheDocument();
    expect(screen.getByText("Restored")).toBeInTheDocument();
  });
});

const engine = (over: Partial<EngineInfo> = {}): EngineInfo => ({
  provider: "auto",
  name: "Automatic overlay removal",
  available: true,
  development_only: false,
  sends_frames_to_third_party: false,
  device: null,
  reason: null,
  expectation: "Runs on this computer.",
  ...over,
});

describe("removal settings", () => {
  const base = {
    settings: DEFAULT_SETTINGS,
    mode: "ai" as const,
    onSettings: vi.fn(),
    onMode: vi.fn(),
    engine: engine(),
    engineProblem: null,
    heights: [720] as Array<720 | 1080 | 1440 | 2160>,
    hasAudio: true,
    acknowledged: false,
    onAcknowledged: vi.fn(),
    disabled: false,
  };

  it("defaults to Balanced and explains each setting under it", () => {
    render(<RemovalSettings {...base} />);
    expect(screen.getByRole("heading", { name: "Removal settings" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Balanced" })).toBeChecked();
    expect(screen.getByText(/about 10 neighbouring frames/)).toBeInTheDocument();
    expect(screen.getByText(/Rebuilds what was behind the mark/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Preserve original audio" })).toBeChecked();
  });

  it("changes quality and mode", async () => {
    const onSettings = vi.fn();
    const onMode = vi.fn();
    render(<RemovalSettings {...base} onSettings={onSettings} onMode={onMode} />);
    await userEvent.click(screen.getByRole("radio", { name: "High quality" }));
    expect(onSettings).toHaveBeenCalledWith({ ...DEFAULT_SETTINGS, quality: "high" });
    await userEvent.click(screen.getByRole("radio", { name: "Fast processing" }));
    expect(onMode).toHaveBeenCalledWith("fast");
  });

  it("switches quality off, and says so, while Fast processing is chosen", () => {
    render(<RemovalSettings {...base} mode="fast" />);
    expect(screen.getByRole("radio", { name: "Balanced" })).toBeDisabled();
    expect(screen.getByText(/Not used while Fast processing is on/)).toBeInTheDocument();
  });

  it("keeps format, size and engine details inside a collapsed Advanced section", async () => {
    render(<RemovalSettings {...base} />);
    expect(screen.queryByLabelText("Format")).not.toBeInTheDocument();
    const toggle = screen.getByRole("button", { name: "Advanced settings" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    expect(screen.getByLabelText("Format")).toBeInTheDocument();
    expect(screen.getByLabelText("Size")).toBeInTheDocument();
    expect(screen.getByText(/Automatic overlay removal/)).toBeInTheDocument();
  });

  it("says plainly when there is no audio to keep", () => {
    render(<RemovalSettings {...base} hasAudio={false} />);
    expect(screen.queryByRole("checkbox", { name: "Preserve original audio" })).not.toBeInTheDocument();
    expect(screen.getByText("This video has no audio track.")).toBeInTheDocument();
  });

  it("always shows the development-engine warning and the outside-service consent, even with Advanced closed", async () => {
    const onAcknowledged = vi.fn();
    render(<RemovalSettings {...base} engine={engine({ development_only: true, sends_frames_to_third_party: true, name: "Remote API" })} onAcknowledged={onAcknowledged} />);
    expect(screen.getByRole("note")).toHaveTextContent(/not an AI restoration/);
    await userEvent.click(screen.getByRole("checkbox", { name: /sent to an outside service \(Remote API\)/ }));
    expect(onAcknowledged).toHaveBeenCalledWith(true);
  });
});

describe("processing progress", () => {
  const stage = (key: string, state: Stage["state"], detail: string | null = null): Stage => ({ key, label: key, state, detail });
  const job = (over: Partial<Job>): Job =>
    ({
      id: "j1",
      project_id: "p1",
      status: "running",
      stage: "reconstructing",
      stages: [
        stage("analyze", "done"),
        stage("mask", "done"),
        stage("track", "skipped", "Not used: the marks stay where you placed them"),
        stage("preparing", "done"),
        stage("measuring", "done"),
        stage("reconstructing", "active", "12 of 96 frames"),
        stage("audio", "pending"),
        stage("verifying", "pending"),
        stage("saving", "pending"),
      ],
      frames_done: 12,
      frames_total: 96,
      percent: 12.5,
      speed_fps: 4,
      eta_seconds: 21,
      jobs_ahead: null,
      error: null,
      warnings: [],
      engine: { provider: "auto", development_only: false },
      settings: {},
      result: null,
      created_at: "",
      started_at: null,
      completed_at: null,
      ...over,
    }) as Job;

  it("shows the four steps and the server's own numbers", () => {
    render(<ProcessingProgress job={job({})} busy={false} onCancel={vi.fn()} />);
    const steps = within(screen.getByRole("list", { name: "Processing steps" })).getAllByRole("listitem");
    expect(steps.map((s) => s.textContent)).toEqual([
      expect.stringContaining("Preparing video"),
      expect.stringContaining("Tracking selected region"),
      expect.stringContaining("Reconstructing video"),
      expect.stringContaining("Finalizing export"),
    ]);
    expect(steps[2]).toHaveAttribute("aria-current", "step");
    expect(screen.getByRole("progressbar", { name: "Reconstruction progress" })).toHaveAttribute("aria-valuenow", "13");
    expect(screen.getByText(/13% · 12 of 96 frames · 4\.0 frames\/s/)).toBeInTheDocument();
    expect(screen.getByText(/About 20 seconds left/)).toBeInTheDocument();
  });

  it("never invents a percentage or an estimate when the server has none", () => {
    render(<ProcessingProgress job={job({ percent: null, eta_seconds: null, speed_fps: null })} busy={false} onCancel={vi.fn()} />);
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
    expect(screen.queryByText(/left for this step/)).not.toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Working" })).not.toHaveAttribute("aria-valuenow");
  });

  it("offers Cancel, and not while the result is being saved", async () => {
    const onCancel = vi.fn();
    const { rerender } = render(<ProcessingProgress job={job({})} busy={false} onCancel={onCancel} />);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
    rerender(<ProcessingProgress job={job({ stage: "saving" })} busy={false} onCancel={onCancel} />);
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });
});

describe("automatic detection banner", () => {
  const box = { x: 0.6, y: 0.1, w: 0.3, h: 0.08 };

  it("shows a loading state, then asks before using anything it found", async () => {
    const onUse = vi.fn();
    const onAdjust = vi.fn();
    const { rerender } = render(<DetectionBanner state={{ status: "working" }} onUse={onUse} onAdjust={onAdjust} onDismiss={vi.fn()} />);
    expect(screen.getByRole("status")).toHaveTextContent("Finding unwanted overlays...");
    rerender(<DetectionBanner state={{ status: "found", boxes: [box] }} onUse={onUse} onAdjust={onAdjust} onDismiss={vi.fn()} />);
    expect(screen.getByText("Possible overlay detected")).toBeInTheDocument();
    expect(onUse).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Use selection" }));
    await userEvent.click(screen.getByRole("button", { name: "Adjust manually" }));
    expect(onUse).toHaveBeenCalledTimes(1);
    expect(onAdjust).toHaveBeenCalledTimes(1);
  });

  it("recommends manual selection when nothing is found or detection cannot tell", () => {
    const { rerender } = render(<DetectionBanner state={{ status: "none" }} onUse={vi.fn()} onAdjust={vi.fn()} onDismiss={vi.fn()} />);
    expect(screen.getByText(/No overlay found/)).toHaveTextContent(/Rectangle or Brush/);
    rerender(<DetectionBanner state={{ status: "unknown", reason: "The picture barely changes." }} onUse={vi.fn()} onAdjust={vi.fn()} onDismiss={vi.fn()} />);
    expect(screen.getByText(/The picture barely changes\. Try selecting/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Use selection" })).not.toBeInTheDocument();
  });
});

describe("project helpers", () => {
  const p = (over: Partial<Project>): Project => ({ id: "x", name: "Beach", original_filename: "beach.mp4", status: "MASKING", ...over }) as Project;

  it("uses the three headline labels", () => {
    expect(STATUS_BADGE.PROCESSING.label).toBe("Processing");
    expect(STATUS_BADGE.COMPLETED.label).toBe("Completed");
    expect(STATUS_BADGE.FAILED.label).toBe("Failed");
  });

  it("opens the editor or the result only where one exists", () => {
    expect(projectHref(p({ status: "COMPLETED" }))).toBe("/result/x");
    expect(projectHref(p({ status: "MASKING" }))).toBe("/editor/x");
    expect(projectHref(p({ status: "FAILED" }))).toBeNull();
    expect(projectHref(p({ status: "UPLOADING" }))).toBeNull();
  });

  it("searches by project name and original file name, and only offers search for long lists", () => {
    const list = [p({ id: "1", name: "Beach day" }), p({ id: "2", name: "Lake", original_filename: "IMG_2041.MOV" })];
    expect(filterProjects(list, "  beach ").map((x) => x.id)).toEqual(["1"]);
    expect(filterProjects(list, "img_2041").map((x) => x.id)).toEqual(["2"]);
    expect(filterProjects(list, "")).toHaveLength(2);
    expect(SEARCH_FROM).toBeGreaterThan(2);
  });
});
