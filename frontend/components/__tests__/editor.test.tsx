// @vitest-environment jsdom
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RegionList } from "@/components/editor/RegionList";
import { SelectionToolbar } from "@/components/editor/SelectionToolbar";
import { VideoTimeline } from "@/components/editor/VideoTimeline";
import type { MaskDocument, Region } from "@/types/mask";

const logo: Region = {
  id: "r1",
  name: "Corner logo",
  type: "rectangle",
  keyframes: [
    { frame: 0, rect: { x: 0.1, y: 0.1, w: 0.2, h: 0.1 } },
    { frame: 30, rect: { x: 0.5, y: 0.1, w: 0.2, h: 0.1 } },
  ],
};
const stamp: Region = { id: "r2", name: "Timestamp", type: "polygon", keyframes: [{ frame: 0, points: [[0, 0], [0.2, 0], [0.2, 0.1]] }] };
const doc = (...regions: Region[]): MaskDocument => ({ version: 1, regions });

describe("selection toolbar", () => {
  const base = { tool: "select" as const, onTool: vi.fn(), brushRadius: 0.02, onBrushRadius: vi.fn(), detecting: false, onDetect: vi.fn() };

  it("marks the active tool and switches tool on click", async () => {
    const onTool = vi.fn();
    render(<SelectionToolbar {...base} onTool={onTool} tool="rectangle" />);
    expect(screen.getByRole("button", { name: "Rectangle" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Polygon" })).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(screen.getByRole("button", { name: "Polygon" }));
    expect(onTool).toHaveBeenCalledWith("polygon");
  });

  it("offers a size control only for brush and eraser", () => {
    const { rerender } = render(<SelectionToolbar {...base} tool="rectangle" />);
    expect(screen.queryByText("Size")).not.toBeInTheDocument();
    rerender(<SelectionToolbar {...base} tool="brush" />);
    expect(screen.getByText("Size")).toBeInTheDocument();
    rerender(<SelectionToolbar {...base} tool="eraser" />);
    expect(screen.getByText("Size")).toBeInTheDocument();
  });

  it("turns every tool, and automatic detection, off while tracking or processing runs", () => {
    const { rerender } = render(<SelectionToolbar {...base} />);
    for (const name of ["Adjust", "Rectangle", "Polygon", "Brush", "Eraser", "Automatic detection"]) expect(screen.getByRole("button", { name })).toBeEnabled();
    rerender(<SelectionToolbar {...base} locked />);
    for (const name of ["Adjust", "Rectangle", "Polygon", "Brush", "Eraser", "Automatic detection"]) expect(screen.getByRole("button", { name })).toBeDisabled();
  });

  it("starts automatic detection on click, and shows it is busy while it runs", async () => {
    const onDetect = vi.fn();
    const { rerender } = render(<SelectionToolbar {...base} onDetect={onDetect} />);
    await userEvent.click(screen.getByRole("button", { name: "Automatic detection" }));
    expect(onDetect).toHaveBeenCalledTimes(1);
    rerender(<SelectionToolbar {...base} onDetect={onDetect} detecting />);
    expect(screen.getByRole("button", { name: "Automatic detection" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Automatic detection" })).toHaveAttribute("aria-busy", "true");
  });

  it("explains each tool in a tooltip that names its shortcut", () => {
    render(<SelectionToolbar {...base} />);
    expect(screen.getByText(/Rectangle \(R\): Drag across the area/)).toHaveAttribute("role", "tooltip");
  });
});

describe("region list (mask selection)", () => {
  const props = { selectedId: null, onSelect: vi.fn(), onRename: vi.fn(), onDelete: vi.fn() };

  it("tells the person what to do when nothing is marked", () => {
    render(<RegionList {...props} doc={doc()} />);
    expect(screen.getByText(/Nothing marked yet/)).toBeInTheDocument();
  });

  it("lists regions with their type and keyframe count and selects on click", async () => {
    const onSelect = vi.fn();
    render(<RegionList {...props} onSelect={onSelect} doc={doc(logo, stamp)} />);
    expect(screen.getByText("Rectangle · 2 keyframes")).toBeInTheDocument();
    expect(screen.getByText("Polygon · 1 keyframe")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Timestamp/ }));
    expect(onSelect).toHaveBeenCalledWith("r2");
  });

  it("renames once on Enter, keeps the old name for an empty one, and clears the region", async () => {
    const onRename = vi.fn();
    const onDelete = vi.fn();
    render(<RegionList {...props} selectedId="r1" onRename={onRename} onDelete={onDelete} doc={doc(logo)} />);
    const field = screen.getByDisplayValue("Corner logo");
    await userEvent.clear(field);
    await userEvent.type(field, "Channel bug{Enter}");
    expect(onRename).toHaveBeenCalledExactlyOnceWith("r1", "Channel bug");
    await userEvent.clear(field);
    fireEvent.blur(field);
    expect(onRename).toHaveBeenCalledTimes(1);
    expect(field).toHaveValue("Corner logo");
    await userEvent.click(screen.getByRole("button", { name: "Clear region" }));
    expect(onDelete).toHaveBeenCalledWith("r1");
  });

  it("cannot be edited while tracking runs", () => {
    render(<RegionList {...props} selectedId="r1" locked doc={doc(logo)} />);
    expect(screen.getByRole("button", { name: "Clear region" })).toBeDisabled();
    expect(screen.getByDisplayValue("Corner logo")).toBeDisabled();
  });
});

describe("video timeline", () => {
  const props = {
    frame: 10,
    frameCount: 60,
    fps: 30,
    playing: false,
    regions: [logo],
    selectedId: "r1",
    lost: [],
    tracked: false,
    locked: false,
    onSeek: vi.fn(),
    onTogglePlay: vi.fn(),
    onStep: vi.fn(),
    onAddKeyframe: vi.fn(),
    onRemoveKeyframe: vi.fn(),
    onFullscreen: vi.fn(),
  };

  it("says where the playhead is, in frames and seconds", () => {
    render(<VideoTimeline {...props} />);
    expect(screen.getByText(/frame 11 of 60/)).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: "Video position" })).toHaveAttribute("aria-valuetext", "Frame 11 of 60");
  });

  it("seeks from the slider and from a keyframe marker", () => {
    const onSeek = vi.fn();
    render(<VideoTimeline {...props} onSeek={onSeek} />);
    fireEvent.change(screen.getByRole("slider", { name: "Video position" }), { target: { value: "42" } });
    expect(onSeek).toHaveBeenLastCalledWith(42);
    fireEvent.click(screen.getByRole("button", { name: "Corner logo: keyframe at frame 31" }));
    expect(onSeek).toHaveBeenLastCalledWith(30);
  });

  it("plays, pauses and steps by one frame", async () => {
    const onTogglePlay = vi.fn();
    const onStep = vi.fn();
    const { rerender } = render(<VideoTimeline {...props} onTogglePlay={onTogglePlay} onStep={onStep} />);
    await userEvent.click(screen.getByRole("button", { name: "Play" }));
    rerender(<VideoTimeline {...props} playing onTogglePlay={onTogglePlay} onStep={onStep} />);
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Back one frame" }));
    await userEvent.click(screen.getByRole("button", { name: "Forward one frame" }));
    expect(onStep.mock.calls).toEqual([[-1], [1]]);
    expect(onTogglePlay).toHaveBeenCalledTimes(1);
  });

  it("allows a keyframe only where there is none and a region is selected", () => {
    const { rerender } = render(<VideoTimeline {...props} />);
    expect(screen.getByRole("button", { name: "Add keyframe here" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Remove keyframe" })).toBeDisabled();
    rerender(<VideoTimeline {...props} frame={30} />);
    expect(screen.getByRole("button", { name: "Add keyframe here" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove keyframe" })).toBeEnabled();
    rerender(<VideoTimeline {...props} selectedId={null} />);
    expect(screen.getByRole("button", { name: "Add keyframe here" })).toBeDisabled();
  });

  it("never lets the last keyframe of a region be removed", () => {
    render(<VideoTimeline {...props} regions={[stamp]} selectedId="r2" frame={0} />);
    expect(screen.getByRole("button", { name: "Remove keyframe" })).toBeDisabled();
  });

  it("shows where tracking lost an area and jumps there", () => {
    const onSeek = vi.fn();
    render(<VideoTimeline {...props} onSeek={onSeek} lost={[{ regionId: "r1", frame: 44 } as never]} />);
    const row = within(screen.getByRole("list", { name: "Keyframes by region" }));
    fireEvent.click(row.getByRole("button", { name: "Corner logo: tracking lost at frame 45" }));
    expect(onSeek).toHaveBeenCalledWith(44);
  });

  it("locks keyframe editing while tracking runs", () => {
    render(<VideoTimeline {...props} locked />);
    expect(screen.getByRole("button", { name: "Add keyframe here" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove keyframe" })).toBeDisabled();
  });
});

describe("video timeline: what the person can see about a selection", () => {
  const props = {
    frame: 10, frameCount: 60, fps: 30, playing: false, regions: [logo], selectedId: "r1", lost: [], tracked: false, locked: false,
    onSeek: vi.fn(), onTogglePlay: vi.fn(), onStep: vi.fn(), onAddKeyframe: vi.fn(), onRemoveKeyframe: vi.fn(), onFullscreen: vi.fn(),
  };

  it("says how long the selected region applies for", () => {
    render(<VideoTimeline {...props} />);
    expect(screen.getByText(/Corner logo is removed from 0:00.00 to 0:02 \(the whole video\)/)).toBeInTheDocument();
  });

  it("says when the region follows the picture", () => {
    render(<VideoTimeline {...props} tracked />);
    expect(screen.getByText(/following the picture \(tracked\)/)).toBeInTheDocument();
  });

  it("is honest that the editing preview has no sound, and still offers fullscreen", async () => {
    const onFullscreen = vi.fn();
    render(<VideoTimeline {...props} onFullscreen={onFullscreen} />);
    expect(screen.getByRole("button", { name: /Volume: the editing preview has no sound/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Fullscreen preview" }));
    expect(onFullscreen).toHaveBeenCalledTimes(1);
  });
});
