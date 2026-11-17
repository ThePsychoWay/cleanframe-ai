"use client";

import { Icon } from "@/components/ui/Icon";
import { Tip } from "@/components/ui/Tip";
import { hasKeyframeAt } from "@/lib/masks";
import { formatDuration } from "@/lib/video";
import type { Region } from "@/types/mask";
import type { LostArea } from "@/types/tracking";

interface VideoTimelineProps {
  frame: number;
  frameCount: number;
  fps: number;
  playing: boolean;
  regions: Region[];
  selectedId: string | null;
  /** Where tracking gave up, marked on each region's row. */
  lost: LostArea[];
  /** True once tracking results match the marks on screen, so the bars can say "tracked". */
  tracked: boolean;
  /** Marks can't change while tracking or processing runs. */
  locked: boolean;
  onSeek: (frame: number) => void;
  onTogglePlay: () => void;
  onStep: (delta: number) => void;
  onAddKeyframe: () => void;
  onRemoveKeyframe: () => void;
  onFullscreen: () => void;
}

function clock(frame: number, fps: number): string {
  const seconds = frame / fps;
  return `${formatDuration(seconds)}.${String(Math.floor((seconds % 1) * 100)).padStart(2, "0")}`;
}

export function VideoTimeline(props: VideoTimelineProps) {
  const { frame, frameCount, fps, playing, regions, selectedId, lost, tracked, locked, onSeek } = props;
  const last = Math.max(1, frameCount - 1);
  const selected = regions.find((r) => r.id === selectedId);
  const onKeyframe = selected ? hasKeyframeAt(selected, frame) : false;
  const total = frameCount / fps;

  return (
    <section aria-label="Timeline" className="surface p-3 sm:p-4">
      <div className="flex flex-wrap items-center gap-2 sm:gap-3">
        <button
          type="button"
          className="btn btn-primary btn-icon !rounded-full"
          onClick={props.onTogglePlay}
          aria-keyshortcuts="Space"
          aria-label={playing ? "Pause" : "Play"}
        >
          <Icon name={playing ? "pause" : "play"} size={18} />
        </button>
        <Tip label="Back one frame (←)" side="top">
          <button type="button" className="btn btn-quiet btn-icon" onClick={() => props.onStep(-1)} aria-keyshortcuts="ArrowLeft" aria-label="Back one frame">
            <Icon name="frameBack" size={18} />
          </button>
        </Tip>
        <Tip label="Forward one frame (→)" side="top">
          <button type="button" className="btn btn-quiet btn-icon" onClick={() => props.onStep(1)} aria-keyshortcuts="ArrowRight" aria-label="Forward one frame">
            <Icon name="frameForward" size={18} />
          </button>
        </Tip>

        <p className="tnum text-sm" aria-live="off">
          <span className="text-paper">{clock(frame, fps)}</span>
          <span className="text-faint"> / {formatDuration(total)}</span>
          <span className="ml-2 hidden text-xs text-faint sm:inline">
            frame {frame + 1} of {frameCount}
          </span>
        </p>

        <span className="ml-auto flex items-center gap-1">
          <Tip label="The editing preview has no sound. Your original audio is kept in the result." side="top">
            <button type="button" className="btn btn-quiet btn-icon" disabled aria-label="Volume: the editing preview has no sound">
              <Icon name="volumeOff" size={18} />
            </button>
          </Tip>
          <Tip label="Fullscreen preview" side="top">
            <button type="button" className="btn btn-quiet btn-icon" onClick={props.onFullscreen} aria-label="Fullscreen preview">
              <Icon name="fullscreen" size={18} />
            </button>
          </Tip>
        </span>
      </div>

      <div className="mt-2">
        <input
          type="range"
          min={0}
          max={last}
          step={1}
          value={frame}
          onChange={(e) => onSeek(Number(e.target.value))}
          aria-label="Video position"
          aria-valuetext={`Frame ${frame + 1} of ${frameCount}`}
          className="range block h-6 w-full cursor-pointer"
        />

        {regions.length > 0 && (
          <ul className="mt-2 space-y-1.5" aria-label="Keyframes by region">
            {regions.map((region) => {
              const isSelected = region.id === selectedId;
              return (
                <li key={region.id} className="flex items-center gap-3">
                  <span className={`w-20 shrink-0 truncate text-xs sm:w-28 ${isSelected ? "font-semibold text-paper" : "text-faint"}`} title={region.name}>
                    {region.name}
                  </span>
                  {/* A region applies to the whole video, so its bar spans the full length. */}
                  <div className={`relative h-6 flex-1 rounded-full transition-colors duration-200 ${isSelected ? "bg-mask/20 ring-1 ring-mask/50" : "bg-raised"}`}>
                    {region.keyframes.map((k) => (
                      <button
                        key={k.frame}
                        type="button"
                        onClick={() => onSeek(k.frame)}
                        aria-label={`${region.name}: keyframe at frame ${k.frame + 1}`}
                        title={`Frame ${k.frame + 1}`}
                        className={`absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rotate-45 rounded-[3px] border transition-transform duration-150 hover:scale-125 ${
                          k.frame === frame ? "border-paper bg-paper" : "border-mask bg-mask/70 hover:bg-mask"
                        }`}
                        style={{ left: `${(k.frame / last) * 100}%` }}
                      />
                    ))}
                    {lost
                      .filter((area) => area.regionId === region.id)
                      .map((area) => (
                        <button
                          key={`lost-${area.frame}`}
                          type="button"
                          onClick={() => onSeek(area.frame)}
                          aria-label={`${region.name}: tracking lost at frame ${area.frame + 1}`}
                          title={`Tracking lost at frame ${area.frame + 1}`}
                          className="absolute top-1/2 h-5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-danger ring-2 ring-ink"
                          style={{ left: `${(area.frame / last) * 100}%` }}
                        />
                      ))}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
        {selected && (
          <p className="tnum mt-2 text-xs text-dim">
            {selected.name} is removed from {clock(0, fps)} to {formatDuration(total)}
            {tracked ? (lost.some((l) => l.regionId === selected.id) ? ", following the picture except where tracking was lost." : ", following the picture (tracked).") : " (the whole video)."}
          </p>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="btn btn-ghost !h-9 !px-3.5 !text-sm"
          onClick={props.onAddKeyframe}
          disabled={locked || !selected || onKeyframe}
          title={!selected ? "Select a region first" : onKeyframe ? "This frame already has a keyframe" : "Pin the selection to where it is on this frame"}
        >
          Add keyframe here
        </button>
        <button
          type="button"
          className="btn btn-ghost !h-9 !px-3.5 !text-sm"
          onClick={props.onRemoveKeyframe}
          disabled={locked || !selected || !onKeyframe || selected.keyframes.length <= 1}
          title={selected && onKeyframe && selected.keyframes.length <= 1 ? "A region needs at least one keyframe" : undefined}
        >
          Remove keyframe
        </button>
        <p className="min-w-0 basis-full text-xs text-faint sm:flex-1 sm:basis-0">Moved or reshaped the selection on a later frame? That frame becomes a keyframe, so the selection can follow a moving mark.</p>
      </div>
    </section>
  );
}
