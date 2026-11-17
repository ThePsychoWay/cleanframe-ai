"use client";

import { formatDuration } from "@/lib/video";
import { trackingFraction } from "@/lib/tracking";
import type { LostArea, TrackingResponse } from "@/types/tracking";

interface TrackingPanelProps {
  tracking: TrackingResponse;
  fps: number;
  frameCount: number;
  regionCount: number;
  /** The marks on screen differ from the saved ones, so any results no longer match them. */
  unsaved: boolean;
  saving: boolean;
  /** Results that describe the marks as they are now. */
  lost: LostArea[];
  resultsCurrent: boolean;
  /** Why tracking can't start right now (a render is running, or the finished video is locked), or null. */
  blockedReason?: string | null;
  onTrack: () => void;
  onGoToFrame: (frame: number) => void;
}

function timeLabel(frame: number, fps: number): string {
  const seconds = frame / fps;
  return `${formatDuration(seconds)}.${String(Math.floor((seconds % 1) * 100)).padStart(2, "0")}`;
}

export function TrackingPanel(props: TrackingPanelProps) {
  const { tracking, fps, frameCount, regionCount, unsaved, saving, lost, resultsCurrent, blockedReason = null, onTrack, onGoToFrame } = props;
  const running = tracking.status === "running";
  const fraction = trackingFraction(tracking.frames_done, tracking.frames_total);
  const percent = fraction === null ? null : Math.round(fraction * 100);
  const hasResults = tracking.status === "done" && tracking.data !== null;

  const buttonLabel = running ? "Tracking…" : unsaved ? "Save and track" : hasResults ? "Track again" : "Track regions";
  const disabledReason = regionCount === 0 ? "Mark an area first" : (blockedReason ?? undefined);

  return (
    <section aria-labelledby="tracking-heading" className="surface p-4">
      <h2 id="tracking-heading" className="heading text-lg">
        Follow movement
      </h2>
      <p className="mt-2 text-sm text-dim">
        Only needed if the mark moves. Follows each selected area through the video from its keyframes, so a moving logo stays covered.
      </p>

      <button
        type="button"
        className="btn btn-ghost mt-3 !h-10 !px-4 !text-sm"
        onClick={onTrack}
        disabled={running || saving || regionCount === 0 || blockedReason !== null}
        title={disabledReason}
      >
        {saving ? "Saving…" : buttonLabel}
      </button>

      {running && (
        <div className="mt-4" role="status">
          {/* A bar only when the job has reported its size; before that, just say it has started. */}
          {percent === null ? (
            <p className="text-sm text-dim">Starting…</p>
          ) : (
            <>
              <div
                role="progressbar"
                aria-label="Tracking progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={percent}
                className="h-2 overflow-hidden rounded-full bg-raised"
              >
                <div className="h-full rounded-full bg-mask transition-[width]" style={{ width: `${percent}%` }} />
              </div>
              <p className="tnum mt-2 text-sm text-dim">
                {percent}% · frame {Math.min(tracking.frames_done, tracking.frames_total)} of {tracking.frames_total}
              </p>
            </>
          )}
          <p className="mt-2 text-xs text-faint">Your marks are locked until tracking finishes. You can leave this page; it keeps going.</p>
        </div>
      )}

      {tracking.status === "failed" && (
        <p role="alert" className="mt-4 text-sm text-paper">
          {tracking.error ?? "Tracking couldn't finish."} Your marks are unchanged.
        </p>
      )}

      {hasResults && !resultsCurrent && (
        <p className="mt-4 text-sm text-paper">
          {unsaved
            ? "You've changed the marks since tracking, so the preview shows keyframes only. Track again to follow the new marks."
            : "These results no longer match the marks. Track again."}
        </p>
      )}

      {hasResults && resultsCurrent && (
        <div className="mt-4 space-y-3">
          {lost.length === 0 ? (
            <p className="text-sm text-paper">
              Followed {regionCount === 1 ? "the region" : `all ${regionCount} regions`} through {frameCount} frames. Play the video
              to check that the marks stay on target.
            </p>
          ) : (
            <>
              <p className="text-sm text-paper">
                Tracking lost {lost.length === 1 ? "one area" : `${lost.length} areas`}. Until you correct it, the mark stays where it
                last was.
              </p>
              <ul className="space-y-2">
                {lost.map((area) => (
                  <li key={`${area.regionId}-${area.frame}`} className="rounded-xl border border-line p-2.5 text-sm">
                    <p className="font-medium">
                      {area.regionName} <span className="tnum font-normal text-dim">at {timeLabel(area.frame, fps)} · frame {area.frame + 1}</span>
                    </p>
                    <p className="mt-0.5 text-dim">{area.reason}</p>
                    <button type="button" className="btn btn-ghost mt-2 !h-8 !px-3 !text-xs" onClick={() => onGoToFrame(area.frame)}>
                      Go to frame {area.frame + 1}
                    </button>
                  </li>
                ))}
              </ul>
              <p className="text-xs text-faint">
                Go to the frame, move or reshape the mark so it covers the logo (that makes a keyframe), then track again.
              </p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
