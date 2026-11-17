"use client";

import { useId, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { MODE_EXPLANATION, QUALITY_EXPLANATION, type ProcessingMode } from "@/lib/render";
import { targetLabel, type OutputHeight } from "@/lib/resolution";
import type { EngineInfo, RenderSettings } from "@/types/render";

interface RemovalSettingsProps {
  settings: RenderSettings;
  mode: ProcessingMode;
  onSettings: (settings: RenderSettings) => void;
  onMode: (mode: ProcessingMode) => void;
  engine: EngineInfo | null;
  engineProblem: string | null;
  /** Only sizes smaller than the original are offered. */
  heights: OutputHeight[];
  hasAudio: boolean;
  acknowledged: boolean;
  onAcknowledged: (value: boolean) => void;
  disabled: boolean;
}

function Seg<T extends string>({
  name,
  legend,
  value,
  options,
  onChange,
  disabled,
}: {
  name: string;
  legend: string;
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <fieldset disabled={disabled}>
      <legend className="mb-1.5 text-sm font-medium">{legend}</legend>
      <div className="seg">
        {options.map((o) => (
          <label key={o.value}>
            <input type="radio" name={name} value={o.value} checked={value === o.value} onChange={() => onChange(o.value)} />
            {o.label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function RemovalSettings(props: RemovalSettingsProps) {
  const { settings, mode, engine, engineProblem, heights, hasAudio, acknowledged, disabled } = props;
  const uid = useId();
  const [advanced, setAdvanced] = useState(false);
  const quality = settings.quality === "high" ? "high" : "balanced";
  const outside = Boolean(engine?.sends_frames_to_third_party);

  return (
    <section aria-labelledby={`${uid}-heading`} className="space-y-5">
      <h2 id={`${uid}-heading`} className="heading text-lg">
        Removal settings
      </h2>

      <div>
        <Seg
          name={`${uid}-quality`}
          legend="Quality"
          value={quality}
          disabled={disabled || mode === "fast"}
          onChange={(q) => props.onSettings({ ...settings, quality: q })}
          options={[
            { value: "balanced", label: "Balanced" },
            { value: "high", label: "High quality" },
          ]}
        />
        <p className="mt-1.5 text-xs leading-relaxed text-faint">
          {mode === "fast" ? "Not used while Fast processing is on." : QUALITY_EXPLANATION[quality]}
        </p>
      </div>

      <div>
        <Seg
          name={`${uid}-mode`}
          legend="Processing mode"
          value={mode}
          disabled={disabled}
          onChange={props.onMode}
          options={[
            { value: "ai", label: "AI reconstruction" },
            { value: "fast", label: "Fast processing" },
          ]}
        />
        <p className="mt-1.5 text-xs leading-relaxed text-faint">{MODE_EXPLANATION[mode]}</p>
      </div>

      <div>
        <p className="mb-1.5 text-sm font-medium">Audio</p>
        {hasAudio ? (
          <label className="flex min-h-11 cursor-pointer items-center gap-3 rounded-xl border border-line bg-ink px-3 text-sm transition-colors hover:border-[#3f3f43]">
            <input
              type="checkbox"
              className="size-4 accent-[var(--color-mask)]"
              checked={settings.keep_audio}
              disabled={disabled}
              onChange={(e) => props.onSettings({ ...settings, keep_audio: e.target.checked })}
            />
            Preserve original audio
          </label>
        ) : (
          <p className="rounded-xl border border-line bg-ink px-3 py-3 text-sm text-dim">This video has no audio track.</p>
        )}
        <p className="mt-1.5 text-xs leading-relaxed text-faint">
          {hasAudio ? "Your sound is copied across unchanged where the format allows, so only the picture is rebuilt." : "Nothing to keep."}
        </p>
      </div>

      <div className="rounded-xl border border-line">
        <button
          type="button"
          aria-expanded={advanced}
          aria-controls={`${uid}-advanced`}
          onClick={() => setAdvanced((v) => !v)}
          className="flex min-h-11 w-full items-center justify-between rounded-xl px-3 text-sm font-medium transition-colors hover:bg-raised"
        >
          Advanced settings
          <Icon name="chevron" size={18} className={`text-dim transition-transform duration-200 ${advanced ? "rotate-180" : ""}`} />
        </button>
        {advanced && (
          <div id={`${uid}-advanced`} className="animate-fade-up space-y-4 border-t border-line p-3">
            <div className="grid grid-cols-2 gap-3">
              <label className="block text-sm text-dim">
                Format
                <select
                  className="field mt-1 w-full"
                  disabled={disabled}
                  value={settings.container}
                  onChange={(e) => props.onSettings({ ...settings, container: e.target.value as RenderSettings["container"] })}
                >
                  <option value="mp4">MP4</option>
                  <option value="webm">WebM</option>
                </select>
              </label>
              <label className="block text-sm text-dim">
                Size
                <select
                  className="field mt-1 w-full"
                  disabled={disabled}
                  value={settings.output_height ?? ""}
                  onChange={(e) => props.onSettings({ ...settings, output_height: e.target.value === "" ? null : (Number(e.target.value) as OutputHeight) })}
                >
                  <option value="">Original</option>
                  {heights.map((h) => (
                    <option key={h} value={h}>
                      {targetLabel(h)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {engineProblem && <p className="text-sm text-paper">{engineProblem}</p>}
            {engine && (
              <div className="space-y-1 text-xs leading-relaxed text-faint">
                <p>
                  <span className="text-dim">Engine:</span> {engine.name}
                  {engine.device ? ` · ${engine.device.name}` : ""}
                </p>
                {!engine.development_only && <p>{engine.expectation}</p>}
                {!engine.available && <p className="text-paper">Not ready: {engine.reason ?? "this engine isn't set up."}</p>}
              </div>
            )}
          </div>
        )}
      </div>

      {/* These two are never hidden in the collapsed section: the person must see them before starting. */}
      {engine?.development_only && (
        <p role="note" className="rounded-xl border border-line bg-raised px-3 py-2.5 text-sm text-paper">
          Development engine: it blurs the marked area instead of rebuilding it. This is not an AI restoration.
        </p>
      )}
      {outside && (
        <label className="flex items-start gap-3 rounded-xl border border-line bg-raised px-3 py-3 text-sm text-paper">
          <input
            type="checkbox"
            className="mt-0.5 size-4 shrink-0 accent-[var(--color-mask)]"
            checked={acknowledged}
            onChange={(e) => props.onAcknowledged(e.target.checked)}
            disabled={disabled}
          />
          <span>
            I understand that my video frames are sent to an outside service ({engine?.name}) to be processed, and I have the right to process
            this video.
          </span>
        </label>
      )}
    </section>
  );
}
