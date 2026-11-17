"use client";

import type { OverlayKey } from "@/lib/overlays";
import { Overlay, type FrameAspect } from "./Overlay";
import { Scene } from "./Scene";
import { VideoComparisonSlider } from "./VideoComparisonSlider";

interface CompareFrameProps {
  aspect: FrameAspect;
  overlay: OverlayKey;
  autoSweep?: boolean;
  className?: string;
}

/** The drawn illustration used for the use-case tabs. It is not footage and not a result; the page says so. */
export function CompareFrame({ aspect, overlay, autoSweep = false, className = "" }: CompareFrameProps) {
  return (
    <VideoComparisonSlider
      className={className}
      aspect={aspect === "portrait" ? "9 / 16" : "16 / 9"}
      autoSweep={autoSweep}
      before={
        <>
          <Scene className="absolute inset-0 h-full w-full" />
          <Overlay kind={overlay} aspect={aspect} />
        </>
      }
      after={<Scene className="absolute inset-0 h-full w-full" />}
    />
  );
}
