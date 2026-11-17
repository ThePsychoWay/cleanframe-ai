import type { CSSProperties } from "react";
import type { OverlayKey } from "@/lib/overlays";

export type FrameAspect = "portrait" | "landscape";

/**
 * The "unwanted" element drawn on the original side of the comparison, wrapped in the
 * dashed selection the editor shows once a region has been masked.
 * Sizes use container-query units so they scale with the frame.
 */
export function Overlay({ kind, aspect }: { kind: OverlayKey; aspect: FrameAspect }) {
  const u = aspect === "portrait" ? 1 : 0.62;
  const cqw = (n: number) => `${(n * u).toFixed(2)}cqw`;

  const mask: CSSProperties = {
    outline: "1.5px dashed var(--color-mask)",
    outlineOffset: cqw(0.9),
    background: "color-mix(in srgb, var(--color-mask) 14%, transparent)",
    borderRadius: cqw(0.8),
  };

  switch (kind) {
    case "logo":
      return (
        <div
          className="absolute flex items-center"
          style={{ top: "6%", right: "7%", ...mask, padding: cqw(0.6) }}
        >
          <div
            style={{
              width: cqw(6.5),
              height: cqw(6.5),
              borderRadius: cqw(1.6),
              background: "rgba(255,255,255,0.86)",
              display: "grid",
              placeItems: "center",
            }}
          >
            <div style={{ width: cqw(2.6), height: cqw(2.6), borderRadius: "9999px", background: "#2a3568" }} />
          </div>
          <span
            style={{
              marginLeft: cqw(1.6),
              fontSize: cqw(4.2),
              fontWeight: 700,
              letterSpacing: "0.1em",
              color: "rgba(255,255,255,0.86)",
            }}
          >
            HALCYON
          </span>
        </div>
      );

    case "text":
      return (
        <div
          className="absolute"
          style={{
            bottom: "10%",
            left: "6%",
            ...mask,
            padding: `${cqw(0.4)} ${cqw(1.2)}`,
            fontSize: cqw(5),
            fontWeight: 600,
            color: "rgba(255,255,255,0.8)",
            textShadow: "0 1px 6px rgba(0,0,0,0.45)",
          }}
        >
          @yourchannel
        </div>
      );

    case "timestamp":
      return (
        <div
          className="tnum absolute"
          style={{
            top: "5%",
            left: "6%",
            ...mask,
            padding: `${cqw(0.4)} ${cqw(1.2)}`,
            fontSize: cqw(4.2),
            fontWeight: 600,
            letterSpacing: "0.04em",
            color: "#ffb347",
            textShadow: "0 1px 4px rgba(0,0,0,0.6)",
          }}
        >
          2019-08-14 18:42:07
        </div>
      );

    case "branding":
      return (
        <div
          className="absolute"
          style={{
            bottom: "9%",
            left: 0,
            width: "64%",
            ...mask,
            borderRadius: `0 ${cqw(0.8)} ${cqw(0.8)} 0`,
            padding: `${cqw(1.6)} ${cqw(3)}`,
            background:
              "linear-gradient(90deg, rgba(0,0,0,0.62), rgba(0,0,0,0.4)), color-mix(in srgb, var(--color-mask) 14%, transparent)",
          }}
        >
          <div style={{ fontSize: cqw(5.2), fontWeight: 700, color: "#fff" }}>Alex Rivera</div>
          <div style={{ fontSize: cqw(3.4), color: "rgba(255,255,255,0.75)" }}>Travel films</div>
        </div>
      );

    case "object":
      return (
        <div
          className="absolute"
          style={{ left: "63%", top: "52%", height: "26%", width: cqw(8), ...mask }}
        >
          <div
            style={{
              position: "absolute",
              top: 0,
              left: "50%",
              transform: "translateX(-50%)",
              width: cqw(7),
              height: cqw(4),
              background: "#0b1020",
              borderRadius: cqw(0.5),
            }}
          />
          <div
            style={{
              position: "absolute",
              top: cqw(4),
              bottom: 0,
              left: "50%",
              transform: "translateX(-50%)",
              width: cqw(1.2),
              background: "#0b1020",
            }}
          />
        </div>
      );
  }
}
