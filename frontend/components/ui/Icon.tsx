import type { SVGProps } from "react";

/** One small icon set (24px grid, 1.75 stroke) so every control looks like it belongs to the same family. */
const PATHS = {
  upload: "M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3",
  rectangle: "M4 6h16v12H4z",
  polygon: "M12 3l8.5 6.2-3.2 10H6.7l-3.2-10z",
  brush: "M18.4 3.6a2.1 2.1 0 0 1 3 3L12 16l-4 1 1-4zM7 17c-1.5 0-3 1-3 3 2 0 4 0 5-1.5",
  eraser: "M7 21h12M5.6 14.4l8-8a2 2 0 0 1 2.8 0l2.2 2.2a2 2 0 0 1 0 2.8L11.4 18.6a2 2 0 0 1-1.4.6H8a2 2 0 0 1-1.4-.6l-1-1a2 2 0 0 1 0-2.8z",
  wand: "M15 4V2m0 14v-2m-7-5H6m14 0h-2M6.3 3.3l1.4 1.4m8.6 8.6l1.4 1.4M17.7 3.3l-1.4 1.4M3 21l9-9",
  cursor: "M5 3l14 7-6 2-2 6z",
  undo: "M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3",
  redo: "M15 14l5-5-5-5M20 9H10a6 6 0 0 0 0 12h3",
  reset: "M3 12a9 9 0 1 0 3-6.7M3 4v5h5",
  help: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM9.1 9a3 3 0 0 1 5.8 1c0 2-3 2.5-3 4.5M12 17.5v.01",
  play: "M7 4.5v15l12-7.5z",
  pause: "M7 4h3.5v16H7zM13.5 4H17v16h-3.5z",
  volume: "M11 5L6 9H3v6h3l5 4zM15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13",
  volumeOff: "M11 5L6 9H3v6h3l5 4zM22 9l-6 6m0-6l6 6",
  fullscreen: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  back: "M15 5l-7 7 7 7",
  trash: "M4 7h16M9 7V4h6v3m-8 0l1 13h8l1-13M10 11v5m4-5v5",
  check: "M5 12.5l4.5 4.5L19 7.5",
  close: "M6 6l12 12M18 6L6 18",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
  film: "M4 4h16v16H4zM8 4v16M16 4v16M4 9h4M4 15h4M16 9h4M16 15h4",
  download: "M12 4v12m0 0l-4.5-4.5M12 16l4.5-4.5M4 19h16",
  plus: "M12 5v14M5 12h14",
  chevron: "M6 9l6 6 6-6",
  zoomIn: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3M11 8v6M8 11h6",
  zoomOut: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3M8 11h6",
  info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 11v6M12 7.5v.01",
  alert: "M12 3l10 18H2zM12 10v5M12 18v.01",
  spinner: "M12 3a9 9 0 1 0 9 9",
  edit: "M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16z",
  frameBack: "M19 5l-8 7 8 7M7 5v14",
  frameForward: "M5 5l8 7-8 7M17 5v14",
} as const;

export type IconName = keyof typeof PATHS;

interface IconProps extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 20, className = "", ...rest }: IconProps) {
  const filled = name === "play" || name === "pause";
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={filled ? 0 : 1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={`shrink-0 ${className}`}
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
