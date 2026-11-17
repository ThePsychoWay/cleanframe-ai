import { useId } from "react";

/**
 * Illustrative lake-at-dusk scene used for the comparison demos.
 * Drawn to be cropped to either 9:16 or 16:9 (slice), so key shapes sit near the centre.
 * This is an illustration, not footage and not a processing result.
 */
export function Scene({ className = "" }: { className?: string }) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const sky = `sky${uid}`;
  const lake = `lake${uid}`;
  const glow = `glow${uid}`;

  return (
    <svg
      viewBox="0 0 960 960"
      preserveAspectRatio="xMidYMid slice"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <linearGradient id={sky} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#141d42" />
          <stop offset="0.45" stopColor="#4a5c9c" />
          <stop offset="0.75" stopColor="#e39a6c" />
          <stop offset="1" stopColor="#f6c48a" />
        </linearGradient>
        <linearGradient id={lake} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#d99a72" />
          <stop offset="0.35" stopColor="#4b5686" />
          <stop offset="1" stopColor="#141b3a" />
        </linearGradient>
        <radialGradient id={glow}>
          <stop offset="0" stopColor="#ffd9a0" stopOpacity="0.55" />
          <stop offset="1" stopColor="#ffd9a0" stopOpacity="0" />
        </radialGradient>
      </defs>

      <rect width="960" height="640" fill={`url(#${sky})`} />
      <circle cx="480" cy="480" r="230" fill={`url(#${glow})`} />
      <circle cx="480" cy="480" r="74" fill="#ffe4b8" />

      <path
        d="M0 575 L110 488 L205 540 L335 430 L465 548 L600 452 L735 540 L850 466 L960 530 L960 650 L0 650 Z"
        fill="#38457d"
      />
      <path d="M0 610 C150 556 260 604 400 584 S700 546 960 604 L960 660 L0 660 Z" fill="#242d5a" />

      <rect y="640" width="960" height="320" fill={`url(#${lake})`} />
      <ellipse cx="480" cy="690" rx="78" ry="9" fill="#ffd9a0" opacity="0.55" />
      <ellipse cx="480" cy="716" rx="52" ry="6" fill="#ffd9a0" opacity="0.35" />
      <ellipse cx="480" cy="742" rx="30" ry="4" fill="#ffd9a0" opacity="0.2" />

      <path d="M0 808 C110 764 220 796 310 840 L310 960 L0 960 Z" fill="#0e142e" />
      <path d="M960 790 C860 756 770 790 690 836 L690 960 L960 960 Z" fill="#0e142e" />
      <path d="M40 800 L66 700 L92 800 Z M78 804 L108 676 L138 804 Z M120 812 L144 724 L168 812 Z" fill="#0a1026" />
      <path d="M820 790 L848 690 L876 790 Z M868 800 L898 668 L928 800 Z" fill="#0a1026" />
    </svg>
  );
}
