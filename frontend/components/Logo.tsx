import Link from "next/link";

/** Crop-mark frame with one corner missing: the spot where an overlay used to be. */
export function LogoMark({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="24"
      height="24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      <path d="M3 9V5a2 2 0 0 1 2-2h4" />
      <path d="M15 3h4a2 2 0 0 1 2 2v4" />
      <path d="M21 15v4a2 2 0 0 1-2 2h-4" />
      <rect
        x="3.5"
        y="15.5"
        width="5"
        height="5"
        rx="1"
        stroke="var(--color-mask)"
        strokeWidth="1.5"
        strokeDasharray="2 2"
      />
    </svg>
  );
}

export function Logo() {
  return (
    <Link href="/" className="inline-flex items-center gap-2.5 rounded-sm" aria-label="CleanFrame AI home">
      <LogoMark />
      <span className="text-[1.0625rem] font-semibold tracking-tight">
        CleanFrame
      </span>
    </Link>
  );
}
