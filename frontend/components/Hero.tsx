import Link from "next/link";
import { SampleCompare } from "./SampleCompare";

export function Hero() {
  return (
    <section
      aria-labelledby="hero-heading"
      className="mx-auto grid max-w-6xl items-center gap-12 px-5 pb-20 pt-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,24rem)] lg:gap-20 lg:pb-28 lg:pt-20"
    >
      <div>
        <h1 id="hero-heading" className="display text-[clamp(2.5rem,6vw,4.25rem)]">
          Your video.
          <br />
          <span className="text-mask">Cleaned up.</span>
        </h1>
        <p className="mt-6 max-w-[32rem] text-lg leading-relaxed text-dim">
          Remove unwanted logos, text, and visual overlays from your videos with AI-powered restoration.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link href="/dashboard" className="btn btn-primary !h-12 !px-6">
            Upload a video
          </Link>
          <a href="#how-it-works" className="btn btn-ghost !h-12 !px-6">
            See how it works
          </a>
        </div>
        <p className="mt-5 text-sm text-faint">MP4, MOV, WebM and AVI · 720p to 4K · Your original video stays unchanged</p>
      </div>

      <SampleCompare className="mx-auto w-full max-w-[22rem] lg:max-w-none" />
    </section>
  );
}
