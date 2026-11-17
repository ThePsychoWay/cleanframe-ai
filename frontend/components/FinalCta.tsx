import Link from "next/link";

export function FinalCta() {
  return (
    <section aria-labelledby="cta-heading" className="border-t border-line">
      <div className="mx-auto flex max-w-6xl flex-col items-start gap-8 px-5 py-24 sm:flex-row sm:items-end sm:justify-between">
        <h2 id="cta-heading" className="display text-[clamp(2rem,4.5vw,3.25rem)]">
          Your video. Cleaned up.
        </h2>
        <Link href="/dashboard" className="btn btn-primary !h-12 !px-6">
          Upload a video
        </Link>
      </div>
    </section>
  );
}
