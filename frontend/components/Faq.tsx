import { OWNERSHIP_NOTICE } from "@/lib/site";

const FAQS = [
  {
    q: "What video formats are supported?",
    a: "MP4, MOV, M4V, WebM, MKV and AVI, at 720p, 1080p, 1440p or 4K. File size and length limits apply, and you'll see them before you upload.",
  },
  {
    q: "How long does processing take?",
    a: "It depends on the clip's length, its resolution and the size of the area being rebuilt. Small static logos finish fastest; large or moving overlays take longer. While it runs you see real progress by stage and frame count, not a timer.",
  },
  {
    q: "Does the original video get modified?",
    a: "No. Your upload is kept as it was, and processing writes a new file.",
  },
  {
    q: "Is my video stored?",
    a: "Temporarily. Uploads and results are deleted automatically once the retention period ends, and downloads use signed links that expire.",
  },
  {
    q: "What happens to uploaded files?",
    a: "Only your account can open them. They're used to create your result, then removed when the retention period ends or when you delete the project, whichever comes first.",
  },
  {
    q: "Whose videos can I use this on?",
    a: OWNERSHIP_NOTICE,
  },
] as const;

export function Faq() {
  return (
    <section id="faq" aria-labelledby="faq-heading" className="scroll-mt-16 border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-12 px-5 py-24 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.7fr)] lg:gap-20">
        <h2 id="faq-heading" className="heading text-[clamp(1.75rem,3.6vw,2.75rem)]">
          Questions
        </h2>
        <div className="divide-y divide-line border-y border-line">
          {FAQS.map((item) => (
            <details key={item.q} className="group py-5">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-6 text-left font-semibold [&::-webkit-details-marker]:hidden">
                {item.q}
                <svg
                  viewBox="0 0 24 24"
                  width="20"
                  height="20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                  className="shrink-0 text-faint transition-transform group-open:rotate-45"
                >
                  <path d="M12 5v14M5 12h14" />
                </svg>
              </summary>
              <p className="mt-3 max-w-xl leading-relaxed text-dim">{item.a}</p>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}
