const FEATURES = [
  {
    name: "AI inpainting",
    text: "Rebuilds the area behind the overlay from the surrounding scene. Blur is never the default and is only used if you choose it.",
  },
  {
    name: "Automatic detection",
    text: "Suggests likely logos, text and timestamps with a confidence score. You approve or edit every suggestion before anything runs.",
  },
  {
    name: "Manual masking",
    text: "Rectangle, brush, eraser and polygon tools, with as many regions as the clip needs.",
  },
  {
    name: "Watermark tracking",
    text: "Follows an overlay that moves. Add a keyframe wherever the mask drifts and the frames in between are filled in.",
  },
  {
    name: "Multiple watermark removal",
    text: "Take out a logo, a username and a timestamp in a single job.",
  },
  {
    name: "High-quality export",
    text: "Keeps the original resolution, frame rate and aspect ratio. Choose Fast, Balanced or High quality.",
  },
  {
    name: "Audio preservation",
    text: "The original audio track is copied back rather than re-encoded wherever the container allows, so sound stays in sync.",
  },
] as const;

export function Features() {
  return (
    <section id="features" aria-labelledby="features-heading" className="scroll-mt-16 border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-12 px-5 py-24 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.7fr)] lg:gap-20">
        <div>
          <h2 id="features-heading" className="heading text-[clamp(1.75rem,3.6vw,2.75rem)] lg:sticky lg:top-28">
            What the tool does
          </h2>
        </div>
        <dl className="divide-y divide-line border-y border-line">
          {FEATURES.map((f) => (
            <div key={f.name} className="grid gap-1.5 py-6 sm:grid-cols-[13rem_minmax(0,1fr)] sm:gap-8">
              <dt className="font-semibold">{f.name}</dt>
              <dd className="max-w-lg leading-relaxed text-dim">{f.text}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
