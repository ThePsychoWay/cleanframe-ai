const STEPS = [
  {
    title: "Upload",
    body: "Drop in an MP4, MOV, M4V, WebM, MKV or AVI (720p up to 4K) that you own or have permission to edit. Resolution, frame rate and duration are read first, so you know what you're working with.",
  },
  {
    title: "Select",
    body: "Mark the overlay with a rectangle, brush or polygon, or let automatic detection suggest it, which you confirm. If it moves, add keyframes. Nothing changes until you approve the mask.",
  },
  {
    title: "Restore",
    body: "The area behind the overlay is rebuilt from the surrounding scene, with smoothing across frames so it doesn't flicker. Your original audio is added back untouched.",
  },
  {
    title: "Export",
    body: "Compare the result with your original, then download it. Your original upload is never changed; the result is a new file.",
  },
] as const;

export function HowItWorks() {
  return (
    <section id="how-it-works" aria-labelledby="how-heading" className="scroll-mt-16 border-t border-line">
      <div className="mx-auto max-w-6xl px-5 py-24">
        <h2 id="how-heading" className="heading max-w-2xl text-[clamp(1.75rem,3.6vw,2.75rem)]">
          From upload to a clean export in four steps
        </h2>

        <ol className="mt-14 grid gap-12 md:grid-cols-4 md:gap-0">
          {STEPS.map((step, i) => (
            <li key={step.title} className="md:border-l md:border-line md:px-6 md:first:border-l-0 md:first:pl-0">
              <span className="tnum block text-4xl font-light text-faint" aria-hidden="true">
                {i + 1}
              </span>
              <h3 className="mt-5 text-xl font-semibold">
                <span className="sr-only">Step {i + 1}: </span>
                {step.title}
              </h3>
              <p className="mt-3 max-w-sm leading-relaxed text-dim">{step.body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
