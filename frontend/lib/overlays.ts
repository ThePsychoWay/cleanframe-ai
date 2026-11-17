export type OverlayKey = "logo" | "text" | "timestamp" | "branding" | "object";

export interface UseCase {
  key: OverlayKey;
  label: string;
  description: string;
}

export const USE_CASES: readonly UseCase[] = [
  {
    key: "logo",
    label: "Logos",
    description:
      "A corner logo from an old channel, a retired brand, or a template you outgrew. Small static marks need the least reconstruction.",
  },
  {
    key: "text",
    label: "Text overlays",
    description:
      "A username, caption or channel name burned into the picture. The text is masked and the scene behind it is rebuilt.",
  },
  {
    key: "timestamp",
    label: "Timestamps",
    description:
      "The date and time stamped on older camera footage. Because the digits change, the mask stays put while the content under it keeps updating.",
  },
  {
    key: "branding",
    label: "Personal branding",
    description:
      "A lower third or name bar you added yourself and now want gone, for a re-edit or a different platform.",
  },
  {
    key: "object",
    label: "Unwanted objects",
    description:
      "A sign, pole or stray item in shot. Larger areas pull in more of the surrounding frame for context.",
  },
] as const;
