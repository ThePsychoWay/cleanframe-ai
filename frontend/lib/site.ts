export const SITE = {
  name: "CleanFrame AI",
  title: "AI Video Watermark Remover | CleanFrame AI",
  description:
    "AI-powered video restoration for removing unwanted visual overlays from videos you own or have permission to edit.",
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000",
} as const;

export const OWNERSHIP_NOTICE =
  "Only remove overlays from content you own or have permission to edit.";

export const PRIVACY_NOTICE =
  "Your uploaded videos are processed securely and temporary processing files are automatically deleted according to our retention policy.";
