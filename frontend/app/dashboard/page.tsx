import type { Metadata } from "next";
import { Navbar } from "@/components/Navbar";
import { UploadDropzone } from "@/components/UploadDropzone";

export const metadata: Metadata = {
  title: "Upload a video",
  robots: { index: false, follow: false },
};

export default function DashboardPage() {
  return (
    <>
      <Navbar step={1} active="upload" />
      <main id="main" className="mx-auto max-w-6xl px-4 py-12 sm:px-5 sm:py-16">
        <UploadDropzone />
      </main>
    </>
  );
}
