import type { Metadata } from "next";
import { AuthGate } from "@/components/AuthGate";
import { AuthProvider } from "@/components/AuthProvider";

export const metadata: Metadata = {
  title: "Processing",
  robots: { index: false, follow: false },
};

export default function ProcessLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <AuthGate>{children}</AuthGate>
    </AuthProvider>
  );
}
