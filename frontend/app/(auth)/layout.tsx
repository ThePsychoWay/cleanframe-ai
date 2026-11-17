import type { Metadata } from "next";
import { AuthProvider } from "@/components/AuthProvider";
import { Logo } from "@/components/Logo";

export const metadata: Metadata = {
  robots: { index: false, follow: false },
};

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <header className="border-b border-line">
        <div className="mx-auto flex h-16 max-w-5xl items-center px-5">
          <Logo />
        </div>
      </header>
      <main id="main" className="mx-auto max-w-md px-5 py-16">
        {children}
      </main>
    </AuthProvider>
  );
}
