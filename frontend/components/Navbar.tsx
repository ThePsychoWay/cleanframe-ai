"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useOptionalAuth } from "./AuthProvider";
import { Logo } from "./Logo";
import { StepIndicator, type StepNumber } from "./StepIndicator";
import { useToast } from "./Toast";

interface NavbarProps {
  /** On the landing page the links scroll; elsewhere they go back to it. */
  landing?: boolean;
  /** Shows the Upload → Select → Restore → Export indicator in the middle. */
  step?: StepNumber;
  /** The active app page, so its link is marked. */
  active?: "projects" | "upload";
  className?: string;
}

/** The one top bar used everywhere outside the editor: logo left, a few links and the upload button right. */
export function Navbar({ landing = false, step, active, className = "" }: NavbarProps) {
  const auth = useOptionalAuth();
  const router = useRouter();
  const toast = useToast();
  const [leaving, setLeaving] = useState(false);
  const signedIn = auth?.status === "authenticated";

  const logout = async () => {
    if (!auth) return;
    setLeaving(true);
    try {
      await auth.logout();
      router.replace("/");
    } catch (error) {
      toast({ kind: "error", message: error instanceof Error ? error.message : "Couldn't sign out. Please try again." });
    } finally {
      setLeaving(false);
    }
  };

  const link = (isActive: boolean) =>
    `rounded-lg px-3 py-2 text-sm transition-colors duration-150 ${isActive ? "text-paper" : "text-dim hover:text-paper"}`;

  return (
    <header className={`sticky top-0 z-30 border-b border-line bg-ink/85 backdrop-blur ${className}`}>
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-3 px-4 sm:px-5">
        <Logo />
        {step && <StepIndicator current={step} className="ml-2 hidden md:block lg:ml-6" />}
        <nav aria-label="Primary" className="ml-auto flex items-center gap-0.5 sm:gap-1">
          <a href={landing ? "#how-it-works" : "/#how-it-works"} className={`${link(false)} hidden sm:inline-block`}>
            How it works
          </a>
          <Link href="/projects" className={link(active === "projects")} aria-current={active === "projects" ? "page" : undefined}>
            My projects
          </Link>
          {signedIn ? (
            <button type="button" onClick={() => void logout()} disabled={leaving} className={`${link(false)} hidden sm:inline-block`}>
              {leaving ? "Signing out…" : "Sign out"}
            </button>
          ) : (
            <Link href="/login" className={`${link(false)} hidden sm:inline-block`}>
              Sign in
            </Link>
          )}
          <Link href="/dashboard" className="btn btn-primary ml-1 !h-10 !rounded-xl !px-4 !text-sm" aria-current={active === "upload" ? "page" : undefined}>
            Upload video
          </Link>
        </nav>
      </div>
      {step && (
        <div className="border-t border-line px-4 py-2.5 md:hidden">
          <StepIndicator current={step} />
        </div>
      )}
    </header>
  );
}
