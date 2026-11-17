"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";
import { useAuth } from "./AuthProvider";

/**
 * Keeps signed-out visitors off app pages. This is a convenience for the person using the
 * site; the API enforces access on every request regardless of what the page shows.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const { status, errorMessage, refresh } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (status === "anonymous") router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [status, pathname, router]);

  if (status === "authenticated") return <>{children}</>;

  if (status === "error") {
    return (
      <div className="mx-auto max-w-md px-5 py-24" role="alert">
        <h1 className="heading text-2xl">We couldn&apos;t load your workspace</h1>
        <p className="mt-3 text-dim">{errorMessage}</p>
        <button type="button" className="btn btn-primary mt-6" onClick={() => void refresh()}>
          Try again
        </button>
      </div>
    );
  }

  return (
    <p role="status" className="px-5 py-24 text-center text-dim">
      {status === "anonymous" ? "Taking you to log in…" : "Loading your workspace…"}
    </p>
  );
}
