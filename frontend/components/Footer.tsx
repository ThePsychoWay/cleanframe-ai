import { OWNERSHIP_NOTICE } from "@/lib/site";
import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-5 py-10 sm:flex-row sm:items-center sm:justify-between">
        <Logo />
        <p className="max-w-md text-sm text-faint">{OWNERSHIP_NOTICE}</p>
        <p className="text-sm text-faint">&copy; 2026 CleanFrame AI</p>
      </div>
    </footer>
  );
}
