/**
 * Accepts a post-login destination from the URL only if it is a path on this site.
 * Anything else (other origins, protocol-relative URLs, backslash tricks) falls back,
 * so the login page can't be used as an open redirect.
 */
export function safeNextPath(next: string | null | undefined, fallback = "/dashboard"): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) return fallback;
  try {
    const parsed = new URL(next, "http://placeholder.invalid");
    if (parsed.origin !== "http://placeholder.invalid") return fallback;
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return fallback;
  }
}
