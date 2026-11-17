/** @type {import('next').NextConfig} */

const production = process.env.NODE_ENV === "production";
const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
// "same-origin": the browser talks to this site, which forwards /api/* to API_PROXY_TARGET (see rewrites below).
const sameOriginApi = configuredApiUrl === "same-origin";
const apiProxyTarget = process.env.API_PROXY_TARGET?.trim().replace(/\/+$/, "");
if (sameOriginApi && !apiProxyTarget) {
  throw new Error('NEXT_PUBLIC_API_URL=same-origin needs API_PROXY_TARGET (the API server\'s https address).');
}
const apiOrigin = sameOriginApi ? "'self'" : new URL(configuredApiUrl).origin;
// Where the browser sends upload parts to and plays videos from (the S3-compatible store's public address).
// Set it in production to keep the policy tight; without it any https address is allowed for those two things.
const storageOrigin = process.env.NEXT_PUBLIC_STORAGE_ORIGIN?.trim();
const storage = storageOrigin ? [new URL(storageOrigin).origin] : production ? ["https:"] : ["http://localhost:*", "http://127.0.0.1:*"];

const policy = [
  "default-src 'self'",
  // Next.js writes small inline scripts into every page; a nonce-based policy would make every page dynamic.
  `script-src 'self' 'unsafe-inline'${production ? "" : " 'unsafe-eval'"}`,
  "style-src 'self' 'unsafe-inline'",
  `img-src 'self' data: blob: ${storage.join(" ")}`,
  `media-src 'self' blob: ${storage.join(" ")}`,
  `connect-src 'self' ${apiOrigin} ${storage.join(" ")}`,
  "font-src 'self' data:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: policy },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), payment=()" },
  ...(production ? [{ key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains" }] : []),
];

const nextConfig = {
  // The container image runs a self-contained server (docker/frontend.Dockerfile sets this); `next start` is for everything else.
  ...(process.env.NEXT_OUTPUT === "standalone" ? { output: "standalone" } : {}),
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return sameOriginApi ? [{ source: "/api/:path*", destination: `${apiProxyTarget}/api/:path*` }] : [];
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
