/**
 * Thin fetch wrapper for the CleanFrame API.
 * The session is an httpOnly cookie, so requests always send credentials and the
 * page never handles a token.
 */

/**
 * Where the browser sends API calls. "same-origin" (used on Vercel) means the site's own address: next.config.mjs forwards
 * /api/* to the real API server, so the session cookie is first-party. Otherwise it is the API's full address.
 */
const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const API_URL = configuredApiUrl === "same-origin" ? "" : configuredApiUrl.replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const FIELD_LABELS: Record<string, string> = {
  email: "Email",
  password: "Password",
  name: "Name",
};

function fallbackMessage(status: number): string {
  if (status === 429) return "Too many attempts. Wait a moment and try again.";
  if (status >= 500) return "Something went wrong on our side. Please try again.";
  return "Something went wrong. Please try again.";
}

/** Turns FastAPI's `detail` (a string, or a list of validation issues) into one readable sentence. */
export function messageFromDetail(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.trim() !== "") return detail;

  if (Array.isArray(detail) && detail.length > 0) {
    const first: unknown = detail[0];
    if (typeof first === "object" && first !== null) {
      const { msg, loc } = first as { msg?: unknown; loc?: unknown };
      if (typeof msg === "string") {
        const field = Array.isArray(loc) ? String(loc[loc.length - 1]) : "";
        const label = FIELD_LABELS[field];
        const sentence = msg.endsWith(".") ? msg : `${msg}.`;
        return label ? `${label}: ${sentence}` : sentence;
      }
    }
  }
  return fallbackMessage(status);
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  json?: unknown;
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { json, headers, ...rest } = options;

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      cache: "no-store",
      ...rest,
      credentials: "include",
      headers: { ...(json !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
      body: json !== undefined ? JSON.stringify(json) : undefined,
    });
  } catch {
    throw new ApiError(0, "We couldn't reach the server. Check your connection and try again.");
  }

  if (response.status === 204) return undefined as T;

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body === "object" && body !== null ? (body as { detail?: unknown }).detail : undefined;
    throw new ApiError(response.status, messageFromDetail(detail, response.status));
  }
  return body as T;
}
