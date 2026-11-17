"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";
import { ApiError } from "@/lib/api";
import { safeNextPath } from "@/lib/navigation";
import { useAuth } from "./AuthProvider";

const PASSWORD_MIN_LENGTH = 10;

const inputClass =
  "mt-1.5 h-11 w-full rounded-lg border border-line bg-panel px-3.5 text-paper placeholder:text-faint";

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const params = useSearchParams();
  const rawNext = params.get("next");
  const next = safeNextPath(rawNext);
  const { status, login, register } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const isSignup = mode === "signup";

  // Already signed in, or just signed in: go where the person was headed.
  useEffect(() => {
    if (status === "authenticated") router.replace(next);
  }, [status, next, router]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (isSignup && password.length < PASSWORD_MIN_LENGTH) {
      setError(`Password must be at least ${PASSWORD_MIN_LENGTH} characters.`);
      return;
    }

    setPending(true);
    try {
      if (isSignup) await register({ name: name.trim(), email: email.trim(), password });
      else await login({ email: email.trim(), password });
      // The effect above navigates once the session is in place; keep the button busy until then.
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Something went wrong. Please try again.");
      setPending(false);
    }
  };

  const otherHref = `${isSignup ? "/login" : "/signup"}${rawNext ? `?next=${encodeURIComponent(next)}` : ""}`;

  return (
    <div>
      <h1 className="heading text-3xl">{isSignup ? "Create your account" : "Log in"}</h1>
      <p className="mt-3 text-dim">
        {isSignup ? "Start restoring videos you own or have permission to edit." : "Welcome back."}
      </p>

      <form onSubmit={onSubmit} className="mt-9 space-y-5" noValidate>
        {isSignup && (
          <div>
            <label htmlFor="name" className="text-sm font-medium">
              Name
            </label>
            <input
              id="name"
              name="name"
              type="text"
              autoComplete="name"
              required
              maxLength={120}
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={inputClass}
            />
          </div>
        )}

        <div>
          <label htmlFor="email" className="text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={inputClass}
          />
        </div>

        <div>
          <label htmlFor="password" className="text-sm font-medium">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete={isSignup ? "new-password" : "current-password"}
            required
            maxLength={128}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            aria-describedby={isSignup ? "password-hint" : undefined}
            className={inputClass}
          />
          {isSignup && (
            <p id="password-hint" className="mt-1.5 text-sm text-faint">
              At least {PASSWORD_MIN_LENGTH} characters. A short phrase of unrelated words works well.
            </p>
          )}
        </div>

        {error && (
          <p role="alert" className="rounded-lg border border-line bg-raised px-4 py-3 text-sm">
            {error}
          </p>
        )}

        <button type="submit" className="btn btn-primary w-full" disabled={pending}>
          {pending ? (isSignup ? "Creating account…" : "Logging in…") : isSignup ? "Create account" : "Log in"}
        </button>
      </form>

      <p className="mt-7 text-sm text-dim">
        {isSignup ? "Already have an account? " : "New to CleanFrame? "}
        <Link href={otherHref} className="font-medium text-paper underline underline-offset-4">
          {isSignup ? "Log in" : "Create an account"}
        </Link>
      </p>
    </div>
  );
}
