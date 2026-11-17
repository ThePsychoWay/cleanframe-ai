"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { ApiError } from "@/lib/api";
import { authService } from "@/services/auth";
import type { LoginInput, RegisterInput, User } from "@/types/auth";

type AuthStatus = "loading" | "authenticated" | "anonymous" | "error";

interface AuthContextValue {
  status: AuthStatus;
  user: User | null;
  errorMessage: string | null;
  refresh: () => Promise<void>;
  login: (input: LoginInput) => Promise<void>;
  register: (input: RegisterInput) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (value === null) throw new Error("useAuth must be used inside <AuthProvider>.");
  return value;
}

/** Like useAuth, but returns null outside an AuthProvider (public pages that only adapt a link). */
export function useOptionalAuth(): AuthContextValue | null {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<User | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setUser(await authService.me());
      setErrorMessage(null);
      setStatus("authenticated");
    } catch (error) {
      setUser(null);
      if (error instanceof ApiError && error.status === 401) {
        setErrorMessage(null);
        setStatus("anonymous");
      } else {
        setErrorMessage(error instanceof ApiError ? error.message : "Something went wrong. Please try again.");
        setStatus("error");
      }
    }
  }, []);

  useEffect(() => {
    // Initial session check; refresh() only sets state after the request resolves.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const login = useCallback(async (input: LoginInput) => {
    setUser(await authService.login(input));
    setStatus("authenticated");
  }, []);

  const register = useCallback(async (input: RegisterInput) => {
    setUser(await authService.register(input));
    setStatus("authenticated");
  }, []);

  const logout = useCallback(async () => {
    await authService.logout();
    setUser(null);
    setStatus("anonymous");
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ status, user, errorMessage, refresh, login, register, logout }),
    [status, user, errorMessage, refresh, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
