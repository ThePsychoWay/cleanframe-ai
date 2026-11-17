import { apiFetch } from "@/lib/api";
import type { LoginInput, RegisterInput, User } from "@/types/auth";

export const authService = {
  register: (input: RegisterInput) => apiFetch<User>("/api/auth/register", { method: "POST", json: input }),
  login: (input: LoginInput) => apiFetch<User>("/api/auth/login", { method: "POST", json: input }),
  logout: () => apiFetch<void>("/api/auth/logout", { method: "POST" }),
  me: () => apiFetch<User>("/api/auth/me"),
};
