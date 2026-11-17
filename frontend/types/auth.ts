export interface User {
  id: string;
  email: string;
  name: string;
  role: "user" | "admin";
  created_at: string;
}

export interface RegisterInput {
  name: string;
  email: string;
  password: string;
}

export interface LoginInput {
  email: string;
  password: string;
}
