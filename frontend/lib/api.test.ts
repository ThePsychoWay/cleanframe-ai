import { describe, expect, it } from "vitest";
import { messageFromDetail } from "./api";
import { safeNextPath } from "./navigation";

describe("messageFromDetail", () => {
  it("passes string details through", () => {
    expect(messageFromDetail("Incorrect email or password.", 401)).toBe("Incorrect email or password.");
  });

  it("labels the first validation issue by field", () => {
    const detail = [{ loc: ["body", "password"], msg: "String should have at least 10 characters" }];
    expect(messageFromDetail(detail, 422)).toBe("Password: String should have at least 10 characters.");
  });

  it("falls back by status when there is nothing readable", () => {
    expect(messageFromDetail(undefined, 429)).toContain("Too many attempts");
    expect(messageFromDetail(null, 503)).toContain("on our side");
    expect(messageFromDetail([], 400)).toBe("Something went wrong. Please try again.");
  });
});

describe("safeNextPath", () => {
  it("keeps same-site paths", () => {
    expect(safeNextPath("/dashboard")).toBe("/dashboard");
    expect(safeNextPath("/dashboard?tab=1#top")).toBe("/dashboard?tab=1#top");
  });

  it("rejects anything that could leave the site", () => {
    for (const bad of ["https://evil.example", "//evil.example", "/\\evil.example", "javascript:alert(1)", "evil", ""]) {
      expect(safeNextPath(bad)).toBe("/dashboard");
    }
    expect(safeNextPath(null)).toBe("/dashboard");
    expect(safeNextPath(undefined, "/home")).toBe("/home");
  });
});
