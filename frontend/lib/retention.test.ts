import { describe, expect, it } from "vitest";
import { describeRetention, formatWhen } from "./retention";

const NOW = new Date("2026-11-03T12:00:00Z");

describe("describeRetention", () => {
  it("names the day a project's files will be deleted", () => {
    const text = describeRetention({ status: "COMPLETED", expires_at: "2026-11-05T09:30:00Z" }, NOW, "en-US");
    expect(text).toMatch(/^Files are deleted automatically on .*2026.*\.$/);
  });

  it("does not claim deletion for a project that is only past its deadline", () => {
    const text = describeRetention({ status: "COMPLETED", expires_at: "2026-11-03T11:00:00Z" }, NOW, "en-US");
    expect(text).toBe("Past its retention period; the files are being deleted.");
    expect(text).not.toMatch(/were deleted/);
  });

  it("says the files were deleted only once the backend reports the project expired", () => {
    const text = describeRetention({ status: "EXPIRED", expires_at: "2026-11-01T00:00:00Z" }, NOW);
    expect(text).toMatch(/^The files were deleted/);
  });

  it("says nothing rather than something wrong when the date is unreadable", () => {
    expect(describeRetention({ status: "MASKING", expires_at: "not a date" }, NOW)).toBe("");
    expect(formatWhen("not a date")).toBe("");
  });
});
