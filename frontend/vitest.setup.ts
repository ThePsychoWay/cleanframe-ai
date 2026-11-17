import { afterEach } from "vitest";

// Only component tests have a DOM; library tests run without one.
if (typeof document !== "undefined") {
  const { cleanup } = await import("@testing-library/react");
  await import("@testing-library/jest-dom/vitest");
  afterEach(() => cleanup());
}
