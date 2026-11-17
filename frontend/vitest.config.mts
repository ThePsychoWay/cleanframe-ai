import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Library tests run in Node. Component tests (*.test.tsx) opt into a browser-like DOM with
// `// @vitest-environment jsdom` on their first line.
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL("./", import.meta.url)) } },
  test: { setupFiles: ["./vitest.setup.ts"], include: ["**/*.test.{ts,tsx}"], exclude: ["node_modules/**", ".next/**"] },
});
