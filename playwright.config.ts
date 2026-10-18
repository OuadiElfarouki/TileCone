import { defineConfig, devices } from "@playwright/test";

/**
 * Browser journeys: the few paths whose correctness depends on what unit tests
 * cannot reach - pointer capture, real key events, textarea selection, the
 * analysis Worker and the clipboard. The unit suite stays the main net; these
 * check that its pieces are joined up in a real page.
 *
 * The dev server runs in the `e2e` mode, which exposes the store to the page
 * (`main.tsx`) so a journey can set up state directly and assert on results.
 */
const PORT = 4179;

export default defineConfig({
  testDir: "e2e",
  testMatch: "**/*.e2e.ts",
  // Traces and run state, kept out of the working tree.
  outputDir: "node_modules/.cache/playwright/results",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  reporter: "list",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  webServer: {
    command: `npx vite --mode e2e --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
