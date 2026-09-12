import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    include: ["src/test/**/*.test.ts"],
    environment: "node",
    // The source editor's two layers have to agree on their geometry, and the
    // test that checks that reads styles.css as text (`?raw`), which vitest
    // stubs out unless it is processing CSS.
    css: true,
  },
});
