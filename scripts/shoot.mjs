#!/usr/bin/env node
/**
 * Screenshots of the running app, for UI review without a throwaway e2e file.
 *
 *   node scripts/shoot.mjs <steps.mjs> [--sizes 1280x800,760x800] [--out DIR] [--theme light|dark]
 *
 * `steps.mjs` default-exports `async (page, t) => {}`. It runs once per size on
 * a fresh page that has built its starting graph. `t` offers:
 *   t.act(name, ...args)   call a store action (the e2e mode exposes the store)
 *   t.state(fn)            evaluate `fn(state)` in the page and return the result
 *   t.runSource(text)      type a program into the editor and build it
 *   t.shot(label)          save <out>/<label>-<w>x<h>.png
 * With no t.shot call, one "final" shot is taken at the end.
 *
 * Uses the e2e dev server on port 4179 (the Playwright config's), starting it
 * when it is not already up and stopping it afterwards.
 */
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "@playwright/test";

const PORT = 4179;
const BASE = `http://127.0.0.1:${PORT}`;

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const stepsPath = args.find((a, i) => !a.startsWith("--") && !args[i - 1]?.startsWith("--"));
if (!stepsPath) {
  console.error("usage: node scripts/shoot.mjs <steps.mjs> [--sizes 1280x800,760x800] [--out DIR] [--theme light|dark]");
  process.exit(2);
}
const sizes = option("sizes", "1280x800").split(",").map((s) => s.split("x").map(Number));
const out = resolve(option("out", "screenshots"));
const theme = option("theme", "light");
mkdirSync(out, { recursive: true });

const up = async () => {
  try {
    return (await fetch(BASE)).ok;
  } catch {
    return false;
  }
};

let server = null;
if (!(await up())) {
  server = spawn("npx", ["vite", "--mode", "e2e", "--port", String(PORT), "--strictPort", "--host", "127.0.0.1"], {
    stdio: "ignore",
    detached: true,
  });
  const deadline = Date.now() + 60_000;
  while (!(await up())) {
    if (Date.now() > deadline) throw new Error("dev server did not start on " + BASE);
    await new Promise((r) => setTimeout(r, 300));
  }
}

const steps = (await import(pathToFileURL(resolve(stepsPath)).href)).default;
const browser = await chromium.launch();
const saved = [];
const errors = [];
try {
  for (const [width, height] of sizes) {
    const page = await browser.newPage({ viewport: { width, height }, colorScheme: theme });
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));
    await page.goto(BASE);
    await page.waitForFunction(() => window.__TILECONE__?.getState().resolved != null);
    let shots = 0;
    const t = {
      act: (name, ...rest) =>
        page.evaluate(([n, r]) => window.__TILECONE__.getState()[n](...r), [name, rest]),
      state: (fn) => page.evaluate(`(${fn})(window.__TILECONE__.getState())`),
      runSource: async (text) => {
        const editor = page.getByLabel("graph source", { exact: true });
        await editor.fill(text);
        await editor.press("Control+Enter");
        await page.waitForFunction(
          (src) => window.__TILECONE__.getState().dslText === src,
          text
        );
      },
      shot: async (label) => {
        const path = `${out}/${label}-${width}x${height}.png`;
        await page.screenshot({ path });
        saved.push(path);
        shots++;
      },
    };
    await steps(page, t);
    if (shots === 0) await t.shot("final");
    await page.close();
  }
} finally {
  await browser.close();
  if (server) process.kill(-server.pid);
}
for (const path of saved) console.log(path);
if (errors.length) {
  console.log("page errors:");
  for (const e of errors) console.log("  " + e);
}
