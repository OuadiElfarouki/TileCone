/// <reference types="vite/client" />
import { describe, expect, it } from "vitest";

/*
 * The source tree is layered, and imports only point downward:
 *
 *   core, parse  <-  examples  <-  view  <-  worker, state  <-  components  <-  App, main
 *
 * `core` and `parse` are the engine and know nothing about presentation. `view`
 * is the presentation model: pure functions and data, no React and no store, so
 * the Worker can use it. `state` is the store and the Worker client. React lives
 * in `components` and above. A layout that is only described drifts; this walks
 * every import and fails on one that points up or sideways.
 */

/** Every source file under `src/`, keyed `src/…`, with its text. */
const SOURCES = Object.fromEntries(
  Object.entries(
    import.meta.glob("../../src/**/*.{ts,tsx}", { query: "?raw", import: "default", eager: true })
  ).map(([path, text]) => [path.replace(/^(\.\.\/)+/, ""), text as string])
);
const DIRS = new Set(
  Object.keys(SOURCES).flatMap((path) =>
    path.split("/").slice(0, -1).map((_, i, parts) => parts.slice(0, i + 1).join("/")))
);

/** `a/b/../c` -> `a/c`, for the relative specifiers imports use. */
function join(dir: string, spec: string): string {
  const out = dir.split("/");
  for (const part of spec.split("/")) {
    if (part === "..") out.pop();
    else if (part !== ".") out.push(part);
  }
  return out.join("/");
}

/** Which layers each layer may import from, besides itself. */
const ALLOWED: Record<string, string[]> = {
  core: ["parse"],
  parse: ["core"],
  examples: [],
  view: ["core", "parse"],
  worker: ["core", "parse", "view"],
  state: ["core", "parse", "examples", "view", "worker"],
  components: ["core", "parse", "examples", "view", "worker", "state"],
  app: ["core", "parse", "examples", "view", "worker", "state", "components"],
};

/** Packages a layer may use. Anything not listed is refused. */
const PACKAGES: Record<string, string[]> = {
  core: ["zod"],
  parse: ["zod"],
  examples: [],
  view: ["dagre"],
  worker: [],
  state: ["zustand"],
  components: ["react", "react-dom"],
  app: ["react", "react-dom"],
};

/** The top-level folder under `src/`; files at the root (App, main, styles) are the app. */
function layerOf(path: string): string {
  const parts = path.split("/").slice(1);
  return parts.length === 1 && !DIRS.has(path) ? "app" : parts[0];
}

function imports(text: string): string[] {
  const found: string[] = [];
  // Statements only, so prose that happens to say "from" inside a string is not read as one.
  for (const m of text.matchAll(/^\s*(?:import|export)\b[^"'`;]*?\bfrom\s*["']([^"']+)["']/gm)) found.push(m[1]);
  for (const m of text.matchAll(/^\s*import\s*["']([^"']+)["']/gm)) found.push(m[1]);
  for (const m of text.matchAll(/\bimport\(\s*["']([^"']+)["']/g)) found.push(m[1]);
  for (const m of text.matchAll(/new URL\(\s*["']([^"']+)["']/g)) found.push(m[1]);
  return found;
}

describe("source layering", () => {
  const files = Object.keys(SOURCES);

  it("finds the source tree", () => {
    expect(files.length).toBeGreaterThan(50);
  });

  it("knows the layer of every source file", () => {
    for (const file of files) expect(ALLOWED, file).toHaveProperty(layerOf(file));
  });

  it("imports only from its own layer or the layers below it", () => {
    const violations: string[] = [];
    for (const file of files) {
      const from = layerOf(file);
      const dir = file.split("/").slice(0, -1).join("/");
      for (const spec of imports(SOURCES[file])) {
        const bare = spec.split("?")[0];
        if (bare.startsWith(".")) {
          const target = layerOf(join(dir, bare));
          if (target !== from && !ALLOWED[from].includes(target))
            violations.push(`${file} (${from}) -> ${spec} (${target})`);
        } else {
          const pkg = bare.startsWith("@") ? bare.split("/").slice(0, 2).join("/") : bare.split("/")[0];
          if (!PACKAGES[from].includes(pkg)) violations.push(`${file} (${from}) -> package ${pkg}`);
        }
      }
    }
    expect(violations).toEqual([]);
  });
});
