import { expect, test as base, type Page } from "@playwright/test";
import type { State } from "../src/state/types";

/** What a journey may read off the store: plain data, cloned out of the page. */
export type Snapshot = {
  dslText: string;
  draftText: string;
  loadError: string | null;
  diagnostics: { severity: string; message: string; span: { start: { offset: number }; end: { offset: number } } }[];
  viewCfgs: State["viewCfgs"];
  nodeOffsets: State["nodeOffsets"];
  history: number;
  parts: { tensorId: string; box: { lo: number; hi: number }[]; hue?: number }[];
  planTiles: State["planTiles"];
  planTask: State["planTask"];
  resizingTensor: string | null;
  built: boolean;
};

export async function snapshot(page: Page): Promise<Snapshot> {
  return page.evaluate(() => {
    const s = (window as unknown as { __TILECONE__: { getState(): State } }).__TILECONE__.getState();
    return JSON.parse(JSON.stringify({
      dslText: s.dslText, draftText: s.draftText, loadError: s.loadError, diagnostics: s.diagnostics,
      viewCfgs: s.viewCfgs, nodeOffsets: s.nodeOffsets, history: s.workspaceHistory.length,
      parts: s.selection?.parts ?? [], planTiles: s.planTiles, planTask: s.planTask,
      resizingTensor: s.resizingTensor, built: s.resolved !== null,
    }));
  });
}

/** Run a store action in the page, by name, with plain-data arguments. */
export async function act(page: Page, action: keyof State, ...args: unknown[]): Promise<void> {
  await page.evaluate(([name, rest]) => {
    const s = (window as unknown as { __TILECONE__: { getState(): Record<string, (...a: unknown[]) => unknown> } })
      .__TILECONE__.getState();
    return s[name as string](...(rest as unknown[]));
  }, [action, args] as const);
}

/** Type a program into the source editor and run it, as the author would. */
export async function runSource(page: Page, source: string): Promise<void> {
  const editor = page.getByLabel("graph source", { exact: true });
  await editor.fill(source);
  await editor.press("Control+Enter");
  await expect.poll(async () => {
    const s = await snapshot(page);
    return s.dslText === source || s.diagnostics.some((d) => d.severity === "error");
  }).toBe(true);
}

/** Every journey fails on an uncaught page error or a console error. */
export const test = base.extend<{ page: Page }>({
  page: async ({ page }, use) => {
    const problems: string[] = [];
    page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
    page.on("console", (message) => {
      if (message.type() === "error") problems.push(`console: ${message.text()}`);
    });
    await page.goto("/");
    await expect.poll(async () => (await snapshot(page)).built).toBe(true);
    await use(page);
    expect(problems).toEqual([]);
  },
});

export { expect };
