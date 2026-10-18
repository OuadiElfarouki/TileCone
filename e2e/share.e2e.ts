import { decodeWorkspace, encodeWorkspace } from "../src/state/share";
import { act, expect, runSource, snapshot, test } from "./fixtures";

const SOURCE = "M = 16\nN = 32\nX = Tensor(M, N, dtype=fp16)\nY = relu(X)\nZ = exp(Y)\n";
const region = (rows: [number, number], cols: [number, number]) => ({
  boxes: [[{ lo: rows[0], hi: rows[1] }, { lo: cols[0], hi: cols[1] }]], exact: true, reasons: [],
});

test("a shared link restores the source, tiles with their colours, and card sizes in a fresh browser", async ({ page, browser, context }) => {
  await runSource(page, SOURCE);
  await act(page, "setSelection", "Y", region([0, 4], [0, 4]), "replace");
  await act(page, "setSelection", "Y", region([4, 8], [0, 4]), "union");
  await act(page, "setSelection", "Y", region([8, 12], [0, 4]), "union");
  // Removing the middle tile leaves a gap in the colours, which must survive.
  await act(page, "deleteBox", 1);
  await act(page, "setCardScale", "Z", 1.5);
  const shared = await snapshot(page);
  expect(shared.parts.map((p) => p.hue)).toEqual([0, 2]);

  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.locator("button.share-btn").click();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  expect(link).toContain("#s=");

  // A new context has no autosave to fall back on: only the link can restore this.
  const fresh = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const other = await fresh.newPage();
  const problems: string[] = [];
  other.on("pageerror", (error) => problems.push(error.message));
  await other.goto(link);
  await expect.poll(async () => (await snapshot(other)).dslText, { timeout: 15_000 }).toBe(SOURCE);
  const restored = await snapshot(other);
  expect(restored.parts).toEqual(shared.parts);
  expect(restored.viewCfgs.Z.scale).toBe(1.5);
  expect(restored.nodeOffsets).toEqual(shared.nodeOffsets);
  expect(problems).toEqual([]);
  await fresh.close();
});

test("an older link without colour slots restores its tiles in order", async ({ page, context }) => {
  await runSource(page, SOURCE);
  await act(page, "setSelection", "Y", region([0, 4], [0, 4]), "replace");
  await act(page, "setSelection", "Y", region([8, 12], [0, 4]), "union");
  await act(page, "deleteBox", 0);
  await act(page, "setSelection", "Y", region([4, 8], [0, 4]), "union");
  expect((await snapshot(page)).parts.map((p) => p.hue)).toEqual([1, 0]);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.locator("button.share-btn").click();
  const link = new URL(await page.evaluate(() => navigator.clipboard.readText()));
  // The same link as a version written before tiles carried colour slots.
  const current = decodeWorkspace(link.hash)!;
  link.hash = `s=${encodeWorkspace({ ...current, sel: current.sel!.map(({ t, box }) => ({ t, box })) })}`;
  // A hash-only change does not reload the page, so leave it first. Autosave
  // would bring back slots [1, 0]; slots given in order show the link won.
  await page.goto("about:blank");
  await page.goto(link.toString());
  await expect.poll(async () => (await snapshot(page)).parts.map((p) => p.hue), { timeout: 15_000 }).toEqual([0, 1]);
});
