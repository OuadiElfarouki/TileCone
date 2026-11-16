import type { State } from "../src/state/types";
import { act, expect, runSource, snapshot, test } from "./fixtures";

const source = `M = 96
K = 80
N = 72
A = Tensor(M, K)
B = Tensor(K, N)
Y = matmul(A, B)`;

test("a plan starts with explicit independent tile axes, without a canvas gesture", async ({ page }) => {
  await runSource(page, source);
  await act(page, "setInspectorTab", "plan");
  await page.getByText("tile tensor", { exact: true }).click();
  await expect(page.getByLabel("tensor to tile")).toHaveValue("Y");
  await page.getByLabel("new Y tile M", { exact: true }).fill("24");
  await page.getByLabel("new Y tile N", { exact: true }).fill("40");
  expect((await snapshot(page)).planTiles).toEqual({});
  await page.getByRole("button", { name: "tile", exact: true }).click();
  await expect.poll(async () => (await snapshot(page)).planTiles).toEqual({ Y: [24, 40] });
  await expect(page.getByRole("group", { name: "Y tile", exact: true })).toBeVisible();
  await expect(page.getByText("New tile grid", { exact: true })).toHaveCount(0);
  const m = page.getByLabel("Y tile M", { exact: true });
  await m.fill("13");
  await m.press("Enter");
  await expect.poll(async () => (await snapshot(page)).planTiles.Y).toEqual([13, 40]);
  const n = page.getByLabel("Y tile N", { exact: true });
  await n.fill("27");
  await n.press("Escape");
  await expect(n).toHaveValue("40");
  await n.fill("0");
  await n.press("Enter");
  await expect(n).toHaveAttribute("aria-invalid", "true");
  expect((await snapshot(page)).planTiles.Y).toEqual([13, 40]);
});

test("the new tile grid only seeds untiled tensors and a replay ends on retiling", async ({ page }) => {
  await runSource(page, source.replace("Y = matmul(A, B)", "C = matmul(A, B)\nY = relu(C)"));
  await act(page, "setPlanTile", "Y", [24, 40]);
  await act(page, "selectPlanTask", { tensorId: "Y", coord: [0, 0] });
  await act(page, "setInspectorTab", "plan");
  const summary = page.getByText("New tile grid", { exact: true });
  await summary.focus();
  await summary.press("Enter");
  const slider = page.getByRole("slider", { name: "new tile grid detail", exact: true });
  await slider.focus();
  await slider.press("End");
  expect((await snapshot(page)).planTiles.Y).toEqual([24, 40]);
  await expect(page.getByText("For untiled tensors. Existing plan tiles stay unchanged.")).toBeVisible();
  await page.getByRole("button", { name: "replay", exact: true }).click();
  await expect.poll(async () => (await snapshot(page)).replayPlayback).not.toBeNull();
  const m = page.getByLabel("Y tile M", { exact: true });
  await m.fill("13");
  await m.press("Enter");
  await expect.poll(() => page.evaluate(() => {
    const s = (window as unknown as { __TILECONE__: { getState(): State } }).__TILECONE__.getState();
    return s.replayPlayback === null || s.replayPlayback.exiting;
  })).toBe(true);
  expect((await snapshot(page)).planTiles.Y).toEqual([13, 40]);
});

test("a narrow workspace keeps canvas space and source focus without changing panel preferences", async ({ page }) => {
  const sourceField = page.getByLabel("graph source", { exact: true });
  const panels = () => page.evaluate(() => {
    const s = (window as unknown as { __TILECONE__: { getState(): State } }).__TILECONE__.getState();
    return { width: s.panelW, collapsed: s.panelCollapsed };
  });
  const before = await panels();
  await sourceField.focus();
  await page.setViewportSize({ width: 760, height: 800 });
  await expect(sourceField).toBeFocused();
  await expect.poll(async () => (await page.locator(".graph-canvas").boundingBox())?.width ?? 0).toBeGreaterThanOrEqual(300);
  expect(await panels()).toEqual(before);
  await page.setViewportSize({ width: 1280, height: 800 });
  await expect(sourceField).toBeFocused();
  expect(await panels()).toEqual(before);
  await page.setViewportSize({ width: 760, height: 800 });
  await act(page, "setPanelWidth", "right", 560);
  const separator = page.getByRole("separator", { name: "tiles panel width", exact: true });
  const drawn = (await page.locator(".panel-frame.right").boundingBox())!.width;
  await separator.focus();
  await separator.press("ArrowRight");
  await expect.poll(async () => (await page.locator(".panel-frame.right").boundingBox())!.width).toBeLessThan(drawn);
  await expect(page.getByRole("separator", { name: "source panel width", exact: true })).toBeHidden();
});

test("tile creation handles scalar and rank-four tensors and resets on graph replacement", async ({ page }) => {
  await runSource(page, "X = Tensor()\nY = relu(X)");
  await act(page, "setInspectorTab", "plan");
  await page.getByText("tile tensor", { exact: true }).click();
  await expect(page.getByRole("group", { name: "new Y tile", exact: true }).getByRole("textbox")).toHaveCount(0);
  await page.getByRole("button", { name: "tile", exact: true }).click();
  await expect.poll(async () => (await snapshot(page)).planTiles.Y).toEqual([]);
  await runSource(page, "X = Tensor(batch=2, head=4, seq=32, dim=64)\nY = relu(X)");
  await act(page, "setInspectorTab", "plan");
  const summary = page.getByText("tile tensor", { exact: true });
  const details = page.locator("details.plan-add-tensor");
  if (!(await details.evaluate((element) => (element as HTMLDetailsElement).open))) await summary.click();
  for (const [axis, value] of Object.entries({ batch: "1", head: "3", seq: "11", dim: "17" })) {
    await page.getByLabel(`new Y tile ${axis}`, { exact: true }).fill(value);
  }
  await page.getByRole("button", { name: "tile", exact: true }).click();
  await expect.poll(async () => (await snapshot(page)).planTiles.Y).toEqual([1, 3, 11, 17]);
});
