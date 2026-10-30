import type { Page } from "@playwright/test";
import { act, expect, runSource, snapshot, test } from "./fixtures";

/* A card between two operations: growing it has to move a neighbour aside,
   which is the case the old resize stopped dead at. */
const SOURCE = "X = Tensor(8, 16, dtype=float32)\nY = relu(X)\nZ = exp(Y)\nW = relu(Z)\n";

async function enterResize(page: Page, tensor: string) {
  // The card takes the press over its canvas, and owns the context menu.
  await page.locator(`.tensor-card[data-tensor="${tensor}"]`).click({ button: "right" });
  await page.getByRole("menuitem", { name: "resize card" }).click();
  await expect.poll(async () => (await snapshot(page)).resizingTensor).toBe(tensor);
}

/** Press the card's bottom-right handle and drag it by `dx, dy` screen px. */
async function dragHandle(page: Page, tensor: string, dx: number, dy: number) {
  const handle = page.locator(`.tensor-card[data-tensor="${tensor}"] [data-card-resize="bottom"]`);
  const box = (await handle.boundingBox())!;
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y + dy, { steps: 8 });
}

test.beforeEach(async ({ page }) => {
  await runSource(page, SOURCE);
  await act(page, "setCanvasTool", "move");
});

test("a resize grows live, moves its neighbours, and cancels to where it began", async ({ page }) => {
  const before = await snapshot(page);
  await enterResize(page, "Y");
  await dragHandle(page, "Y", 260, 130);
  const during = await snapshot(page);
  expect(during.viewCfgs.Y.scale).toBeGreaterThan(1.5);
  // Something other than Y moved aside, and nothing is recorded mid-gesture.
  expect(Object.keys(during.nodeOffsets).some((key) => key !== "t:Y")).toBe(true);
  expect(during.history).toBe(before.history);
  await page.keyboard.press("Escape");
  await page.mouse.up();
  const after = await snapshot(page);
  expect(after.viewCfgs.Y.scale).toBe(before.viewCfgs.Y.scale);
  expect(after.nodeOffsets).toEqual(before.nodeOffsets);
  expect(after.history).toBe(before.history);
});

test("a released resize is one undo step, covering the neighbours it moved", async ({ page }) => {
  const before = await snapshot(page);
  await enterResize(page, "Y");
  await dragHandle(page, "Y", 260, 130);
  await page.mouse.up();
  const after = await snapshot(page);
  expect(after.viewCfgs.Y.scale).toBeGreaterThan(1.5);
  expect(after.history).toBe(before.history + 1);
  // Leave resize mode the way a reader would, then undo from the keyboard.
  await page.keyboard.press("Escape");
  await expect.poll(async () => (await snapshot(page)).resizingTensor).toBeNull();
  await page.keyboard.press("Control+z");
  const undone = await snapshot(page);
  expect(undone.viewCfgs.Y.scale).toBe(before.viewCfgs.Y.scale);
  expect(undone.nodeOffsets).toEqual(before.nodeOffsets);
});
