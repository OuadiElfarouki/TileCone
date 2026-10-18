import type { Page } from "@playwright/test";
import { expect, runSource, snapshot, test } from "./fixtures";

const editor = (page: Page) => page.getByLabel("graph source", { exact: true });
const selected = (page: Page) => editor(page).evaluate((el: HTMLTextAreaElement) =>
  el.value.slice(el.selectionStart, el.selectionEnd));

test("clicking an error selects what it underlines, below a note that the old graph remains", async ({ page }) => {
  await runSource(page, "X = Tensor(4, 4)\nY = relu(X)\n");
  // Line 2 reads a tensor nobody declared.
  const broken = "X = Tensor(4, 4)\nY = relu(Q)\n";
  await editor(page).fill(broken);
  await editor(page).press("Control+Enter");
  const diagnostic = page.locator(".source-status button.diag").filter({ hasText: "line 2" });
  await expect(diagnostic).toBeVisible();
  await expect(page.getByText("The canvas still shows the last graph that built.")).toBeVisible();
  // Move the caret away first, so the selection can only come from the click.
  await editor(page).evaluate((el: HTMLTextAreaElement) => el.setSelectionRange(0, 0));
  await diagnostic.click();
  await expect(editor(page)).toBeFocused();
  expect(await selected(page)).toBe("Q");
  expect((await snapshot(page)).dslText).toBe("X = Tensor(4, 4)\nY = relu(X)\n");
});

test("a build warning is listed beside the build, and selects its operand", async ({ page }) => {
  const source = "M = 4\nN = 4\nA = Tensor(3, M)\nB = Tensor(N, 5)\nC = matmul(A, B)\n";
  await runSource(page, source);
  const s = await snapshot(page);
  expect(s.dslText).toBe(source);
  expect(s.diagnostics.map((d) => d.severity)).toEqual(["warning"]);
  await expect(page.locator(".source-status .warning")).toContainText("1 warning");
  await page.locator(".source-status .warning button.diag").click();
  expect(await selected(page)).toBe("B");
});

test("an edit makes diagnostics stop pointing into the source until the next run", async ({ page }) => {
  await runSource(page, "X = Tensor(4, 4)\nY = relu(Q)\n");
  const links = page.locator(".source-status button.diag");
  await expect(links).toHaveCount(1);
  // A line inserted above the error: its old offsets now fall on other code.
  await editor(page).press("Control+Home");
  await editor(page).type("# inserted above\n");
  await expect(links).toHaveCount(0);
  await expect(page.locator(".source-status .diag.stale")).toContainText("line 2");
  await expect(page.getByText("Edited since this run")).toBeVisible();
  // Running again re-anchors them to the text as it now stands.
  await editor(page).press("Control+Enter");
  await expect(links).toHaveCount(1);
  await expect(links.first()).toContainText("line 3");
  await links.first().click();
  expect(await selected(page)).toBe("Q");
});
