import { expect, snapshot, test } from "./fixtures";

test("the app builds its starting graph in a real browser", async ({ page }) => {
  const s = await snapshot(page);
  expect(s.built).toBe(true);
  await expect(page.getByLabel("graph source", { exact: true })).toBeVisible();
});
