import { expect, snapshot, test } from "./fixtures";

const menu = (page: import("@playwright/test").Page) => page.getByRole("listbox", { name: "sample graphs" });

/* Every sample is listed whole under its group, and the arrows walk them in
   the order shown, across the group headings. */
test("the sample menu lists every sample whole, by group", async ({ page }) => {
  await page.getByRole("button", { name: /^sample graphs/ }).click();
  const options = menu(page).getByRole("option");
  await expect(options).toHaveCount(15);
  await expect(menu(page).getByRole("group")).toHaveCount(5);
  // No option is cut by the menu's edge.
  const box = (await menu(page).boundingBox())!;
  for (const option of await options.all()) {
    const row = (await option.boundingBox())!;
    expect(row.y).toBeGreaterThanOrEqual(box.y);
    expect(row.y + row.height).toBeLessThanOrEqual(box.y + box.height);
  }

  // The menu opens on the loaded sample; ↓ follows the listing, not the index.
  await expect(menu(page)).toBeFocused();
  const active = () => menu(page).getAttribute("aria-activedescendant");
  for (const name of ["Shared operand (A @ A)", "Batched matmul", "Multi-head attention"]) {
    await page.keyboard.press("ArrowDown");
    const id = await active();
    await expect(page.locator(`#${id}`)).toHaveText(name);
  }
  await page.keyboard.press("Enter");
  await expect(menu(page)).toHaveCount(0);
  await expect(page.getByRole("button", { name: /^sample graphs/ })).toBeFocused();
  await expect.poll(async () => (await snapshot(page)).draftText).toContain("H = 4");
});

/* A window too short for the whole list scrolls it inside the window, faded
   at the edge it continues past, rather than running off the screen. */
test("a short window scrolls the menu within it", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 420 });
  await page.getByRole("button", { name: /^sample graphs/ }).click();
  const box = (await menu(page).boundingBox())!;
  expect(box.y + box.height).toBeLessThanOrEqual(420);
  await expect(menu(page)).toHaveClass(/more-below/);
  await page.keyboard.press("End");
  await expect(page.locator(`#${await menu(page).getAttribute("aria-activedescendant")}`)).toBeInViewport();
  await expect(menu(page)).toHaveClass(/more-above/);
  // The menu scrolled, not the panel holding it.
  await expect(page.getByRole("button", { name: /^sample graphs/ })).toBeInViewport();
  await expect(page.getByRole("heading", { name: "Graph source" })).toBeInViewport();
});

/* Under the "Plan experiments" heading the prefix would repeat it; the
   trigger, which stands alone, keeps the full name. */
test("lists experiments without the prefix their heading states", async ({ page }) => {
  await page.getByRole("button", { name: /^sample graphs/ }).click();
  await menu(page).getByRole("option", { name: "Fusion and recomputation" }).click();
  await expect(page.getByRole("button", { name: /^sample graphs/ })).toContainText("Experiment: fusion and recomputation");
});
