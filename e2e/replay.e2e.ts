import type { Page } from "@playwright/test";
import { act, expect, runSource, snapshot, test } from "./fixtures";

const SOURCE = `M = 16
N = 16
K = 128
E = M * N
A = Tensor(M, K, dtype=float16)
B = Tensor(K, N, dtype=float16)
C = matmul(A, B)
D = reshape(C, [E])
D2 = add(D, D)
`;

/** Press the D2 card's canvas at a fraction of its width: the Plan view
 * inspects the task under the press. */
async function pressTask(page: Page, fraction: number) {
  const canvas = page.locator('.tensor-card[data-tensor="D2"] canvas').first();
  const box = (await canvas.boundingBox())!;
  await page.mouse.click(box.x + box.width * fraction, box.y + box.height / 2);
}

const controls = (page: Page) => page.getByRole("group", { name: "replay controls" });

test.beforeEach(async ({ page }) => {
  await runSource(page, SOURCE);
  await page.locator("#view-plan").click();
  // The plan from the report: D2 and D in tiles of 16, C fused into D's tasks.
  // A first tiling also tiles the operands it computes, so C is fused after.
  await act(page, "setPlanTileAt", "D", [16], [0]);
  await act(page, "setPlanTileAt", "D2", [16], [128]);
  await page.getByRole("button", { name: "fuse C into its consumers" }).click();
  await expect.poll(async () => Object.keys((await snapshot(page)).planTiles).sort()).toEqual(["D", "D2"]);
  await expect.poll(async () => (await snapshot(page)).planTask).toEqual({ tensorId: "D2", coord: [8] });
});

test("a replay steps the producers at depth 2 and stops when the inspected task changes", async ({ page }) => {
  await page.getByRole("group", { name: "replay depth" }).getByRole("button", { name: "2" }).click();
  // The depth names what a step visits; the producers are named once found.
  await expect(page.getByText("steps through producer tasks 1 level up")).toBeVisible();
  await page.getByRole("button", { name: "replay", exact: true }).click();
  await expect(controls(page)).toBeVisible();
  await expect(page.getByText("steps through D tasks 1 level up")).toBeVisible();
  await expect(page.getByText(/D2 \(\d+\) ← D \(\d+\) · \d+ of 16/)).toBeVisible();

  // Mid-playback, inspect another task from the card.
  await pressTask(page, 15.5 / 16);
  await expect.poll(async () => (await snapshot(page)).planTask).toEqual({ tensorId: "D2", coord: [15] });
  await expect(controls(page)).toHaveCount(0);

  // A new replay is about the new task, and runs to its end.
  await page.getByRole("button", { name: "replay", exact: true }).click();
  await expect(controls(page)).toBeVisible();
  await expect(page.getByText(/· 16 of 16/)).toBeVisible({ timeout: 15_000 });
});

test("clearing the plan's tiling mid-replay retires it without an error", async ({ page }) => {
  await page.getByRole("button", { name: "replay", exact: true }).click();
  await expect(controls(page)).toBeVisible();
  await act(page, "setPlanTileAt", "D2", [32], [0]);
  await expect(controls(page)).toHaveCount(0);
});

/* On the button that starts the replay, or on the transport's buttons after a
   press, Space toggles the walk and does not press the button again, which
   would restart the replay. */
test("Space pauses and resumes a replay, from the button that started it", async ({ page }) => {
  const replayButton = page.getByRole("button", { name: "replay", exact: true });
  await replayButton.click();
  await expect(controls(page).getByRole("button", { name: "pause" })).toBeVisible();
  // It is disabled while the frames are prepared, which drops focus; a
  // keyboard reader who tabs back to it is the case that would restart.
  await replayButton.focus();
  await expect(replayButton).toBeFocused();

  await page.keyboard.press("Space");
  await expect(controls(page).getByRole("button", { name: "play", exact: true })).toBeVisible();
  const held = (await snapshot(page)).replayPlayback!;
  expect(held.paused).toBe(true);
  await page.waitForTimeout(400);
  expect((await snapshot(page)).replayPlayback!.visited).toBe(held.visited);

  await page.keyboard.press("Space");
  await expect(controls(page).getByRole("button", { name: "pause" })).toBeVisible();
  await expect.poll(async () => (await snapshot(page)).replayPlayback!.visited).toBeGreaterThan(held.visited);

  // From a focused transport button, one press is one toggle.
  const pause = controls(page).getByRole("button", { name: "pause" });
  await pause.focus();
  await page.keyboard.press("Space");
  // The transport keeps focus through the swap, so the next press continues from it.
  await expect(controls(page).getByRole("button", { name: "play", exact: true })).toBeFocused();
  const before = (await snapshot(page)).replayPlayback!.visited;
  await page.keyboard.press("Space");
  await expect(controls(page).getByRole("button", { name: "pause" })).toBeVisible();
  await page.keyboard.press("Space");
  await expect(controls(page).getByRole("button", { name: "play", exact: true })).toBeVisible();
  expect((await snapshot(page)).replayPlayback!.visited).toBeGreaterThanOrEqual(before);
});
