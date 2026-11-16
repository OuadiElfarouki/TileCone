import type { State } from "../src/state/types";
import { act, expect, runSource, test } from "./fixtures";

const gemm = `A = Tensor(8, 8, dtype=float16)
B = Tensor(8, 8, dtype=float16)
Y = matmul(A, B)`;

test("budget edits keep exact bytes through Enter, blur and restored values", async ({ page }) => {
  await act(page, "setInspectorTab", "plan");
  const step = page.getByLabel("bytes a task may hold of its operands per step", { exact: true });
  const accumulator = page.getByLabel("bytes an accumulator may take", { exact: true });
  for (const field of [step, accumulator]) {
    await field.fill("99999");
    await field.press("Enter");
    await field.press("Tab");
  }
  const budget = () => page.evaluate(() =>
    (window as unknown as { __TILECONE__: { getState(): State } }).__TILECONE__.getState().planBudget);
  expect(await budget()).toEqual({ stepBytes: 99999, accumulatorBytes: 99999 });
  await act(page, "setPlanBudget", { stepBytes: 123457, accumulatorBytes: 234569 });
  for (const field of [step, accumulator]) {
    await field.focus();
    await field.press("Tab");
  }
  expect(await budget()).toEqual({ stepBytes: 123457, accumulatorBytes: 234569 });
  await accumulator.fill("none");
  await accumulator.press("Tab");
  expect(await budget()).toEqual({ stepBytes: 123457 });
});

test("an empty planned family renders its zero payload", async ({ page }) => {
  await runSource(page, "X = Tensor(4, 8)\nY = X[0:0, :]");
  await act(page, "setPlanTile", "Y", [1, 1]);
  await act(page, "setInspectorTab", "plan");
  await expect(page.getByText("largest step 0 B", { exact: false })).toBeVisible();
});

test("a sweep row stops being current when its accumulator changes", async ({ page }) => {
  await runSource(page, gemm);
  await act(page, "setPlanTile", "Y", [4, 4]);
  await act(page, "setPlanBudget", { stepBytes: 64, accumulatorBytes: 64 });
  await act(page, "setInspectorTab", "plan");
  await act(page, "runSweep", "current");
  await act(page, "installSweepRow", 0);
  const table = page.getByRole("table", { name: "plans within the budget" });
  await expect(table.locator(".plan-sweep-row.current")).toHaveCount(1);
  await page.evaluate(() => {
    const s = (window as unknown as { __TILECONE__: { getState(): State } }).__TILECONE__.getState();
    s.setPlanSteps("Y", { ...s.planSteps.Y, accumulator: "f16" });
  });
  await expect(table.locator(".plan-sweep-row.current")).toHaveCount(0);
  await table.getByRole("button", { name: "use", exact: true }).first().click();
  await expect(table.locator(".plan-sweep-row.current")).toHaveCount(1);
});

test("accepted rows disclose a capped chunk search", async ({ page }) => {
  await runSource(page, `A = Tensor(16, 16, 16, 16, dtype=float16)
B = Tensor(16, 16, 16, 16, dtype=float16)
Y = einsum("abcd,abcd->", A, B)`);
  await act(page, "setPlanTile", "Y", []);
  await act(page, "setPlanBudget", { stepBytes: 64, accumulatorBytes: 4 });
  await act(page, "setInspectorTab", "plan");
  await act(page, "runSweep", "current");
  await expect(page.getByText("capped chunk search", { exact: true })).toBeVisible();
  await expect(page.getByText(/fewer steps or a smaller step payload may exist/)).toBeVisible();
});

test("whole-task computed detail opens by keyboard and stays open between steps", async ({ page }) => {
  await runSource(page, gemm.replace("Y = matmul(A, B)", "C = matmul(A, B)\nY = relu(C)"));
  await act(page, "setPlanTile", "Y", [4, 4]);
  await act(page, "setPlanSteps", "Y", { chunk: { k: 2 } });
  await act(page, "selectPlanTask", { tensorId: "Y", coord: [0, 0] });
  await act(page, "setInspectorTab", "plan");
  const summary = page.getByText("Computed over the whole task", { exact: true });
  const detail = page.locator("details.plan-computed");
  await expect(detail).not.toHaveAttribute("open", "");
  await summary.focus();
  await summary.press("Enter");
  await expect(detail).toHaveAttribute("open", "");
  await expect(detail.getByRole("button", { name: "materialize", exact: true })).toBeVisible();
  await act(page, "setPlanStep", 1);
  await expect(detail).toHaveAttribute("open", "");
});
