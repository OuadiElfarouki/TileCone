/**
 * The budget and the sweep in the store: the budget as a stated setting that a
 * link carries, the sweep as a transient result tied to one graph and one
 * budget, and a sweep row installed as an ordinary plan edit.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { useStore } from "../../src/state/store";
import { decodeWorkspace, encodeWorkspace, linkToRestore, workspaceToLink } from "../../src/state/share";
import { DEFAULT_BUDGET } from "../../src/core/plan/presets";
import { planReport } from "../../src/core/plan/interfaces";
import { sameSavedWorkspace } from "../../src/state/workspace-document";
import { EXAMPLES } from "../../src/examples/index";

const S = () => useStore.getState();

const MLP = `X = Tensor(8, 4, dtype=float16)
W1 = Parameter(4, 8, dtype=float16)
W2 = Parameter(8, 4, dtype=float16)
H = matmul(X, W1)
Y = matmul(H, W2)
`;

const done = () => {
  const sweep = S().sweep;
  if (sweep.status !== "done") throw new Error(`sweep is ${sweep.status}`);
  return sweep;
};

describe("the budget", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().setPlanBudget(DEFAULT_BUDGET);
    S().applyDSL(MLP);
  });

  it("starts at the preset's and takes a stated one", () => {
    expect(S().planBudget).toEqual({ stepBytes: 99 * 1024 });
    S().setPlanBudget({ stepBytes: 64, accumulatorBytes: 128 });
    expect(S().planBudget).toEqual({ stepBytes: 64, accumulatorBytes: 128 });
  });

  it("is unchanged by a budget that is not positive, finite numbers, or is the same", () => {
    S().setPlanBudget({ stepBytes: 64 });
    const before = S().planBudget;
    for (const bad of [{ stepBytes: 0 }, { stepBytes: NaN }, { stepBytes: 64, accumulatorBytes: -1 }, { stepBytes: Infinity }])
      S().setPlanBudget(bad);
    S().setPlanBudget({ stepBytes: 64 });
    expect(S().planBudget).toBe(before);
  });

  it("outlives a graph replacement and is not an undo step", () => {
    const history = S().workspaceHistory;
    S().setPlanBudget({ stepBytes: 64 });
    expect(S().workspaceHistory).toBe(history);
    S().loadExample(0);
    expect(S().planBudget).toEqual({ stepBytes: 64 });
  });

  it("survives loading a built-in experiment", () => {
    const experimentIndex = EXAMPLES.findIndex((example) => example.experiment);
    expect(experimentIndex).toBeGreaterThanOrEqual(0);
    S().setPlanBudget({ stepBytes: 64, accumulatorBytes: 128 });
    S().loadExample(experimentIndex);
    expect(S().planBudget).toEqual({ stepBytes: 64, accumulatorBytes: 128 });
  });

  it("is part of the saved workspace and of a link, unless it is the default", () => {
    expect("budget" in workspaceToLink(S())).toBe(false);
    const before = S();
    S().setPlanBudget({ stepBytes: 64, accumulatorBytes: 128 });
    expect(sameSavedWorkspace(before, S())).toBe(false);
    const link = workspaceToLink(S());
    expect(link.budget).toEqual({ s: 64, a: 128 });
    const decoded = decodeWorkspace(`#s=${encodeWorkspace(link)}`)!;

    S().setPlanBudget(DEFAULT_BUDGET);
    expect(S().restoreWorkspace(linkToRestore(decoded))).toBe(true);
    expect(S().planBudget).toEqual({ stepBytes: 64, accumulatorBytes: 128 });
  });

  it("is the default after restoring a link that states none", () => {
    const link = workspaceToLink(S());
    S().setPlanBudget({ stepBytes: 64 });
    expect(S().restoreWorkspace(linkToRestore(decodeWorkspace(`#s=${encodeWorkspace(link)}`)!))).toBe(true);
    expect(S().planBudget).toEqual(DEFAULT_BUDGET);
  });

  it("is refused in a link when malformed", () => {
    const link = workspaceToLink(S());
    for (const budget of [{ s: 0 }, { s: "64" }, { s: 64, a: 0 }, { s: 64, x: 1 }, [], null, { a: 64 }])
      expect(decodeWorkspace(`#s=${encodeWorkspace({ ...link, budget } as never)}`)).toBeNull();
  });
});

describe("the sweep", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().applyDSL(MLP);
    S().setPlanBudget({ stepBytes: 48, accumulatorBytes: 64 });
  });

  it("lists plans within the budget, each with its own report", async () => {
    expect(S().sweep).toEqual({ status: "idle" });
    await S().runSweep("all");
    const { result, budget, scope } = done();
    expect(scope).toBe("all");
    expect(budget).toBe(S().planBudget);
    expect(result.rows.length).toBeGreaterThan(0);
    expect(result.counts.fusionSets).toBe(2);
    for (const row of result.rows) {
      expect(row.stepBytes).toBeLessThanOrEqual(48);
      expect(row.accumulatorBytes).toBeLessThanOrEqual(64);
    }
  });

  it("sweeps the tensors the plan tiles when asked for the current set", async () => {
    S().setPlanTile("Y", [4, 4]);
    await S().runSweep("current");
    const { result } = done();
    expect(result.counts.fusionSetsTotal).toBe(1);
    for (const row of result.rows) expect(Object.keys(row.tiles)).toEqual(["Y"]);
    // With nothing tiled there is no current set, and every set is searched.
    S().setPlanTile("Y", null);
    await S().runSweep("current");
    expect(done().result.counts.fusionSets).toBe(2);
  });

  it("installs a row as a plan edit that undoes", async () => {
    S().setPlanTile("Y", [8, 4]);
    await S().runSweep("all");
    const { result } = done();
    const index = result.rows.findIndex((row) => Object.keys(row.steps).length > 0);
    expect(index).toBeGreaterThanOrEqual(0);
    const row = result.rows[index];
    const history = S().workspaceHistory.length;

    S().installSweepRow(index);
    expect(S().planTiles).toEqual(row.tiles);
    expect(S().planSteps).toEqual(row.steps);
    expect(S().inspectorTab).toBe("plan");
    expect(S().workspaceHistory.length).toBe(history + 1);
    expect(planReport(S().plan!).total).toEqual(row.total);
    // The result stays: the rows are still plans of this graph and budget.
    expect(S().sweep.status).toBe("done");

    // Installing the plan that is already current is no edit.
    S().installSweepRow(index);
    expect(S().workspaceHistory.length).toBe(history + 1);

    S().undoWorkspace();
    expect(S().planTiles).toEqual({ Y: [8, 4] });
    expect(S().planSteps).toEqual({});
  });

  it("ignores a row that is not there", async () => {
    S().installSweepRow(0);
    await S().runSweep("all");
    const before = S().plan;
    S().installSweepRow(10_000);
    expect(S().plan).toBe(before);
  });

  it("is dropped by a new budget, a new graph, and on request", async () => {
    await S().runSweep("all");
    S().setPlanBudget({ stepBytes: 64 });
    expect(S().sweep).toEqual({ status: "idle" });
    await S().runSweep("all");
    S().applyDSL(`${MLP}# edited\n`);
    expect(S().sweep).toEqual({ status: "idle" });
    await S().runSweep("all");
    S().clearSweep();
    expect(S().sweep).toEqual({ status: "idle" });
  });

  it("is not saved: a sweep leaves the saved workspace as it was", async () => {
    const before = S();
    await S().runSweep("all");
    expect(sameSavedWorkspace(before, S())).toBe(true);
    expect("sweep" in workspaceToLink(S())).toBe(false);
  });

  it("reports a failure instead of a result", async () => {
    // A set that names a graph input is refused by the engine.
    useStore.setState({ planTiles: { X: [8, 4] } as never });
    await S().runSweep("current");
    expect(S().sweep).toMatchObject({ status: "failed", scope: "current" });
  });
});
