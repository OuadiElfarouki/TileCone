import { describe, expect, it } from "vitest";
import { compileDSL } from "../../src/parse/compiler";
import { sweepPlans, type Sweep } from "../../src/core/plan/sweep";
import { rowIsCurrent, sweepCaveats, sweepSummary } from "../../src/view/plan-sweep";

const counts = (over: Partial<Sweep["counts"]> = {}): Sweep["counts"] => ({
  fusionSets: 1, fusionSetsTotal: 1, setsInexact: 0, setsSkipped: 0, measured: 10, overTasks: 0, overBudget: 0,
  overSteps: 0, chunkingsCapped: 0, cappedUnresolved: 0, combinationsDropped: 0, unbeaten: 3, ...over,
});

describe("what a sweep left out", () => {
  it("is nothing when no limit bit", () => {
    expect(sweepCaveats(counts())).toEqual([]);
  });

  it("names each limit that bit, the ones that weaken the rows first", () => {
    const caveats = sweepCaveats(counts({
      fusionSets: 1, fusionSetsTotal: 128, setsSkipped: 8, setsInexact: 2, overTasks: 36, overSteps: 1,
      cappedUnresolved: 1, combinationsDropped: 5,
    }));
    expect(caveats.map((c) => c.split(":")[0].split(".")[0])).toEqual([
      "5 partial combinations dropped at the plan limit",
      "8 fusion choices not reached within the work limit",
      "2 fusion choices not searched",
      "117 fusion choices not considered",
      "36 tile extents not measured",
      "1 tile extent left out",
      "1 tile extent left out",
    ]);
    expect(caveats[0]).toContain("not among all");
  });

  it("counts sets against the total without going negative", () => {
    expect(sweepCaveats(counts({ fusionSets: 2, fusionSetsTotal: 2 }))).toEqual([]);
  });
});

describe("a sweep's summary and rows", () => {
  const graph = compileDSL(`A = Tensor(8, 8, dtype=float16)\nB = Tensor(8, 4, dtype=float16)\nC = matmul(A, B)\n`).resolved;
  const sweep = sweepPlans(graph, { budget: { stepBytes: 64, accumulatorBytes: 64 }, maxRows: 2 });

  it("says how many rows are shown of how many unbeaten", () => {
    expect(sweep.counts.unbeaten).toBeGreaterThan(2);
    expect(sweepSummary(sweep)).toContain(`the first 2 of ${sweep.counts.unbeaten} unbeaten`);
    const all = sweepPlans(graph, { budget: { stepBytes: 64, accumulatorBytes: 64 } });
    expect(sweepSummary(all)).toContain(`${all.rows.length} plans unbeaten`);
  });

  it("recognises the row that is the current plan by tiling and chunks", () => {
    const row = sweep.rows[0];
    expect(rowIsCurrent(row, row.tiles, row.steps)).toBe(true);
    expect(rowIsCurrent(row, { ...row.tiles, C: [1, 1] }, row.steps)).toBe(false);
    expect(rowIsCurrent(row, row.tiles, { C: { chunk: { k: 1 } } })).toBe(Boolean(row.steps.C?.chunk?.k === 1));
    expect(rowIsCurrent(row, {}, {})).toBe(false);
    // An accumulator or order declaration does not make it a different tiling.
    expect(rowIsCurrent(row, row.tiles, Object.fromEntries(Object.entries(row.steps).map(([id, s]) => [id, { ...s, accumulator: "f32" as const }])))).toBe(Object.keys(row.steps).length > 0);
  });
});
