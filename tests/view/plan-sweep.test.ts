import { describe, expect, it } from "vitest";
import { compileDSL } from "../../src/parse/compiler";
import { sweepPlans, type Sweep } from "../../src/core/plan/sweep";
import { tilePlan } from "../../src/core/plan/plan";
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

  it("warns about capped searches even when all accepted extents were resolved", () => {
    expect(sweepCaveats(counts({ chunkingsCapped: 1, cappedUnresolved: 0 }))).toEqual([
      expect.stringContaining("fewer steps or a smaller step payload may exist among untried chunkings"),
    ]);
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

  it("recognises effective tiling and steps including accumulator defaults", () => {
    const row = sweep.rows[0];
    const rowPlan = tilePlan(graph, row.tiles, row.steps);
    const current = (tiles = row.tiles, steps = row.steps) => rowIsCurrent(row, tilePlan(graph, tiles, steps), rowPlan);
    expect(current()).toBe(true);
    expect(current({}, {})).toBe(false);
    expect(current(row.tiles, { C: { ...row.steps.C, accumulator: "f16" } })).toBe(false);
    expect(current(row.tiles, { C: { ...row.steps.C, accumulator: "f32", order: ["k"] } })).toBe(true);
  });

  it("normalizes whole chunks and distinguishes reduced-axis order", () => {
    const g = compileDSL('A = Tensor(2, 3, 4, dtype=float16)\nB = Tensor(3, 4, 5, dtype=float16)\nC = einsum("ijk,jkl->il", A, B)').resolved;
    const row = { tiles: { C: [1, 2] }, steps: {} };
    const plan = tilePlan(g, row.tiles, row.steps);
    expect(rowIsCurrent(row, tilePlan(g, row.tiles, { C: { chunk: { j: 3, k: 4 }, order: ["j", "k"], accumulator: "f32" } }), plan)).toBe(true);
    expect(rowIsCurrent(row, tilePlan(g, row.tiles, { C: { order: ["k", "j"] } }), plan)).toBe(false);
    expect(rowIsCurrent(row, tilePlan(g, row.tiles, { C: { chunk: { j: 1 } } }), plan)).toBe(false);
  });

  it("handles unstaged tensors and invalid declarations without crashing", () => {
    const g = compileDSL('A = Tensor(4, 4)\nB = relu(A)').resolved;
    const row = { tiles: { B: [2, 2] }, steps: {} };
    const plan = tilePlan(g, row.tiles);
    expect(rowIsCurrent(row, tilePlan(g, row.tiles, { B: {} }), plan)).toBe(true);
    expect(rowIsCurrent(row, tilePlan(g, row.tiles, { B: { accumulator: "f32" } }), plan)).toBe(false);
  });
});
