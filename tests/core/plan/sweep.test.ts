/**
 * The sweep against brute force.
 *
 * `sweepPlans` measures each tiled tensor on its own and combines the options
 * nothing beats, on the claim that a plan's ranked figures separate by tensor.
 * The brute force here makes no such claim: it builds every plan - every set of
 * tiled tensors, every power-of-two tile extent of every one of them - evaluates
 * each whole with `planReport`, takes each step payload as the largest over
 * every task and every step, and only then discards the beaten plans. The two
 * must list the same plans with the same figures, tied plans included.
 *
 * What the brute force shares with the sweep: the tile and chunk extents tried
 * (`extentOptions`) and the rule that a tile takes its largest chunk within
 * the budget. Its graphs have one reduced axis each; the chunking cap and the
 * sets that are not exact have tests of their own below.
 */

import { describe, expect, it } from "vitest";
import { compileDSL } from "../../../src/parse/compiler";
import type { ResolvedGraph } from "../../../src/core/graph";
import { graphOutputs } from "../../../src/core/graph";
import { planReport, taskSupplyOf } from "../../../src/core/plan/interfaces";
import { tilePlan, type StepSpec } from "../../../src/core/plan/plan";
import { tiles as tilesOf, PlanError } from "../../../src/core/plan/tile-family";
import { payloadOf, stagingOf, stepPayload, stepsOf, taskSteps } from "../../../src/core/plan/steps";
import { extentOptions, sweepPlans, type Budget } from "../../../src/core/plan/sweep";

const graphOf = (src: string) => compileDSL(src).resolved;

const MLP = `X = Tensor(4, 4, dtype=float16)
W1 = Parameter(4, 4, dtype=float16)
W2 = Parameter(4, 4, dtype=float16)
H = matmul(X, W1)
Y = matmul(H, W2)
`;
const GEMM = `A = Tensor(6, 8, dtype=float16)
B = Tensor(8, 4, dtype=float16)
C = matmul(A, B)
`;
const EPILOGUE = `A = Tensor(4, 4, dtype=float16)
B = Tensor(4, 2, dtype=float16)
Bias = Parameter(2, dtype=float16)
C = matmul(A, B)
D = add(C, Bias)
Y = relu(D)
`;
const SOFTMAX = `X = Tensor(4, 8, dtype=float16)
W = Parameter(8, 4, dtype=float16)
L = matmul(X, W)
P = softmax(L, axis=-1)
`;

// A task past the first and the middle reads more: the tiles over S read two
// tensors where the tiles over A read one.
const CONCAT_TASK = `A = Tensor(6, 4, dtype=float16)
B1 = Tensor(2, 4, dtype=float16)
B2 = Tensor(2, 4, dtype=float16)
S = add(B1, B2)
Y = concat([A, S], axis=0)
`;
// A later step reads more: the second half of k reaches the fused S.
const CONCAT_STEP = `A1 = Tensor(2, 4, dtype=float16)
P = Tensor(2, 4, dtype=float16)
Q = Tensor(2, 4, dtype=float16)
B = Tensor(8, 2, dtype=float16)
S = add(P, Q)
Ac = concat([A1, S], axis=1)
C = matmul(Ac, B)
`;
// Two plans tie on every figure: one tensor's payload is the plan's, so the
// other's smaller payload changes nothing.
const TIE = `X = Tensor(2, 3, dtype=float16)
W = Parameter(3, 4, dtype=float16)
H = relu(X)
Y = matmul(H, W)
`;

function cartesian<T>(lists: T[][]): T[][] {
  return lists.reduce<T[][]>((acc, list) => acc.flatMap((head) => list.map((item) => [...head, item])), [[]]);
}

const beats = (a: number[], b: number[]) => a.every((v, i) => v <= b[i]) && a.some((v, i) => v < b[i]);

type Brute = { tiles: Record<string, number[]>; steps: Record<string, StepSpec>; key: number[] };

function bruteForce(graph: ResolvedGraph, budget: Budget): Brute[] {
  const outputs = graphOutputs(graph).map((t) => t.id);
  const intermediates = Object.values(graph.tensors).filter((t) => t.producer && !outputs.includes(t.id)).map((t) => t.id);
  const sets = cartesian(intermediates.map((id) => [[], [id]])).map((parts) => [...outputs, ...parts.flat()].sort());
  const plans: Brute[] = [];

  for (const set of sets)
    for (const combo of cartesian(set.map((id) => cartesian(graph.tensors[id].resolved.map(extentOptions))))) {
      const tiles = Object.fromEntries(set.map((id, i) => [id, combo[i]]));
      const steps: Record<string, StepSpec> = {};
      let largest = 0;
      let within = true;

      for (const id of set) {
        const bare = tilePlan(graph, tiles);
        const staging = stagingOf(bare, id);
        if (!staging.staged) {
          let worst = 0;
          for (const coord of tilesOf(bare.families.get(id)!))
            worst = Math.max(worst, payloadOf(bare, taskSupplyOf(bare, { tensorId: id, coord }).demand).bytes.value!);
          if (worst > budget.stepBytes) within = false;
          largest = Math.max(largest, worst);
          continue;
        }
        // Every chunking, fewest steps first; the payload is the largest over
        // every task and every step.
        const [axis] = staging.reduced;
        expect(staging.reduced.length).toBe(1);
        const chunks = extentOptions(axis.extent).sort((a, b) => b - a);
        let found = false;
        for (const chunk of chunks) {
          const spec: StepSpec = { chunk: { [axis.label]: chunk } };
          const plan = tilePlan(graph, tiles, { [id]: spec });
          let worst = 0;
          let accumulator = 0;
          for (const coord of tilesOf(plan.families.get(id)!)) {
            const task = taskSteps(plan, { tensorId: id, coord });
            if (!("staging" in task)) throw new Error("unstaged");
            accumulator = Math.max(accumulator, task.accumulator.bytes.value!);
            for (const step of stepsOf(plan, task)) worst = Math.max(worst, stepPayload(plan, step).bytes.value!);
          }
          if (budget.accumulatorBytes !== undefined && accumulator > budget.accumulatorBytes) break;
          if (worst <= budget.stepBytes) {
            if (chunk !== axis.extent) steps[id] = spec;
            largest = Math.max(largest, worst);
            found = true;
            break;
          }
        }
        if (!found) within = false;
      }
      if (!within) continue;

      const total = planReport(tilePlan(graph, tiles, steps)).total!;
      plans.push({
        tiles,
        steps,
        key: [total.recomputed.value!, total.read.value! + total.written.value!, total.tasks, largest],
      });
    }
  return plans.filter((a) => !plans.some((b) => beats(b.key, a.key)));
}

const idOf = (p: { tiles: Record<string, number[]>; steps: Record<string, StepSpec> }) =>
  JSON.stringify([Object.entries(p.tiles).sort(), Object.entries(p.steps).sort()]);

describe("the sweep lists what brute force lists", () => {
  const cases: [string, string, Budget][] = [
    ["gemm, a budget that forces chunks", GEMM, { stepBytes: 48 }],
    ["gemm, a budget nothing exceeds", GEMM, { stepBytes: 1e9 }],
    ["gemm, a bounded accumulator", GEMM, { stepBytes: 64, accumulatorBytes: 32 }],
    ["mlp, a tight budget", MLP, { stepBytes: 40 }],
    ["mlp, a bounded accumulator", MLP, { stepBytes: 96, accumulatorBytes: 64 }],
    ["an epilogue", EPILOGUE, { stepBytes: 32 }],
    ["a tensor without steps", SOFTMAX, { stepBytes: 48 }],
    ["a task that reads more than the first and the middle", CONCAT_TASK, { stepBytes: 20 }],
    ["a step that reads more than the first", CONCAT_STEP, { stepBytes: 40 }],
    ["plans tied on every figure", TIE, { stepBytes: 10 }],
  ];

  for (const [name, src, budget] of cases)
    it(name, () => {
      const graph = graphOf(src);
      const brute = bruteForce(graph, budget);
      const sweep = sweepPlans(graph, { budget, maxRows: 10_000, maxPlans: 1_000_000 });
      expect(sweep.counts.combinationsSkipped).toBe(0);
      expect(sweep.counts.fusionSets).toBe(sweep.counts.fusionSetsTotal);
      expect(sweep.rows.length).toBe(sweep.counts.unbeaten);
      expect(brute.length).toBeGreaterThan(0);

      expect(sweep.rows.map(idOf).sort()).toEqual(brute.map(idOf).sort());
      const byId = new Map(brute.map((p) => [idOf(p), p]));
      for (const row of sweep.rows) {
        const truth = byId.get(idOf(row))!;
        // Every row is the plan's own report, and within the budget.
        const alone = planReport(tilePlan(graph, row.tiles, row.steps));
        expect(row.total).toEqual(alone.total);
        expect([row.total.recomputed.value, row.total.read.value! + row.total.written.value!, row.total.tasks, row.stepBytes]).toEqual(truth.key);
        expect(row.stepBytes).toBeLessThanOrEqual(budget.stepBytes);
        if (budget.accumulatorBytes !== undefined && row.accumulatorBytes !== null)
          expect(row.accumulatorBytes).toBeLessThanOrEqual(budget.accumulatorBytes);
      }
    });
});

describe("the budget", () => {
  const graph = graphOf(`M = 64
N = 64
K = 128
A = Tensor(M, K, dtype=float16)
B = Tensor(K, N, dtype=float16)
C = matmul(A, B)
`);
  // With no bound on the accumulator, one task computing all of C beats every
  // smaller tile on recomputation, bytes and task count, so these bound it at
  // a 16 x 16 float32 tile.
  const bounds = { accumulatorBytes: 1024 };
  // The task budget only bounds the work here: a 16 x 16 tiling is 16 tasks.
  const limits = { maxRows: 10_000, taskBudget: 32 };
  const square = (r: { tiles: Record<string, number[]> }) => r.tiles.C[0] === 16 && r.tiles.C[1] === 16;
  // A 16 x 16 tile reads 16 x BK of A and BK x 16 of B per step: 64 * BK bytes.
  const chunkAt = (stepBytes: number) =>
    sweepPlans(graph, { budget: { stepBytes, ...bounds }, ...limits }).rows.find(square)?.steps.C?.chunk?.k;

  it("takes the largest chunk within it, and changes at the exact byte", () => {
    expect(chunkAt(2048)).toBe(32);
    expect(chunkAt(2047)).toBe(16);
    expect(chunkAt(64 * 128)).toBeUndefined(); // the whole axis: one step, nothing declared
    expect(chunkAt(64 * 128 - 1)).toBe(64);
  });

  it("leaves out a tile extent no chunk brings within it", () => {
    // One element of K under a 16 x 16 tile is still 64 bytes.
    const sweep = sweepPlans(graph, { budget: { stepBytes: 63, ...bounds }, ...limits });
    expect(sweep.rows.some(square)).toBe(false);
    expect(sweep.counts.overBudget).toBeGreaterThan(0);
    for (const row of sweep.rows) expect(row.stepBytes).toBeLessThanOrEqual(63);
  });

  it("reports the row's steps and accumulator", () => {
    const row = sweepPlans(graph, { budget: { stepBytes: 2048, ...bounds }, ...limits }).rows.find(square)!;
    expect(row.perTensor.C).toEqual({ stepsPerTask: 4, stepBytes: 2048, accumulatorBytes: 1024 });
    expect(row.accumulatorBytes).toBe(1024);
  });

  it("counts what it did not measure and did not form", () => {
    const sweep = sweepPlans(graph, { budget: { stepBytes: 99 * 1024, ...bounds }, taskBudget: 64 });
    expect(sweep.counts.overTasks).toBeGreaterThan(0);
    // H and Y each keep several options, and one combination is all that is allowed.
    const capped = sweepPlans(graphOf(MLP), { budget: { stepBytes: 40 }, maxPlans: 1 });
    expect(capped.counts.combined).toBe(1);
    expect(capped.counts.combinationsSkipped).toBeGreaterThan(0);
  });

  it("without an accumulator bound, lists the one task that computes everything", () => {
    const sweep = sweepPlans(graph, { budget: { stepBytes: 99 * 1024 }, ...limits });
    expect(sweep.rows[0].tiles).toEqual({ C: [64, 64] });
    expect(sweep.rows[0].total.tasks).toBe(1);
    expect(sweep.rows[0].accumulatorBytes).toBe(64 * 64 * 4);
  });

  it("refuses a budget that is not a positive number", () => {
    const codeOf = (options: Parameters<typeof sweepPlans>[1]) => {
      try {
        sweepPlans(graph, options);
      } catch (e) {
        return (e as PlanError).code;
      }
      return null;
    };
    for (const n of [0, -1, NaN, Infinity]) {
      expect(codeOf({ budget: { stepBytes: n } })).toBe("PLAN_BUDGET");
      expect(codeOf({ budget: { stepBytes: 64, accumulatorBytes: n } })).toBe("PLAN_BUDGET");
      expect(codeOf({ budget: { stepBytes: 64 }, taskBudget: n })).toBe("PLAN_BUDGET");
    }
  });
});

describe("what is measured", () => {
  it("is every task: a row over the budget on an unprobed task is not listed", () => {
    const sweep = sweepPlans(graphOf(CONCAT_TASK), { budget: { stepBytes: 20 }, tiled: [["Y"]], maxRows: 10_000 });
    // Tiles two rows tall read 16 bytes of A and 32 of B1 and B2 together.
    expect(sweep.rows.some((r) => r.tiles.Y[0] === 2 && r.tiles.Y[1] === 4)).toBe(false);
    for (const row of sweep.rows) expect(row.stepBytes).toBeLessThanOrEqual(20);
  });

  it("is every step: the payload is the largest step's", () => {
    // One task over all of C, k in two halves: the first half reads A1 and four
    // rows of B, 32 bytes; the second reads P and Q and four rows of B, 48.
    const sweep = sweepPlans(graphOf(CONCAT_STEP), { budget: { stepBytes: 48 }, tiled: [["C"]], maxRows: 10_000 });
    const row = sweep.rows.find((r) => r.tiles.C[0] === 2 && r.tiles.C[1] === 2)!;
    expect(row.steps.C).toEqual({ chunk: { k: 4 } });
    expect(row.stepBytes).toBe(48);
    // One byte less and the halves no longer fit.
    const tighter = sweepPlans(graphOf(CONCAT_STEP), { budget: { stepBytes: 47 }, tiled: [["C"]], maxRows: 10_000 });
    expect(tighter.rows.find((r) => r.tiles.C[0] === 2 && r.tiles.C[1] === 2)!.steps.C).toEqual({ chunk: { k: 2 } });
  });

  it("keeps the whole axis among the chunkings when there are too many to try", () => {
    const graph = graphOf(`A = Tensor(2, 64, 64, 64, dtype=float16)
B = Tensor(64, 64, 64, 2, dtype=float16)
C = einsum("abcd,bcde->ae", A, B)
`);
    const sweep = sweepPlans(graph, { budget: { stepBytes: 1e12 }, tiled: [["C"]] });
    expect(sweep.counts.chunkingsCapped).toBeGreaterThan(0);
    for (const row of sweep.rows) expect(row.steps).toEqual({});
  });
});

describe("a set with a figure that is not exact", () => {
  it("is not searched, and is counted", () => {
    // A strided slice past its enumeration limit over-approximates what a task
    // of Y reads, so the work the set does once is not one number.
    const graph = graphOf(`X = Tensor(2048, dtype=float16)
H = relu(X)
S = H[::2]
Y = relu(S)
`);
    const sweep = sweepPlans(graph, { budget: { stepBytes: 1e9 }, tiled: [["Y"]] });
    expect(sweep.counts).toMatchObject({ fusionSets: 0, setsInexact: 1 });
    expect(sweep.rows).toEqual([]);
  });
});

describe("the sets of tiled tensors", () => {
  it("are checked as tilings, each tensor and each set once", () => {
    const graph = graphOf(MLP);
    const codeOf = (tiled: string[][]) => {
      try {
        sweepPlans(graph, { budget: { stepBytes: 1e9 }, tiled });
      } catch (e) {
        return (e as PlanError).code;
      }
      return null;
    };
    expect(codeOf([["Nope"]])).toBe("PLAN_UNKNOWN_TENSOR");
    expect(codeOf([["X"]])).toBe("PLAN_GRAPH_INPUT");
    const once = sweepPlans(graph, { budget: { stepBytes: 1e9 }, tiled: [["Y"]], maxRows: 10_000 });
    const twice = sweepPlans(graph, { budget: { stepBytes: 1e9 }, tiled: [["Y", "Y"], ["Y"]], maxRows: 10_000 });
    expect(twice.rows).toEqual(once.rows);
    expect(twice.counts).toEqual(once.counts);
  });

  it("past the plan limit are counted as not searched", () => {
    const sweep = sweepPlans(graphOf(MLP), { budget: { stepBytes: 40 }, maxPlans: 1 });
    expect(sweep.counts.fusionSets + sweep.counts.setsSkipped).toBe(2);
    expect(sweep.counts.setsSkipped).toBe(1);
  });

  it("can be given, and are otherwise every fusion of a few intermediates", () => {
    const graph = graphOf(MLP);
    const all = sweepPlans(graph, { budget: { stepBytes: 1e9 } });
    expect(all.counts.fusionSets).toBe(2);
    const fused = sweepPlans(graph, { budget: { stepBytes: 1e9 }, tiled: [["Y"]] });
    expect(fused.counts.fusionSets).toBe(1);
    for (const row of fused.rows) expect(Object.keys(row.tiles)).toEqual(["Y"]);
  });
});
