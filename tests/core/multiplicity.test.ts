/**
 * Read multiplicity over whole cones, as fixed cases: the numbers
 * `docs/dependency_multiplicity.md` derives for a matmul tile, and what a cone
 * reports for the operations around one. Per-operation correctness against
 * brute force lives in `tests/corpus/multiplicity.test.ts`.
 */

import { describe, expect, it } from "vitest";
import { executeQuery } from "../../src/core/executor";
import type { ResolvedGraph } from "../../src/core/graph";
import { operandReads, type OperandReads } from "../../src/core/multiplicity";
import { cumsumOp } from "../../src/core/ops/scan";
import { DEFAULT_LIMITS } from "../../src/core/ops/limits";
import { tallyAt } from "../../src/core/tally";
import { Region, box, fromBox } from "../../src/core/region";
import { compileDSL } from "../../src/parse/compiler";
import { supplyOf } from "../../src/core/plan/interfaces";
import { tilePlan } from "../../src/core/plan/plan";

const GEMM = "M = 4\nN = 6\nK = 8\nA = Tensor(M, K, dtype=float16)\nB = Tensor(K, N, dtype=float16)\nC = matmul(A, B)\n";

function readsFor(dsl: string, tensorId: string, region: Region): { graph: ResolvedGraph; rows: OperandReads[] } {
  const { resolved } = compileDSL(dsl);
  const back = executeQuery(resolved, { tensorId, region, direction: "backward" }).backward!;
  return { graph: resolved, rows: operandReads(resolved, back.tensors) };
}

const row = (rows: OperandReads[], tensorId: string) => rows.find((r) => r.tensorId === tensorId)!;
const values = (r: OperandReads) => ({
  reads: r.reads.value,
  elements: r.elements.value,
  once: r.once.value,
  reused: r.reused.value,
  min: r.min.value,
  max: r.max.value,
});

describe("read multiplicity of a matmul tile", () => {
  it("reports an unsafe read total as unknown while retaining safe element counts", () => {
    const n = 262145;
    const { rows } = readsFor(`A = Tensor(${n}, ${n})\nB = Tensor(${n}, ${n})\nC = matmul(A, B)\n`,
      "C", fromBox(box([0, n], [0, n])));
    for (const r of rows) {
      expect(r.reads).toMatchObject({ value: null, status: "unknown" });
      expect(r.reads.reasons).toContain("read multiplicity exceeds safe integer range");
      expect(r.mean).toMatchObject({ value: null, status: "unknown" });
      expect(r.elements).toMatchObject({ value: n * n, status: "exact" });
      expect(r.max).toMatchObject({ value: n, status: "exact" });
    }
  });

  it("reads each element of A by n outputs and each of B by m", () => {
    // m = 2, n = 3, K = 8
    const { rows } = readsFor(GEMM, "C", fromBox(box([0, 2], [0, 3])));
    expect(values(row(rows, "A"))).toEqual({ reads: 48, elements: 16, once: 0, reused: 16, min: 3, max: 3 });
    expect(values(row(rows, "B"))).toEqual({ reads: 48, elements: 24, once: 0, reused: 24, min: 2, max: 2 });
    expect(row(rows, "A").mean).toMatchObject({ value: 3, status: "exact" });
    // R = 2mn / (m + n): every operand occurrence over every element read.
    const a = row(rows, "A");
    const b = row(rows, "B");
    expect((a.reads.value! + b.reads.value!) / (a.elements.value! + b.elements.value!)).toBeCloseTo((2 * 2 * 3) / (2 + 3));
  });

  it("reads everything once for a single output, and for a diagonal", () => {
    expect(values(row(readsFor(GEMM, "C", fromBox(box([1, 2], [2, 3]))).rows, "A")).once).toBe(8);
    const diagonal = { boxes: [box([0, 1], [0, 1]), box([1, 2], [1, 2]), box([2, 3], [2, 3])], exact: true, reasons: [] };
    const { rows } = readsFor(GEMM, "C", diagonal);
    expect(values(row(rows, "A"))).toMatchObject({ reads: 24, elements: 24, once: 24, reused: 0, max: 1 });
    expect(values(row(rows, "B"))).toMatchObject({ reads: 24, elements: 24, once: 24, reused: 0, max: 1 });
  });

  it("varies by row for a selection that is not a rectangle", () => {
    // Row 0 holds three outputs, row 1 one: A's row 0 is read three times, row 1 once.
    const shape = { boxes: [box([0, 1], [0, 3]), box([1, 2], [0, 1])], exact: true, reasons: [] };
    const { rows } = readsFor(GEMM, "C", shape);
    expect(values(row(rows, "A"))).toEqual({ reads: 32, elements: 16, once: 8, reused: 8, min: 1, max: 3 });
    // Column 0 is read by both rows; columns 1 and 2 by row 0 only.
    expect(values(row(rows, "B"))).toEqual({ reads: 32, elements: 24, once: 16, reused: 8, min: 1, max: 2 });
  });

  it("states bytes in the operand's dtype", () => {
    const { rows } = readsFor(GEMM, "C", fromBox(box([0, 2], [0, 3])));
    expect(row(rows, "A").reusedBytes).toMatchObject({ value: 32, status: "exact" });
    expect(row(rows, "A").onceBytes).toMatchObject({ value: 0, status: "exact" });
  });

  it("gives one row for a tensor read through two slots, counting both", () => {
    const { rows } = readsFor("A = Tensor(4, 4)\nC = matmul(A, A)\n", "C", fromBox(box([0, 2], [0, 2])));
    expect(rows).toHaveLength(1);
    expect(rows[0].slots).toEqual([0, 1]);
    // Two row bands and two column bands of a 4 x 4 tensor, each element of a
    // band read by 2 outputs, and the 2 x 2 corner in both bands read by 4.
    expect(values(rows[0])).toEqual({ reads: 32, elements: 12, once: 0, reused: 12, min: 2, max: 4 });
  });
});

describe("bounded scan multiplicity", () => {
  it("bounds every input for both directions, offsets, and small budgets", () => {
    const n = 11;
    for (const reverse of [false, true])
      for (const budget of [1, 2, 5, 64])
        for (let lo = 0; lo < n; lo++)
          for (let hi = lo + 1; hi <= n; hi++) {
            const tally = cumsumOp.multiplicity!(0, box([lo, hi]), 0, {
              inShapes: [[n]], outShapes: [[n]], attrs: { axis: 0, reverse },
              limits: { ...DEFAULT_LIMITS, tallyPieces: budget },
            });
            expect(tally.pieces.length).toBeLessThanOrEqual(budget);
            for (let i = 0; i < n; i++) {
              const truth = reverse ? Math.max(0, Math.min(hi, i + 1) - lo) : Math.max(0, hi - Math.max(lo, i));
              const got = tallyAt(tally, [i]);
              if (tally.exact) expect(got).toBe(truth);
              else expect(got).toBeGreaterThanOrEqual(truth);
              expect(got > 0).toBe(truth > 0);
            }
          }
  });

  it("handles a billion positions with only a budget-sized tally", () => {
    const n = 1000000000;
    const lo = 123;
    const hi = n - 123;
    for (const reverse of [false, true]) {
      const tally = cumsumOp.multiplicity!(0, box([lo, hi]), 0, {
        inShapes: [[n]], outShapes: [[n]], attrs: { axis: 0, reverse },
        limits: { ...DEFAULT_LIMITS, tallyPieces: 64 },
      });
      expect(tally.pieces.length).toBeLessThanOrEqual(64);
      expect(tally.exact).toBe(false);
      expect(tally.reasons).toContain("multiplicity cap");
      for (const i of [0, lo - 1, lo, lo + 1, Math.floor(n / 2), hi - 1, hi, n - 1]) {
        const truth = reverse ? Math.max(0, Math.min(hi, i + 1) - lo) : Math.max(0, hi - Math.max(lo, i));
        expect(tallyAt(tally, [i])).toBeGreaterThanOrEqual(truth);
        expect(tallyAt(tally, [i]) > 0).toBe(truth > 0);
      }
    }
  });
});

describe("read multiplicity along a cone", () => {
  it("gives a row per operation and operand, in graph order", () => {
    const dsl = `${GEMM}bias = Tensor(N, dtype=float16)\nY = add(C, bias)\nZ = relu(Y)\n`;
    const { rows } = readsFor(dsl, "Z", fromBox(box([0, 4], [0, 2])));
    expect(rows.map((r) => [r.op, r.tensorId])).toEqual([
      ["matmul", "A"],
      ["matmul", "B"],
      ["add", "C"],
      ["add", "bias"],
      ["relu", "Y"],
    ]);
    // A bias element is broadcast down all four rows of the tile.
    expect(values(row(rows, "bias"))).toMatchObject({ elements: 2, min: 4, max: 4 });
    expect(values(row(rows, "Y"))).toMatchObject({ elements: 8, once: 8, reused: 0 });
    // The matmul is counted over what the cone needs of C, not over all of it.
    expect(values(row(rows, "A"))).toMatchObject({ elements: 32, min: 2, max: 2 });
  });

  it("counts a softmax row once per output in it", () => {
    const { rows } = readsFor("X = Tensor(3, 5)\nY = softmax(X, axis=-1)\n", "Y", fromBox(box([0, 2], [1, 4])));
    expect(values(row(rows, "X"))).toEqual({ reads: 30, elements: 10, once: 0, reused: 10, min: 3, max: 3 });
  });

  it("marks a count over an unknown index as bounds, and the single-use count as approximate", () => {
    const dsl = "W = Tensor(10, 4)\nIdx = Tensor(3, dtype=int32)\nE = take(W, Idx, axis=0)\n";
    const { rows } = readsFor(dsl, "E", fromBox(box([0, 2], [0, 4])));
    const w = row(rows, "W");
    expect(w.exact).toBe(false);
    expect(w.reasons).toContain("data-dependent index");
    expect(w.reads.status).toBe("upper");
    expect(w.reused.status).toBe("upper");
    expect(w.max.status).toBe("upper");
    expect(w.once.status).toBe("approximate");
    expect(w.mean.status).toBe("approximate");
    // The index tensor is still counted exactly: each position is read by its row.
    expect(row(rows, "Idx")).toMatchObject({ exact: true });
    expect(values(row(rows, "Idx"))).toMatchObject({ elements: 2, min: 4, max: 4 });
  });
});

/* Inside a plan task the counts are over what the task computes - its tile and
   the untiled tensors it computes on the way - so fusing an operand into the
   task brings that operand's own operation, and its reuse, into the count. */
describe("operand reuse inside a plan task", () => {
  const CHAIN = "A = Tensor(32, 32, dtype=float16)\nB = Tensor(32, 32, dtype=float16)\n" +
    "W = Tensor(32, 16, dtype=float16)\nC = matmul(A, B)\nY = matmul(C, W)\n";
  const { resolved } = compileDSL(CHAIN);
  const taskReads = (tiles: Record<string, number[]>) => {
    const supply = supplyOf(tilePlan(resolved, tiles), { tensorId: "Y", coord: [1, 1] });
    return operandReads(resolved, supply.work);
  };

  it("counts only the task's own operation when its operand is tiled", () => {
    const rows = taskReads({ C: [8, 8], Y: [8, 4] });
    expect(rows.map((r) => [r.outputIds, r.tensorId])).toEqual([[["Y"], "C"], [["Y"], "W"]]);
    // An 8 x 4 tile of Y: a row of C is read by the tile's 4 columns, a column of W by its 8 rows.
    expect(values(row(rows, "C"))).toMatchObject({ elements: 8 * 32, min: 4, max: 4 });
    expect(values(row(rows, "W"))).toMatchObject({ elements: 32 * 4, min: 8, max: 8 });
  });

  it("brings a fused operand's operation into the count, over the band the task computes", () => {
    const rows = taskReads({ Y: [8, 4] });
    expect(rows.map((r) => [r.outputIds, r.tensorId])).toEqual([[["C"], "A"], [["C"], "B"], [["Y"], "C"], [["Y"], "W"]]);
    // The task computes C[8:16, :] itself: each of A's rows there is read by all
    // 32 columns of that band, and each element of B by its 8 rows.
    expect(values(row(rows, "A"))).toMatchObject({ elements: 8 * 32, min: 32, max: 32 });
    expect(values(row(rows, "B"))).toMatchObject({ elements: 32 * 32, min: 8, max: 8 });
    expect(values(row(rows, "C"))).toMatchObject({ elements: 8 * 32, min: 4, max: 4 });
  });
});
