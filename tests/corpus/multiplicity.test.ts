/**
 * Read multiplicity against brute-force truth, for every registered operation.
 *
 * The truth is counted from `oracleDeps` alone: for each output element of the
 * box, every input element it lists is read once more. That is the relation the
 * `multiplicity` hook claims to count, derived from the definition rather than
 * from the hook or from `backward`. An exact count must equal it at every
 * element of the input, zeros included; an inexact one must be at least it.
 *
 * Driven off the fixture table like the oracle corpus, so a registered
 * operation is checked here whether it states a count, declares `readsOnce`,
 * or falls back to the bound.
 */

import { describe, expect, it } from "vitest";
import { Graph, resolveGraph } from "../../src/core/graph";
import { operandTally } from "../../src/core/multiplicity";
import { getOp } from "../../src/core/ops/index";
import { DEFAULT_LIMITS, type Limits } from "../../src/core/ops/limits";
import { OpCtx, OpSpec } from "../../src/core/ops/types";
import { Box, Region, fromBox, iv, points } from "../../src/core/region";
import { tallyAt } from "../../src/core/tally";
import { OP_FIXTURES } from "./op-fixtures";
import { G, randInt, rng } from "./harness";
import { flatIndex, regionToFlatSet, unflatIndex } from "./oracle";

/** How many outputs of `region` read each element of input `inSlot`, by enumeration. */
function truthCounts(spec: OpSpec, outSlot: number, region: Region, inSlot: number, ctx: OpCtx): Map<number, number> {
  const inShape = ctx.inShapes[inSlot];
  const counts = new Map<number, number>();
  for (const o of points(region)) {
    const read = new Set(spec.oracleDeps(outSlot, o, ctx)[inSlot].map((idx) => flatIndex(idx, inShape)));
    for (const f of read) counts.set(f, (counts.get(f) ?? 0) + 1);
  }
  return counts;
}

function randomBox(shape: number[], r: () => number): Box {
  return shape.map((e) => {
    const lo = randInt(r, 0, e);
    return iv(lo, randInt(r, lo + 1, e + 1));
  });
}

type Outcome = { exact: number; inexact: number };

/**
 * Check one node over the full output and random boxes, every output and input
 * slot. `truthCtx` is the context the oracle runs under, for the one case it
 * differs: a gather whose index values the operation is not told.
 */
function checkNode(
  graph: Graph,
  nodeId: string,
  limits?: Limits,
  edit?: (spec: OpSpec) => OpSpec,
  truthAttrs?: Record<string, unknown>
): Outcome {
  const g = resolveGraph(graph);
  const node = g.topo.find((n) => n.id === nodeId)!;
  const base = getOp(node.op)!;
  const spec = edit ? edit(base) : base;
  const ctx: OpCtx = {
    inShapes: g.shapesOf(node.inputs),
    outShapes: g.shapesOf(node.outputs),
    attrs: node.attrs,
    limits,
  };
  const truthCtx: OpCtx = truthAttrs ? { ...ctx, attrs: { ...node.attrs, ...truthAttrs } } : ctx;
  const r = rng(4242);
  const outcome: Outcome = { exact: 0, inexact: 0 };

  for (let outSlot = 0; outSlot < ctx.outShapes.length; outSlot++) {
    const outShape = ctx.outShapes[outSlot];
    const regions: Region[] = [fromBox(outShape.map((e) => iv(0, e)))];
    for (let k = 0; k < 16; k++) regions.push(fromBox(randomBox(outShape, r)));
    // Two overlapping boxes: an output element in both is still one reader.
    const a = randomBox(outShape, r);
    const b = randomBox(outShape, r);
    regions.push({ boxes: [a, b], exact: true, reasons: [] });

    for (const region of regions)
      for (let inSlot = 0; inSlot < ctx.inShapes.length; inSlot++) {
        const tally = operandTally(spec, outSlot, region, inSlot, ctx);
        const truth = truthCounts(base, outSlot, region, inSlot, truthCtx);
        const inShape = ctx.inShapes[inSlot];
        const n = inShape.reduce((x, y) => x * y, 1);
        for (let f = 0; f < n; f++) {
          const got = tallyAt(tally, unflatIndex(f, inShape));
          const want = truth.get(f) ?? 0;
          const where = `${node.op} out ${outSlot} ${JSON.stringify(region.boxes)} in ${inSlot} at ${unflatIndex(f, inShape)}`;
          if (tally.exact) expect(got, where).toBe(want);
          else expect(got, `${where}: an inexact count is a bound`).toBeGreaterThanOrEqual(want);
        }
        if (tally.exact) outcome.exact++;
        else outcome.inexact++;
      }
  }
  return outcome;
}

describe("read multiplicity: every registered op, via its fixture", () => {
  for (const fixture of OP_FIXTURES)
    it(`${fixture.op}${fixture.note ? ` (${fixture.note.split(":")[0]})` : ""}`, () => {
      const { exact, inexact } = checkNode(fixture.graph, fixture.nodeId);
      // A fixture that only ever produced bounds would test the bound and never
      // the count, unless being inexact is the operation's contract.
      if (fixture.inexactByDesign) expect(exact).toBe(0);
      else expect(exact, `${fixture.op}: no exact count was checked`).toBeGreaterThan(0);
      expect(exact + inexact).toBeGreaterThan(0);
    });
});

/**
 * Variants the one-per-op table does not reach: the counts that vary by
 * position, and the einsum shapes where output labels and operand labels
 * relate in each possible way.
 */
const VARIANTS: [string, Graph][] = [
  ["einsum diagonal", G({ A: [4, 4] }, [["n", "einsum", ["A"], ["Y"], { equation: "ii->i" }]])],
  ["einsum matrix-vector", G({ A: [3, 5], x: [5] }, [["n", "einsum", ["A", "x"], ["Y"], { equation: "ij,j->i" }]])],
  [
    "einsum label in neither the output nor the operand",
    G({ A: [3, 4], B: [5, 4] }, [["n", "einsum", ["A", "B"], ["Y"], { equation: "ik,jk->i" }]]),
  ],
  ["einsum outer product", G({ a: [3], b: [4] }, [["n", "einsum", ["a", "b"], ["Y"], { equation: "i,j->ij" }]])],
  ["matmul of a tensor with itself", G({ A: [4, 4] }, [["n", "matmul", ["A", "A"], ["Y"]]])],
  ["elementwise of a tensor with itself", G({ X: [3, 4] }, [["n", "elementwise", ["X", "X"], ["Y"], { fn: "multiply", nary: 2 }]])],
  ["elementwise rank-0 operand", G({ X: [3, 4], s: [] }, [["n", "elementwise", ["X", "s"], ["Y"], { fn: "multiply", nary: 2 }]])],
  ["pad replicate", G({ X: [4, 3] }, [["n", "pad", ["X"], ["Y"], { pads: [[2, 3], [1, 0]], mode: "edge" }]])],
  ["pad constant", G({ X: [4, 3] }, [["n", "pad", ["X"], ["Y"], { pads: [[2, 1], [0, 2]], mode: "constant" }]])],
  [
    "conv stride 1, padded",
    G({ X: [1, 2, 6, 5], W: [3, 2, 3, 3] }, [
      ["n", "conv", ["X", "W"], ["Y"], { stride: [1, 1], pads: [[1, 1], [1, 1]], dilation: [1, 1], groups: 1 }],
    ]),
  ],
  [
    "conv dilated and strided",
    G({ X: [1, 1, 9] , W: [2, 1, 3] }, [
      ["n", "conv", ["X", "W"], ["Y"], { stride: [2], pads: [[2, 1]], dilation: [2], groups: 1 }],
    ]),
  ],
  [
    "pool overlapping windows",
    G({ X: [1, 1, 7, 6] }, [["n", "pool", ["X"], ["Y"], { kind: "avg", kernelShape: [3, 2], stride: [1, 1], pads: [[1, 1], [0, 0]] }]]),
  ],
  ["cumsum forward", G({ X: [5, 3] }, [["n", "cumulative_sum", ["X"], ["Y"], { axis: 0, reverse: false }]])],
  ["slice with steps", G({ X: [9, 4] }, [["n", "slice", ["X"], ["Y"], { starts: [1, 0], stops: [9, 4], steps: [3, 1] }]])],
  ["normalize without affine params", G({ X: [3, 4, 2] }, [["n", "normalize", ["X"], ["Y"], { kind: "rms_norm", axes: [1, 2] }]])],
  ["expand with a new leading axis", G({ X: [1, 3] }, [["n", "broadcast_to", ["X"], ["Y"], { shape: [4, 2, 3] }]])],
  ["reduce with keepdim", G({ X: [3, 4] }, [["n", "reduce", ["X"], ["Y"], { fn: "max", axes: [1], keepdim: true }]])],
];

describe("read multiplicity: variants", () => {
  for (const [name, graph] of VARIANTS)
    it(name, () => {
      const { exact } = checkNode(graph, "n");
      expect(exact, `${name}: no exact count was checked`).toBeGreaterThan(0);
    });

  it("a gather that is not told its index values bounds every row", () => {
    const graph = G({ D: [6, 3], I: [5] }, [["n", "take", ["D", "I"], ["Y"], { axis: 0 }]]);
    graph.tensors.I.dtype = "i32";
    const { inexact } = checkNode(graph, "n", undefined, undefined, { indexValues: [5, 0, 2, 2, 2] });
    expect(inexact).toBeGreaterThan(0);
  });
});

/**
 * The fallback branches, at shapes the oracle can enumerate: every cap lowered
 * until it is reached, and the operation's own count taken away so that
 * `readsOnce` and the bound are what answer.
 */
describe("read multiplicity: conservative branches", () => {
  const low: Limits = { stridedEnum: 2, diagEnum: 2, reshapeRuns: 2, maxBoxes: 2, tallyPieces: 2 };
  const cases: [string, Graph][] = [
    ...OP_FIXTURES.filter((f) => !f.inexactByDesign).map((f): [string, Graph] => [f.op, f.graph]),
    ...VARIANTS,
  ];

  it("lowered limits keep every count a bound, and reach the bound", () => {
    let inexact = 0;
    for (const [, graph] of cases) {
      const nodeId = graph.nodes[0].id;
      inexact += checkNode(graph, nodeId, low).inexact;
    }
    expect(inexact).toBeGreaterThan(0);
  });

  it("an operation without a count of its own is bounded by its box", () => {
    let inexact = 0;
    for (const [, graph] of cases) {
      const nodeId = graph.nodes[0].id;
      const strip = (spec: OpSpec): OpSpec => ({ ...spec, multiplicity: undefined, readsOnce: undefined });
      inexact += checkNode(graph, nodeId, DEFAULT_LIMITS, strip).inexact;
    }
    expect(inexact).toBeGreaterThan(0);
  });

  it("the elements counted are the elements backward names, where both are exact", () => {
    for (const fixture of OP_FIXTURES) {
      const g = resolveGraph(fixture.graph);
      const node = g.topo.find((n) => n.id === fixture.nodeId)!;
      const spec = getOp(node.op)!;
      const ctx: OpCtx = { inShapes: g.shapesOf(node.inputs), outShapes: g.shapesOf(node.outputs), attrs: node.attrs };
      const r = rng(7);
      for (let k = 0; k < 12; k++) {
        const box = randomBox(ctx.outShapes[0], r);
        const read = spec.backward(0, box, ctx);
        read.forEach((region, inSlot) => {
          const tally = operandTally(spec, 0, fromBox(box), inSlot, ctx);
          if (!tally.exact || !region.exact) return;
          const support = new Set<number>();
          for (const p of tally.pieces) for (const f of regionToFlatSet(fromBox(p.box), ctx.inShapes[inSlot])) support.add(f);
          expect([...support].sort((x, y) => x - y)).toEqual(
            [...regionToFlatSet(region, ctx.inShapes[inSlot])].sort((x, y) => x - y)
          );
        });
      }
    }
  });
});
