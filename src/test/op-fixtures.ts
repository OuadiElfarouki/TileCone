/**
 * One representative instance of every registered operation.
 *
 * This table is the single place that says "here is what this op looks like",
 * and both registry-wide properties read it: the oracle corpus checks each
 * fixture against brute-force truth, and the adjointness law checks each
 * fixture's `forward` against its own `backward`. `registry.test.ts` asserts the
 * table covers `listOps()`, so a newly registered operation fails the suite
 * until it is given a fixture, rather than shipping untested.
 *
 * Shapes are deliberately tiny and mutually prime where it costs nothing: the
 * oracle enumerates every element of every tensor, and an off-by-one in an axis
 * mapping hides behind square shapes.
 */

import { Graph, Tensor } from "../core/graph";
import { DType } from "../core/dtypes";

export type OpFixture = {
  /** Registered op name this exercises. */
  op: string;
  /** A graph whose node `nodeId` is the single instance under test. */
  graph: Graph;
  nodeId: string;
  /** Why this instance and not a simpler one, when that is not obvious. */
  note?: string;
};

type NodeSpec = [
  id: string,
  op: string,
  ins: string[],
  outs: string[],
  attrs?: Record<string, unknown>,
];

function build(
  inputs: Record<string, { shape: number[]; dtype?: DType }>,
  nodes: NodeSpec[]
): Graph {
  const tensors: Record<string, Tensor> = {};
  for (const [id, { shape, dtype }] of Object.entries(inputs))
    tensors[id] = { id, name: id, shape, dtype: dtype ?? "f32" };
  for (const [, , , outs] of nodes)
    for (const o of outs)
      if (!tensors[o]) tensors[o] = { id: o, name: o, shape: [], dtype: "f32" };
  return {
    nodes: nodes.map(([id, op, ins, outs, attrs]) => ({
      id,
      op,
      inputs: ins,
      outputs: outs,
      attrs: attrs ?? {},
    })),
    tensors,
    params: {},
  };
}

/** A fixture whose graph is exactly one node named "n". */
function one(
  op: string,
  inputs: Record<string, { shape: number[]; dtype?: DType }>,
  ins: string[],
  outs: string[],
  attrs?: Record<string, unknown>,
  note?: string
): OpFixture {
  return { op, graph: build(inputs, [["n", op, ins, outs, attrs]]), nodeId: "n", note };
}

const f = (shape: number[]) => ({ shape });
const i32 = (shape: number[]) => ({ shape, dtype: "i32" as DType });

export const OP_FIXTURES: OpFixture[] = [
  one(
    "einsum",
    { A: f([3, 4]), B: f([4, 5]) },
    ["A", "B"],
    ["C"],
    { equation: "mk,kn->mn" }
  ),
  one("matmul", { A: f([3, 4]), B: f([4, 5]) }, ["A", "B"], ["C"]),
  one("bmm", { A: f([2, 3, 4]), B: f([2, 4, 5]) }, ["A", "B"], ["C"]),
  one("linear", { X: f([3, 4]), W: f([5, 4]) }, ["X", "W"], ["Y"]),
  one(
    "elementwise",
    { A: f([2, 3, 4]), B: f([3, 1]) },
    ["A", "B"],
    ["C"],
    { fn: "add", nary: 2 },
    "unequal ranks and a broadcast axis, so the trailing alignment is exercised"
  ),
  one(
    "reduce",
    { X: f([2, 3, 4]) },
    ["X"],
    ["Y"],
    { fn: "sum", axes: [0, 2], keepdim: false },
    "two non-adjacent axes without keepdim: the hardest output-index remapping"
  ),
  one("softmax", { X: f([3, 5, 2]) }, ["X"], ["Y"], { axis: 1 }, "a middle axis, not the last"),
  one(
    "normalize",
    { X: f([2, 3, 4]), W: f([4]), B: f([4]) },
    ["X", "W", "B"],
    ["Y"],
    { kind: "layernorm", axes: [-1], hasWeight: true, hasBias: true },
    "both affine params present, so slots 1 and 2 are covered"
  ),
  one("transpose", { X: f([2, 3, 4]) }, ["X"], ["Y"], { perm: [2, 0, 1] }),
  one(
    "slice",
    { X: f([7, 9]) },
    ["X"],
    ["Y"],
    { starts: [1, 0], stops: [6, 9], steps: [2, 4] },
    "strided on both axes, below the enumeration cap so the result stays exact"
  ),
  one(
    "pad",
    { X: f([5, 4]) },
    ["X"],
    ["Y"],
    { pads: [[2, 1], [1, 3]], mode: "reflect" },
    "reflect is the only mode whose preimage is more than one interval per axis"
  ),
  one("concat", { A: f([2, 3]), B: f([4, 3]), C: f([1, 3]) }, ["A", "B", "C"], ["Y"], { axis: 0 }),
  one(
    "split",
    { X: f([6, 3]) },
    ["X"],
    ["Y0", "Y1", "Y2"],
    { axis: 0, sizes: [2, 3, 1] },
    "the only multi-output op, so output-slot handling is only covered here"
  ),
  one("expand", { X: f([3, 1]) }, ["X"], ["Y"], { shape: [2, 3, 5] }),
  one(
    "reshape",
    { X: f([4, 4]) },
    ["X"],
    ["Y"],
    { shape: [2, 8] },
    "the reshape trap: a contiguous tile of the output straddles rows of the input"
  ),
  one(
    "conv",
    { X: f([2, 2, 5, 5]), W: f([4, 1, 3, 3]) },
    ["X", "W"],
    ["Y"],
    { stride: [2, 1], pads: [[1, 1], [1, 1]], dilation: [1, 1], groups: 2 },
    "grouped, asymmetric stride, padded: the interesting corner of the halo rule"
  ),
  one(
    "pool",
    { X: f([1, 2, 8, 6]) },
    ["X"],
    ["Y"],
    { kind: "max", kernelShape: [2, 3], stride: [2, 2], pads: [[0, 0], [1, 1]] }
  ),
  one("cumsum", { X: f([4, 5]) }, ["X"], ["Y"], { axis: 1, reverse: true }),
  one(
    "gather",
    { D: f([6, 3]), I: i32([4]) },
    ["D", "I"],
    ["Y"],
    { axis: 0, indexValues: [5, 0, 2, 2] },
    "concrete indices, including a repeat: without them the op is inexact by design"
  ),
  one("identity", { X: f([3, 4]) }, ["X"], ["Y"]),
  one("cast", { X: f([3, 4]) }, ["X"], ["Y"], { dtype: "f16" }),
  one("contiguous", { X: f([3, 4]) }, ["X"], ["Y"]),
];
