import { describe, expect, it } from "vitest";
import { resolveGraph } from "../core/graph";
import { propagateBackward } from "../core/propagate";
import { checkGraph, G, rng, randInt } from "./harness";
import { einsumBackward } from "../core/ops/einsum";
import { fromBox, box, count, full } from "../core/region";
import { getOp } from "../core/ops/index";
import { compileDSL } from "../parse/compiler";

describe("shape inference (known-good table)", () => {
  const cases: [string, ReturnType<typeof G>, Record<string, number[]>][] = [
    [
      "matmul",
      G({ A: [3, 4], B: [4, 5] }, [["n", "matmul", ["A", "B"], ["C"]]]),
      { C: [3, 5] },
    ],
    [
      "einsum bhqd,bhkd->bhqk",
      G({ Q: [2, 2, 3, 4], K: [2, 2, 5, 4] }, [
        ["n", "einsum", ["Q", "K"], ["S"], { equation: "bhqd,bhkd->bhqk" }],
      ]),
      { S: [2, 2, 3, 5] },
    ],
    [
      "reduce keepdim",
      G({ X: [2, 3, 4] }, [["n", "reduce", ["X"], ["Y"], { fn: "sum", axes: [1], keepdim: true }]]),
      { Y: [2, 1, 4] },
    ],
    [
      "reduce nokeep",
      G({ X: [2, 3, 4] }, [["n", "reduce", ["X"], ["Y"], { fn: "max", axes: [-1, 0], keepdim: false }]]),
      { Y: [3] },
    ],
    [
      "conv2d",
      G({ X: [1, 4, 8, 8], W: [6, 2, 3, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [2, 2], pads: [[1, 1], [1, 1]], dilation: [1, 1], groups: 2 }],
      ]),
      { Y: [1, 6, 4, 4] },
    ],
    [
      "pad reflect",
      G({ X: [4, 5] }, [["n", "pad", ["X"], ["Y"], { pads: [[1, 2], [0, 3]], mode: "reflect" }]]),
      { Y: [7, 8] },
    ],
    [
      "slice strided",
      G({ X: [10] }, [["n", "slice", ["X"], ["Y"], { starts: [1], stops: [10], steps: [3] }]]),
      { Y: [3] },
    ],
    [
      "reshape",
      G({ X: [2, 3, 4] }, [["n", "reshape", ["X"], ["Y"], { shape: [6, 4] }]]),
      { Y: [6, 4] },
    ],
  ];
  for (const [name, graph, expected] of cases) {
    it(name, () => {
      const g = resolveGraph(graph);
      for (const [tid, sh] of Object.entries(expected)) expect(g.tensors[tid].resolved).toEqual(sh);
    });
  }
});

describe("oracle corpus: single ops", () => {
  it("einsum matmul", () =>
    checkGraph(G({ A: [3, 4], B: [4, 5] }, [["n", "matmul", ["A", "B"], ["C"]]])));

  it("einsum bmm", () =>
    checkGraph(G({ A: [2, 3, 4], B: [2, 4, 3] }, [["n", "bmm", ["A", "B"], ["C"]]])));

  it("einsum outer + trace + transpose", () => {
    checkGraph(G({ a: [4], b: [5] }, [["n", "einsum", ["a", "b"], ["o"], { equation: "i,j->ij" }]]));
    checkGraph(G({ M: [4, 4] }, [["n", "einsum", ["M"], ["t"], { equation: "ii->" }]]));
    checkGraph(G({ M: [3, 5] }, [["n", "einsum", ["M"], ["T"], { equation: "ij->ji" }]]));
  });

  it("einsum diagonal (repeated labels)", () => {
    checkGraph(G({ M: [5, 5] }, [["n", "einsum", ["M"], ["d"], { equation: "ii->i" }]]));
    checkGraph(
      G({ M: [4, 4, 3] }, [["n", "einsum", ["M"], ["d"], { equation: "iij->ij" }]])
    );
    checkGraph(
      G({ A: [3, 3], B: [3, 4] }, [["n", "einsum", ["A", "B"], ["C"], { equation: "ii,ik->k" }]])
    );
  });

  it("elementwise with broadcasting", () => {
    checkGraph(G({ A: [3, 4], B: [3, 4] }, [["n", "elementwise", ["A", "B"], ["C"], { fn: "add", nary: 2 }]]));
    checkGraph(G({ A: [3, 4], B: [1, 4] }, [["n", "elementwise", ["A", "B"], ["C"], { fn: "mul", nary: 2 }]]));
    checkGraph(G({ A: [2, 3, 4], B: [4] }, [["n", "elementwise", ["A", "B"], ["C"], { fn: "add", nary: 2 }]]));
    checkGraph(G({ A: [2, 1, 4], B: [1, 3, 1] }, [["n", "elementwise", ["A", "B"], ["C"], { fn: "add", nary: 2 }]]));
    // A rank-2 operand trailing-aligned against rank 4: how an attention mask
    // reaches [B, H, Q, K].
    checkGraph(G({ A: [2, 2, 3, 4], B: [3, 4] }, [["n", "elementwise", ["A", "B"], ["C"], { fn: "add", nary: 2 }]]));
    // Rank 0 on either side, and on both. The offset arithmetic in the
    // broadcast box mappings is at its most degenerate here.
    checkGraph(G({ A: [2, 3], s: [] }, [["n", "elementwise", ["A", "s"], ["C"], { fn: "mul", nary: 2 }]]));
    checkGraph(G({ s: [], A: [2, 3, 2] }, [["n", "elementwise", ["s", "A"], ["C"], { fn: "add", nary: 2 }]]));
    checkGraph(G({ s: [], t: [] }, [["n", "elementwise", ["s", "t"], ["C"], { fn: "add", nary: 2 }]]));
  });

  it("reduce variants", () => {
    for (const keepdim of [true, false])
      for (const axes of [[0], [1], [0, 2], [-1]])
        checkGraph(
          G({ X: [3, 4, 2] }, [["n", "reduce", ["X"], ["Y"], { fn: "sum", axes, keepdim }]])
        );
  });

  it("softmax", () => {
    checkGraph(G({ X: [3, 5] }, [["n", "softmax", ["X"], ["Y"], { axis: -1 }]]));
    checkGraph(G({ X: [3, 5, 2] }, [["n", "softmax", ["X"], ["Y"], { axis: 1 }]]));
  });

  it("normalize", () => {
    checkGraph(
      G({ X: [3, 4], W: [4], B: [4] }, [
        ["n", "normalize", ["X", "W", "B"], ["Y"], { kind: "layernorm", axes: [-1], hasWeight: true, hasBias: true }],
      ])
    );
    checkGraph(
      G({ X: [2, 3, 4], W: [4] }, [
        ["n", "normalize", ["X", "W"], ["Y"], { kind: "rmsnorm", axes: [2], hasWeight: true, hasBias: false }],
      ])
    );
    checkGraph(
      G({ X: [2, 3, 4] }, [
        ["n", "normalize", ["X"], ["Y"], { kind: "layernorm", axes: [1, 2], hasWeight: false, hasBias: false }],
      ])
    );
  });

  it("transpose", () => {
    checkGraph(G({ X: [3, 4] }, [["n", "transpose", ["X"], ["Y"], { perm: [1, 0] }]]));
    checkGraph(G({ X: [2, 3, 4] }, [["n", "transpose", ["X"], ["Y"], { perm: [2, 0, 1] }]]));
  });

  it("slice incl. strided", () => {
    checkGraph(G({ X: [8] }, [["n", "slice", ["X"], ["Y"], { starts: [2], stops: [8], steps: [1] }]]));
    checkGraph(G({ X: [10] }, [["n", "slice", ["X"], ["Y"], { starts: [1], stops: [10], steps: [3] }]]));
    checkGraph(
      G({ X: [7, 9] }, [["n", "slice", ["X"], ["Y"], { starts: [1, 0], stops: [6, 9], steps: [2, 4] }]])
    );
  });

  it("pad all modes", () => {
    for (const mode of ["constant", "reflect", "replicate"] as const)
      checkGraph(
        G({ X: [5, 4] }, [["n", "pad", ["X"], ["Y"], { pads: [[2, 1], [1, 3]], mode }]])
      );
  });

  it("concat / split", () => {
    checkGraph(
      G({ A: [2, 3], B: [4, 3], C: [1, 3] }, [["n", "concat", ["A", "B", "C"], ["Y"], { axis: 0 }]])
    );
    checkGraph(
      G({ X: [6, 3] }, [["n", "split", ["X"], ["Y0", "Y1", "Y2"], { axis: 0, sizes: [2, 3, 1] }]])
    );
  });

  it("expand", () => {
    checkGraph(G({ X: [1, 4] }, [["n", "expand", ["X"], ["Y"], { shape: [3, 4] }]]));
    checkGraph(G({ X: [3, 1] }, [["n", "expand", ["X"], ["Y"], { shape: [2, 3, 5] }]]));
  });

  it("conv variants", () => {
    // stride <= kernel (contiguous), stride > kernel (gaps), dilation > 1
    checkGraph(
      G({ X: [1, 2, 8], W: [3, 2, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [2], pads: [[1, 1]], dilation: [1], groups: 1 }],
      ])
    );
    checkGraph(
      G({ X: [1, 1, 9], W: [2, 1, 2] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [3], pads: [[0, 0]], dilation: [1], groups: 1 }],
      ])
    );
    checkGraph(
      G({ X: [1, 2, 9], W: [2, 1, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [1], pads: [[2, 2]], dilation: [2], groups: 2 }],
      ])
    );
    checkGraph(
      G({ X: [2, 2, 5, 5], W: [4, 1, 3, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [2, 1], pads: [[1, 1], [1, 1]], dilation: [1, 1], groups: 2 }],
      ]),
      { perTensorElementCap: 12 }
    );
  });

  it("pool", () => {
    checkGraph(
      G({ X: [1, 2, 8, 6] }, [
        ["n", "pool", ["X"], ["Y"], { kind: "max", kernelShape: [2, 3], stride: [2, 2], pads: [[0, 0], [1, 1]] }],
      ]),
      { perTensorElementCap: 12 }
    );
  });

  it("cumsum both directions", () => {
    checkGraph(G({ X: [6] }, [["n", "cumsum", ["X"], ["Y"], { axis: 0, reverse: false }]]));
    checkGraph(G({ X: [4, 5] }, [["n", "cumsum", ["X"], ["Y"], { axis: 1, reverse: true }]]));
  });

  it("gather with concrete indices", () => {
    const axis0 = G({ D: [6, 3], I: [4] }, [
      ["n", "gather", ["D", "I"], ["Y"], { axis: 0, indexValues: [5, 0, 2, 2] }],
    ]);
    axis0.tensors.I.dtype = "i32";
    checkGraph(axis0);

    const axis1 = G({ D: [2, 5], I: [3] }, [
      ["n", "gather", ["D", "I"], ["Y"], { axis: 1, indexValues: [4, 4, 1] }],
    ]);
    axis1.tensors.I.dtype = "i32";
    checkGraph(axis1);
  });

  it("identity family", () => {
    checkGraph(G({ X: [3, 4] }, [["n", "identity", ["X"], ["Y"]]]));
    checkGraph(G({ X: [3, 4] }, [["n", "cast", ["X"], ["Y"], { dtype: "f16" }]]));
  });
});

describe("inexact fallbacks are marked and are supersets", () => {
  it("large diagonal einsum coarsens the staircase, marked and still a superset", () => {
    const ctx = {
      inShapes: [[1000, 1000]],
      outShapes: [[1000]],
      attrs: { equation: "ii->i" },
    };
    const [r] = einsumBackward("ii->i", box([0, 1000]), ctx);
    expect(r.exact).toBe(false);
    expect(r.reasons).toContain("diagonal einsum");
    // Superset: every diagonal cell is still covered by some box.
    const covers = (i: number) =>
      r.boxes.some((b) => i >= b[0].lo && i < b[0].hi && i >= b[1].lo && i < b[1].hi);
    for (let i = 0; i < 1000; i++) expect(covers(i), `diagonal cell ${i}`).toBe(true);
    // But blocks along the diagonal, not the whole square it used to report.
    expect(count(r)).toBeGreaterThanOrEqual(1000);
    expect(count(r)).toBeLessThan(1000 * 1000 / 100);
  });

  it("data-dependent gather is full + inexact", () => {
    const graph = G(
      { D: [6, 3], I: [4] },
      [["n", "gather", ["D", "I"], ["Y"], { axis: 0 }]]
    );
    graph.tensors.I.dtype = "i32";
    const g = resolveGraph(graph);
    const res = propagateBackward(g, { tensorId: "Y", region: fromBox(box([0, 1], [0, 1])) });
    const d = res.tensors.get("D")!;
    expect(d.region.exact).toBe(false);
    expect(d.region.reasons).toContain("data-dependent index");
    expect(count(d.region)).toBe(6); // full axis 0, single column
  });
});

describe("oracle corpus: random composed graphs with diamonds", () => {
  it("random 5-15 node graphs", () => {
    const r = rng(2024);
    for (let trial = 0; trial < 24; trial++) {
      const graph = randomGraph(r, randInt(r, 5, 16));
      checkGraph(graph, { perTensorElementCap: 8, boxSelections: 1, seed: trial });
    }
  });

  /* The same generator with the fallback thresholds lowered, so composed
     graphs take the conservative branches as well as the exact ones. A cone
     that passes through two widened regions is where an over-approximation
     could plausibly be mishandled, and nothing else composes them. */
  it("random graphs under tight fallback thresholds", () => {
    const r = rng(31337);
    for (let trial = 0; trial < 24; trial++) {
      const graph = randomGraph(r, randInt(r, 5, 16));
      checkGraph(graph, {
        perTensorElementCap: 6,
        boxSelections: 1,
        seed: trial,
        limits: { stridedEnum: 2, diagEnum: 2, reshapeRuns: 1, maxBoxes: 2 },
      });
    }
  });
});

function randomGraph(r: () => number, nNodes: number) {
  type T = { id: string; shape: number[] };
  const inputs: Record<string, number[]> = {};
  const pool: T[] = [];
  let tid = 0;
  const newInput = (shape: number[]) => {
    const id = `in${tid++}`;
    inputs[id] = shape;
    const t = { id, shape };
    pool.push(t);
    return t;
  };
  newInput([randInt(r, 2, 5), randInt(r, 2, 5)]);
  newInput([randInt(r, 2, 5), randInt(r, 2, 5), randInt(r, 2, 4)]);
  const nodes: [string, string, string[], string[], Record<string, unknown>?][] = [];
  let nid = 0;
  const emit = (op: string, ins: T[], outShape: number[], attrs?: Record<string, unknown>) => {
    const id = `t${tid++}`;
    nodes.push([`n${nid++}`, op, ins.map((x) => x.id), [id], attrs]);
    const t = { id, shape: outShape };
    pool.push(t);
    return t;
  };
  const pick = () => pool[randInt(r, 0, pool.length)];
  for (let k = 0; k < nNodes; k++) {
    const choice = randInt(r, 0, 14);
    const t = pick();
    const sh = t.shape;
    if (choice === 0 && sh.length >= 2) {
      const perm = sh.map((_, i) => i);
      for (let i = perm.length - 1; i > 0; i--) {
        const j = randInt(r, 0, i + 1);
        [perm[i], perm[j]] = [perm[j], perm[i]];
      }
      emit("transpose", [t], perm.map((p) => sh[p]), { perm });
    } else if (choice === 1) {
      // diamond: t + t
      emit("elementwise", [t, t], sh.slice(), { fn: "add", nary: 2 });
    } else if (choice === 2 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      emit("softmax", [t], sh.slice(), { axis });
    } else if (choice === 3 && sh.length >= 2) {
      const axis = randInt(r, 0, sh.length);
      const out = sh.filter((_, i) => i !== axis);
      emit("reduce", [t], out, { fn: "sum", axes: [axis], keepdim: false });
    } else if (choice === 4) {
      // reshape to random re-factorization
      const vol = sh.reduce((a, b) => a * b, 1);
      const dims: number[] = [];
      let rest = vol;
      while (rest > 1 && dims.length < 3) {
        const divisors: number[] = [];
        for (let d = 2; d <= rest; d++) if (rest % d === 0) divisors.push(d);
        const d = divisors[randInt(r, 0, divisors.length)];
        dims.push(d);
        rest /= d;
      }
      if (rest > 1) dims.push(rest);
      if (dims.length === 0) dims.push(1);
      emit("reshape", [t], dims, { shape: dims });
    } else if (choice === 5 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      emit("cumsum", [t], sh.slice(), { axis, reverse: r() < 0.5 });
    } else if (choice === 6 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      const out = sh.slice();
      out[axis] *= 2;
      emit("concat", [t, t], out, { axis });
    } else if (choice === 7 && sh.length === 2) {
      const other = newInput([sh[1], randInt(r, 2, 4)]);
      emit("matmul", [t, other], [sh[0], other.shape[1]]);
    } else if (choice === 8 && sh.length >= 1) {
      // slice, sometimes strided
      const step = randInt(r, 1, 3);
      const starts = sh.map(() => 0);
      const stops = sh.slice();
      const steps = sh.map(() => 1);
      const axis = randInt(r, 0, sh.length);
      steps[axis] = step;
      starts[axis] = randInt(r, 0, Math.max(1, sh[axis] - 1));
      const out = sh.map((e, i) =>
        i === axis ? Math.max(1, Math.ceil((e - starts[i]) / steps[i])) : e
      );
      emit("slice", [t], out, { starts, stops, steps });
    } else if (choice === 9 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      const pads: [number, number][] = sh.map(() => [0, 0]);
      const mode = ["constant", "replicate", "reflect"][randInt(r, 0, 3)];
      // reflect cannot pad wider than extent-1
      const cap = mode === "reflect" ? Math.max(0, sh[axis] - 1) : 2;
      pads[axis] = [randInt(r, 0, Math.min(2, cap) + 1), randInt(r, 0, Math.min(2, cap) + 1)];
      emit("pad", [t], sh.map((e, i) => e + pads[i][0] + pads[i][1]), { pads, mode });
    } else if (choice === 10 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      if (sh[axis] < 2) {
        emit("elementwise", [t], sh.slice(), { fn: "relu", nary: 1 });
      } else {
        const cut = randInt(r, 1, sh[axis]);
        const sizes = [cut, sh[axis] - cut];
        const outs = sizes.map((sz) => sh.map((e, i) => (i === axis ? sz : e)));
        // split is the only multi-output op; take the first piece onward
        const id = `t${tid++}`;
        const id2 = `t${tid++}`;
        nodes.push([`n${nid++}`, "split", [t.id], [id, id2], { axis, sizes }]);
        pool.push({ id, shape: outs[0] });
        pool.push({ id: id2, shape: outs[1] });
      }
    } else if (choice === 11 && sh.length >= 1) {
      // expand a fresh degenerate axis up to t's shape
      const axis = randInt(r, 0, sh.length);
      const src = newInput(sh.map((e, i) => (i === axis ? 1 : e)));
      emit("expand", [src], sh.slice(), { shape: sh.slice() });
    } else if (choice === 12 && sh.length >= 1) {
      const perm = sh.map((_, i) => i);
      emit("transpose", [t], perm.map((p) => sh[p]), { perm });
    } else if (choice === 13 && sh.length >= 1) {
      const axes = [randInt(r, 0, sh.length)];
      const wShape = axes.map((a) => sh[a]);
      // normalize's affine params must match the normalized axes, and
      // expansion requires them to be trailing
      if (axes[0] === sh.length - 1) {
        const w = newInput(wShape);
        emit("normalize", [t, w], sh.slice(), {
          kind: r() < 0.5 ? "layernorm" : "rmsnorm",
          axes,
          hasWeight: true,
          hasBias: false,
        });
      } else emit("elementwise", [t], sh.slice(), { fn: "relu", nary: 1 });
    } else {
      emit("elementwise", [t], sh.slice(), { fn: "relu", nary: 1 });
    }
  }
  const graph = G(inputs, nodes);
  // declared shapes for intermediates are unknown; leave empty (inferred)
  return graph;
}

/**
 * FLOP counts for the equation as written. The cases that matter are the ones
 * where a term drops out: an outer product only multiplies, a reduction only
 * adds, a transpose does neither.
 */
describe("einsum FLOPs count the fused contraction", () => {
  const perElement = (equation: string, inShapes: number[][], outShape: number[]) => {
    const spec = getOp("einsum")!;
    return spec.flopsPerElement!(0, { inShapes, outShapes: [outShape], attrs: { equation } });
  };

  it("charges the conventional 2K for a matmul", () => {
    expect(perElement("mk,kn->mn", [[3, 8], [8, 5]], [3, 5])).toBe(2 * 8);
  });

  it("charges one multiply for an outer product, which accumulates nothing", () => {
    // Two operands but no contracted axis: 2*(n-1)*positions used to say 2.
    expect(perElement("i,j->ij", [[4], [5]], [4, 5])).toBe(1);
  });

  it("charges three per position for a fused three-operand contraction", () => {
    // Two multiplies and one add, not the four the old form charged.
    expect(perElement("ij,jk,kl->il", [[2, 3], [3, 4], [4, 5]], [2, 5])).toBe(3 * (3 * 4));
  });

  it("charges only adds for a reduction", () => {
    expect(perElement("ij->i", [[4, 7]], [4])).toBe(7);
  });

  it("charges nothing for a transpose", () => {
    expect(perElement("ij->ji", [[4, 5]], [5, 4])).toBe(0);
  });

  it("prices a written pairwise decomposition below the fused equation", () => {
    // The point of leaving the fused count alone: the cheaper program is the
    // one the author writes, and then the graph says so.
    const fused = compileDSL(`A = Tensor(64, 64)
B = Tensor(64, 64)
C = Tensor(64, 64)
D = einsum("ij,jk,kl->il", A, B, C)
`);
    const pairwise = compileDSL(`A = Tensor(64, 64)
B = Tensor(64, 64)
C = Tensor(64, 64)
T = einsum("ij,jk->ik", A, B)
D = einsum("ik,kl->il", T, C)
`);
    const flopsOf = (p: ReturnType<typeof compileDSL>) =>
      p.executor.metrics("D", full(p.resolved.tensors.D.resolved!)).flops;
    expect(flopsOf(pairwise)).toBeLessThan(flopsOf(fused));
  });
});
