/**
 * Entanglement, against brute-force truth.
 *
 * The relation is "combined in the same term", which `oracleTerms` states
 * directly, so the check has the same shape as every other correctness test
 * here: exact means equal to the enumerated truth, inexact means a superset of
 * it. What makes this one worth writing carefully is that it is easy to compute
 * something plausible and looser - the composition of `forward` and `backward`
 * is exactly that, and it is what the fallback deliberately does.
 */

import { describe, expect, it } from "vitest";
import { resolveGraph } from "../core/graph";
import { entangledWith } from "../core/entangle";
import { getOp } from "../core/ops/index";
import { box, count, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";
import { G, randInt, rng } from "./harness";
import { regionToFlatSet, truthEntangled } from "./oracle";

/** Check every (fromSlot, otherSlot) pairing of a one-node graph. */
function checkEntanglement(graph: ReturnType<typeof G>, nodeId = "n", trials = 12) {
  const g = resolveGraph(graph);
  const node = g.topo.find((n) => n.id === nodeId)!;
  const shapes = g.shapesOf(node.inputs);
  const r = rng(99);
  let asserted = 0;

  for (let fromSlot = 0; fromSlot < node.inputs.length; fromSlot++) {
    for (let k = 0; k < trials; k++) {
      const sel = fromBox(
        shapes[fromSlot].map((e) => {
          const lo = randInt(r, 0, e);
          return { lo, hi: randInt(r, lo + 1, e + 1) };
        })
      );
      const results = entangledWith(g, node.inputs[fromSlot], sel).filter(
        (e) => e.nodeId === nodeId && e.fromSlot === fromSlot
      );
      for (let otherSlot = 0; otherSlot < node.inputs.length; otherSlot++) {
        if (otherSlot === fromSlot) continue;
        const entry = results.find((candidate) => candidate.slot === otherSlot);
        const truth = truthEntangled(g, nodeId, fromSlot, otherSlot, sel);
        const got = entry
          ? regionToFlatSet(entry.region, shapes[otherSlot])
          : new Set<number>();
        for (const f of truth)
          expect(got.has(f), `slot ${fromSlot}->${otherSlot}: missing ${f}`).toBe(true);
        if (!entry || entry.region.exact) expect(got).toEqual(truth);
        asserted++;
      }
    }
  }
  expect(asserted, "no entanglement was produced to assert on").toBeGreaterThan(0);
}

describe("einsum entanglement matches brute force", () => {
  it("matmul", () =>
    checkEntanglement(G({ A: [5, 4], B: [4, 6] }, [["n", "matmul", ["A", "B"], ["C"]]])));

  it("batched matmul", () =>
    checkEntanglement(G({ A: [2, 3, 4], B: [2, 4, 3] }, [["n", "bmm", ["A", "B"], ["C"]]])));

  it("attention scores: a shared batch and head, contracted depth", () =>
    checkEntanglement(
      G({ Q: [1, 2, 4, 3], K: [1, 2, 5, 3] }, [
        ["n", "einsum", ["Q", "K"], ["S"], { equation: "bhqd,bhkd->bhqk" }],
      ])
    ));

  it("outer product, which shares no label at all", () =>
    checkEntanglement(
      G({ a: [4], b: [5] }, [["n", "einsum", ["a", "b"], ["o"], { equation: "i,j->ij" }]])
    ));

  it("a diagonal operand", () =>
    checkEntanglement(
      G({ M: [4, 4], B: [4, 3] }, [
        ["n", "einsum", ["M", "B"], ["C"], { equation: "ii,ik->k" }],
      ])
    ));

  it("three operands", () =>
    checkEntanglement(
      G({ A: [2, 3], B: [3, 4], C: [4, 2] }, [
        ["n", "einsum", ["A", "B", "C"], ["D"], { equation: "ij,jk,kl->il" }],
      ])
    ));

  it("one tensor in two slots", () =>
    checkEntanglement(G({ A: [4, 4] }, [["n", "matmul", ["A", "A"], ["C"]]])));
});

describe("elementwise entanglement matches brute force", () => {
  it("equal shapes", () =>
    checkEntanglement(
      G({ A: [4, 5], B: [4, 5] }, [["n", "elementwise", ["A", "B"], ["C"], { fn: "add", nary: 2 }]])
    ));

  it("a broadcast axis", () =>
    checkEntanglement(
      G({ A: [4, 5], B: [4, 1] }, [["n", "elementwise", ["A", "B"], ["C"], { fn: "mul", nary: 2 }]])
    ));

  it("unequal ranks", () =>
    checkEntanglement(
      G({ A: [2, 3, 4], B: [3, 4] }, [
        ["n", "elementwise", ["A", "B"], ["C"], { fn: "add", nary: 2 }],
      ])
    ));

  it("three operands", () =>
    checkEntanglement(
      G({ A: [3, 4], B: [3, 4], C: [3, 4] }, [
        ["n", "elementwise", ["A", "B", "C"], ["D"], { fn: "add", nary: 3 }],
      ])
    ));
});

describe("entanglement is not the composition of the other two relations", () => {
  const program = () =>
    compileDSL("A = Tensor(8, 8)\nB = Tensor(8, 8)\nC = matmul(A, B)\n");

  it("is strictly tighter than forward-then-backward on a matmul", () => {
    const p = program();
    const sel = fromBox(box([0, 4], [0, 4]));

    const [entangled] = p.executor.entangled("A", sel);
    expect(entangled.tensorId).toBe("B");
    // A[m, 0:4] is multiplied only by B[0:4, n]: the first four rows, all cols.
    expect(entangled.region.boxes).toEqual([box([0, 4], [0, 8])]);
    expect(entangled.region.exact).toBe(true);

    // The composition reaches every element of B, because every C[m,n] in the
    // band really does read all of B. Correct, and a different question.
    const down = p.executor.downstream("A", sel).tensors.get("C")!.region;
    const composed = p.executor.upstream("C", down).tensors.get("B")!.region;
    expect(count(composed)).toBe(64);
    expect(count(entangled.region)).toBe(32);
  });

  it("agrees with the composition where the relation is the identity", () => {
    const p = compileDSL("A = Tensor(8, 8)\nB = Tensor(8, 8)\nC = add(A, B)\n");
    const sel = fromBox(box([0, 4], [0, 4]));
    const [entangled] = p.executor.entangled("A", sel);
    const down = p.executor.downstream("A", sel).tensors.get("C")!.region;
    const composed = p.executor.upstream("C", down).tensors.get("B")!.region;
    expect(count(entangled.region)).toBe(count(composed));
  });
});

describe("the shape of the answer", () => {
  it("reports each slot pairing separately when one tensor fills two slots", () => {
    const p = compileDSL("A = Tensor(6, 6)\nC = matmul(A, A)\n");
    const results = p.executor.entangled("A", fromBox(box([0, 2], [0, 6])));
    expect(results).toHaveLength(2);
    // As the left operand the band meets rows; as the right operand, columns.
    // Merging them would report their union as though it were either.
    const bySlot = Object.fromEntries(results.map((e) => [e.fromSlot, e.region.boxes]));
    expect(bySlot[0]).toEqual([box([0, 6], [0, 6])]);
    expect(bySlot[1]).toEqual([box([0, 6], [0, 2])]);
  });

  it("never entangles a slot with itself", () => {
    const p = compileDSL("A = Tensor(4, 4)\nC = matmul(A, A)\n");
    for (const e of p.executor.entangled("A", fromBox(box([0, 2], [0, 2]))))
      expect(e.slot).not.toBe(e.fromSlot);
  });

  it("returns nothing for a tensor no multi-input node reads", () => {
    const p = compileDSL("A = Tensor(4, 4)\nB = relu(A)\n");
    expect(p.executor.entangled("A", fromBox(box([0, 2], [0, 2])))).toEqual([]);
  });

  it("returns nothing for an empty selection", () => {
    const p = compileDSL("A = Tensor(4, 4)\nB = Tensor(4, 4)\nC = add(A, B)\n");
    expect(
      p.executor.entangled("A", { boxes: [], exact: true, reasons: [] })
    ).toEqual([]);
  });

  it("rejects a selection that does not fit the tensor", () => {
    const p = compileDSL("A = Tensor(4, 4)\nB = Tensor(4, 4)\nC = add(A, B)\n");
    expect(() => p.executor.entangled("A", fromBox(box([0, 99], [0, 2])))).toThrow();
  });
});

describe("conv entanglement matches brute force", () => {
  it("plain 1D", () =>
    checkEntanglement(
      G({ X: [1, 2, 7], W: [2, 2, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [1], pads: [[0, 0]], dilation: [1], groups: 1 }],
      ])
    ));

  it("padded, so some terms read a weight and no activation", () =>
    checkEntanglement(
      G({ X: [1, 1, 6], W: [2, 1, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [1], pads: [[2, 2]], dilation: [1], groups: 1 }],
      ])
    ));

  it("grouped: a channel only ever meets its own group", () =>
    checkEntanglement(
      G({ X: [1, 4, 6], W: [4, 2, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [1], pads: [[0, 0]], dilation: [1], groups: 2 }],
      ])
    ));

  it("strided and dilated", () =>
    checkEntanglement(
      G({ X: [1, 2, 9], W: [2, 2, 2] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [2], pads: [[1, 1]], dilation: [2], groups: 1 }],
      ])
    ));

  it("2D", () =>
    checkEntanglement(
      G({ X: [1, 2, 5, 4], W: [2, 2, 3, 2] }, [
        [
          "n",
          "conv",
          ["X", "W"],
          ["Y"],
          { stride: [1, 1], pads: [[1, 1], [0, 0]], dilation: [1, 1], groups: 1 },
        ],
      ]),
      "n",
      8
    ));

  /* Which taps can reach a position is the halo read backwards, and it is the
     one part of conv entanglement random boxes are poor at exercising: only a
     selection near an edge excludes a tap. Named cases rather than trusting the
     fuzzer to stumble on them. */
  it.each([
    ["the first position, reachable only by the first tap", 0, 1, [0, 1]],
    ["the last two, out of the first tap's reach", 5, 7, [1, 3]],
    ["the very last, reachable only by the last tap", 6, 7, [2, 3]],
    ["the whole axis, reachable by every tap", 0, 7, [0, 3]],
  ])("excludes taps that cannot reach: %s", (_label, lo, hi, expected) => {
    // X of 7, kernel 3, stride 1, no padding: 5 output positions, so tap k
    // covers input positions [k, k+5).
    const p = compileDSL(
      "X = Tensor(1,1,7)\nW = Parameter(1,1,3)\n" +
        "Y = conv(X,W,stride=[1],pads=[[0,0]],dilation=[1],groups=1)\n"
    );
    const [e] = p.executor.entangled("X", fromBox(box([0, 1], [0, 1], [lo, hi])));
    expect(e.region.exact).toBe(true);
    expect(e.region.boxes.map((b) => [b[2].lo, b[2].hi])).toEqual([expected]);
  });

  it("names only the group's weights, not the whole filter bank", () => {
    // The property the channel handling exists for: with two groups, channel 0
    // of the activation meets output channels 0..1 and nothing above them.
    const p = compileDSL(
      "X = Tensor(1, 4, 6)\nW = Parameter(4, 2, 3)\n" +
        "Y = conv(X, W, stride=[1], pads=[[0,0]], dilation=[1], groups=2)\n"
    );
    const [e] = p.executor.entangled("X", fromBox(box([0, 1], [0, 1], [0, 6])));
    expect(e.tensorId).toBe("W");
    expect(e.region.exact).toBe(true);
    for (const b of e.region.boxes) expect(b[0].hi).toBeLessThanOrEqual(2);
  });
});

describe("concat and gather", () => {
  it("concat combines nothing: its operands sit side by side", () => {
    const p = compileDSL("A = Tensor(2, 3)\nB = Tensor(4, 3)\nC = concat(A, B, axis=0)\n");
    // Not "unknown" and not a bound: genuinely empty, which says a kernel can
    // write the two pieces independently.
    expect(p.executor.entangled("A", fromBox(box([0, 2], [0, 3])))).toEqual([]);
  });

  it("gather pairs each index with the row it names", () => {
    const graph = G({ D: [6, 3], I: [4] }, [
      ["n", "gather", ["D", "I"], ["Y"], { axis: 0, indexValues: [5, 0, 2, 2] }],
    ]);
    graph.tensors.I.dtype = "i32";
    checkEntanglement(graph);
  });

  it("gather names the positions that select a block of rows", () => {
    const p = compileDSL(
      "D = Tensor(6, 3)\nI = Tensor(4, dtype=int32)\n" +
        "Y = gather(D, I, axis=0, indexValues=[5, 0, 2, 2])\n"
    );
    // Rows 2..3 are named by positions 2 and 3 only.
    const [e] = p.executor.entangled("D", fromBox(box([2, 4], [0, 3])));
    expect(e.tensorId).toBe("I");
    expect(e.region.boxes).toEqual([box([2, 4])]);
  });
});

describe("operations without the hook fall back, and say so", () => {
  it("normalize composes forward and backward, marked inexact", () => {
    const p = compileDSL("X = Tensor(2, 4)\nW = Parameter(4)\nY = layernorm(X, W)\n");
    const [entangled] = p.executor.entangled("X", fromBox(box([0, 1], [0, 2])));
    expect(entangled.tensorId).toBe("W");
    expect(entangled.region.exact).toBe(false);
    expect(entangled.region.reasons).toContain("composed from forward and backward");
  });

  it("the fallback is still a superset of the truth", () => {
    // conv has no oracleTerms, so truth comes from oracleDeps instead: every
    // weight element some reached output reads is at least in the bound.
    const g = resolveGraph(
      G({ X: [1, 2, 6], W: [3, 2, 3] }, [
        ["n", "conv", ["X", "W"], ["Y"], { stride: [1], pads: [[0, 0]], dilation: [1], groups: 1 }],
      ])
    );
    const spec = getOp("conv")!;
    const ctx = {
      inShapes: g.shapesOf(g.topo[0].inputs),
      outShapes: g.shapesOf(g.topo[0].outputs),
      attrs: g.topo[0].attrs,
    };
    const sel = fromBox(box([0, 1], [0, 2], [0, 6]));
    const [entangled] = entangledWith(g, "X", sel);
    const got = regionToFlatSet(entangled.region, ctx.inShapes[1]);
    const outShape = ctx.outShapes[0];
    const n = outShape.reduce((a, b) => a * b, 1);
    for (let f = 0; f < n; f++) {
      const idx = [
        Math.floor(f / (outShape[1] * outShape[2])),
        Math.floor(f / outShape[2]) % outShape[1],
        f % outShape[2],
      ];
      for (const w of spec.oracleDeps(0, idx, ctx)[1]) {
        const flat = (w[0] * ctx.inShapes[1][1] + w[1]) * ctx.inShapes[1][2] + w[2];
        expect(got.has(flat), `weight ${w} missing from the bound`).toBe(true);
      }
    }
  });
});
