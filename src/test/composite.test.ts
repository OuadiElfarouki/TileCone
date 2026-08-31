import { describe, expect, it } from "vitest";
import { expandNode } from "../core/expand";
import { resolveGraph } from "../core/graph";
import { computeMetrics } from "../core/metrics";
import { propagateBackward, propagateForward } from "../core/propagate";
import { box, fromBox, points, Region } from "../core/region";
import { flatIndex } from "./oracle";
import { G } from "./harness";

function flat(r: Region | undefined, shape: number[]): Set<number> {
  const out = new Set<number>();
  if (!r) return out;
  for (const p of points(r)) out.add(flatIndex(p, shape));
  return out;
}

/** The dependency result must be IDENTICAL for primitive vs expanded (IDEA.md §7),
 * on every tensor that exists in both graphs. */
function assertEquivalent(graph: ReturnType<typeof G>, nodeId: string, selTensor: string) {
  const g1 = resolveGraph(JSON.parse(JSON.stringify(graph)));
  const g2 = resolveGraph(expandNode(graph, nodeId));
  const shape = g1.tensors[selTensor].resolved!;
  const n = shape.reduce((a, b) => a * b, 1);
  for (let f = 0; f < Math.min(n, 24); f++) {
    const idx: number[] = [];
    let rest = f;
    for (let i = shape.length - 1; i >= 0; i--) {
      idx.unshift(rest % shape[i]);
      rest = Math.floor(rest / shape[i]);
    }
    const sel = { tensorId: selTensor, region: fromBox(idx.map((v) => ({ lo: v, hi: v + 1 }))) };
    const r1 = propagateBackward(g1, sel);
    const r2 = propagateBackward(g2, sel);
    expect(computeMetrics(g2, r2).flops, `metric FLOPs @${f}`).toBe(
      computeMetrics(g1, r1).flops
    );
    for (const tid of Object.keys(g1.tensors)) {
      const sh = g1.tensors[tid].resolved!;
      expect(flat(r2.tensors.get(tid)?.region, sh), `backward ${tid} @${f}`).toEqual(
        flat(r1.tensors.get(tid)?.region, sh)
      );
    }
  }
  // forward from each graph input
  for (const tid of Object.keys(g1.tensors)) {
    if (g1.tensors[tid].producer) continue;
    const sh = g1.tensors[tid].resolved!;
    const sel = { tensorId: tid, region: fromBox(sh.map(() => ({ lo: 0, hi: 1 }))) };
    const f1 = propagateForward(g1, sel);
    const f2 = propagateForward(g2, sel);
    for (const uid of Object.keys(g1.tensors)) {
      const ush = g1.tensors[uid].resolved!;
      expect(flat(f2.tensors.get(uid)?.region, ush), `forward ${uid} from ${tid}`).toEqual(
        flat(f1.tensors.get(uid)?.region, ush)
      );
    }
  }
}

describe("composite expansion equivalence", () => {
  it("softmax", () => {
    const graph = G({ X: [3, 5] }, [
      ["sm", "softmax", ["X"], ["Y"], { axis: -1 }],
      ["post", "elementwise", ["Y"], ["Z"], { fn: "relu", nary: 1 }],
    ]);
    assertEquivalent(graph, "sm", "Z");
  });

  it("layernorm with weight and bias", () => {
    const graph = G({ X: [2, 3, 4], W: [4], B: [4] }, [
      ["ln", "normalize", ["X", "W", "B"], ["Y"], { kind: "layernorm", axes: [-1], hasWeight: true, hasBias: true }],
    ]);
    assertEquivalent(graph, "ln", "Y");
  });

  it("rmsnorm with weight", () => {
    const graph = G({ X: [3, 4], W: [4] }, [
      ["rn", "normalize", ["X", "W"], ["Y"], { kind: "rmsnorm", axes: [-1], hasWeight: true, hasBias: false }],
    ]);
    assertEquivalent(graph, "rn", "Y");
  });

  it("expanded graph validates and keeps output shape", () => {
    const graph = G({ X: [4, 8] }, [["sm", "softmax", ["X"], ["Y"], { axis: 1 }]]);
    const g2 = resolveGraph(expandNode(graph, "sm"));
    expect(g2.tensors["Y"].resolved).toEqual([4, 8]);
    expect(g2.nodes.length).toBe(5);
  });

  it("expands a composite whose input still has an unresolved placeholder shape", () => {
    const graph = G({ A: [2, 3], B: [3, 4] }, [
      ["mm", "matmul", ["A", "B"], ["S"]],
      ["sm", "softmax", ["S"], ["P"], { axis: -1 }],
    ]);
    expect(graph.tensors.S.shape).toEqual([]);

    const expanded = resolveGraph(expandNode(graph, "sm"));
    expect(expanded.tensors.P.resolved).toEqual([2, 4]);
    expect(expanded.nodes.some((node) => node.op === "softmax")).toBe(false);
  });

  it("shares softmax reduction work across disjoint selections before and after expansion", () => {
    const graph = G({ X: [2, 128] }, [["sm", "softmax", ["X"], ["Y"], { axis: -1 }]]);
    const primitive = resolveGraph(graph);
    const expanded = resolveGraph(expandNode(graph, "sm"));
    const selection = {
      tensorId: "Y",
      region: {
        boxes: [box([0, 1], [0, 1]), box([0, 1], [64, 65])],
        exact: true,
        reasons: [],
      },
    };
    const primitiveMetrics = computeMetrics(primitive, propagateBackward(primitive, selection));
    const expandedMetrics = computeMetrics(expanded, propagateBackward(expanded, selection));

    expect(primitiveMetrics.flops).toBe(898);
    expect(expandedMetrics.flops).toBe(primitiveMetrics.flops);
  });

  it("shares normalization statistics across disjoint selections before and after expansion", () => {
    const graph = G({ X: [2, 128], W: [128], B: [128] }, [
      [
        "ln",
        "normalize",
        ["X", "W", "B"],
        ["Y"],
        { kind: "layernorm", axes: [-1], hasWeight: true, hasBias: true },
      ],
    ]);
    const primitive = resolveGraph(graph);
    const expanded = resolveGraph(expandNode(graph, "ln"));
    const selection = {
      tensorId: "Y",
      region: {
        boxes: [box([0, 1], [0, 1]), box([0, 1], [64, 65])],
        exact: true,
        reasons: [],
      },
    };
    const primitiveMetrics = computeMetrics(primitive, propagateBackward(primitive, selection));
    const expandedMetrics = computeMetrics(expanded, propagateBackward(expanded, selection));

    expect(primitiveMetrics.flops).toBe(522);
    expect(expandedMetrics.flops).toBe(primitiveMetrics.flops);
  });
});
