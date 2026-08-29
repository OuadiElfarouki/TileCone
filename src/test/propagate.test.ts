import { describe, expect, it } from "vitest";
import { Graph, resolveGraph } from "../core/graph";
import { propagateBackward, propagateForward } from "../core/propagate";
import { box, fromBox, sortRegion } from "../core/region";
import { G, checkGraph } from "./harness";

describe("GEMM end-to-end (the canonical example)", () => {
  const gemm = () => resolveGraph(G({ A: [512, 2048], B: [2048, 512] }, [["mm", "matmul", ["A", "B"], ["C"]]]));

  it("C[64:128, 0:64] needs A[64:128, :] and B[:, 0:64]", () => {
    const g = gemm();
    const res = propagateBackward(g, { tensorId: "C", region: fromBox(box([64, 128], [0, 64])) });
    expect(res.tensors.get("A")!.region.boxes).toEqual([box([64, 128], [0, 2048])]);
    expect(res.tensors.get("B")!.region.boxes).toEqual([box([0, 2048], [0, 64])]);
    expect(res.tensors.get("A")!.region.exact).toBe(true);
    expect(res.tensors.get("A")!.depth).toBe(1);
    expect(res.tensors.get("C")!.depth).toBe(0);
  });

  it("forward: A row influences C row", () => {
    const g = gemm();
    const res = propagateForward(g, { tensorId: "A", region: fromBox(box([7, 8], [0, 2048])) });
    expect(res.tensors.get("C")!.region.boxes).toEqual([box([7, 8], [0, 512])]);
  });
});

describe("diamond merging (tensor-level union before further propagation)", () => {
  it("residual diamond visits shared tensor once with merged region", () => {
    // X -> norm -> H;  Y = H + X  (X consumed twice)
    const g = resolveGraph(
      G({ X: [4, 6] }, [
        ["n0", "normalize", ["X"], ["H"], { kind: "layernorm", axes: [-1], hasWeight: false, hasBias: false }],
        ["n1", "elementwise", ["H", "X"], ["Y"], { fn: "add", nary: 2 }],
      ])
    );
    const res = propagateBackward(g, { tensorId: "Y", region: fromBox(box([1, 2], [2, 3])) });
    // through norm: full row 1; direct: element (1,2). Union = full row 1.
    expect(res.tensors.get("X")!.region.boxes).toEqual([box([1, 2], [0, 6])]);
    expect(res.tensors.get("X")!.region.exact).toBe(true);
  });

  it("diamond correctness against oracle", () => {
    checkGraph(
      G({ X: [3, 4] }, [
        ["n0", "normalize", ["X"], ["H"], { kind: "layernorm", axes: [-1], hasWeight: false, hasBias: false }],
        ["n1", "elementwise", ["H", "X"], ["Y"], { fn: "add", nary: 2 }],
      ])
    );
  });
});

describe("determinism", () => {
  it("same graph + selection -> byte-identical region output", () => {
    const build = () =>
      resolveGraph(
        G({ A: [8, 8], B: [8, 8] }, [
          ["mm", "matmul", ["A", "B"], ["C"]],
          ["sm", "softmax", ["C"], ["D"], { axis: -1 }],
          ["mm2", "matmul", ["D", "A"], ["E"]],
        ])
      );
    const sel = { tensorId: "E", region: fromBox(box([2, 5], [1, 3])) };
    const ser = (g: ReturnType<typeof build>) => {
      const res = propagateBackward(g, sel);
      return JSON.stringify(
        [...res.tensors.entries()]
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([id, tr]) => [id, sortRegion(tr.region), tr.depth])
      );
    };
    expect(ser(build())).toBe(ser(build()));
  });
});

describe("performance", () => {
  it("deep transformer-like chain propagates fast", () => {
    // 32 layers x (norm, qkv linear, attention einsums, softmax, out proj, residuals)
    const nodes: [string, string, string[], string[], Record<string, unknown>?][] = [];
    const inputs: Record<string, number[]> = { X0: [1024, 512] };
    let cur = "X0";
    for (let l = 0; l < 32; l++) {
      inputs[`Wq${l}`] = [512, 512];
      inputs[`Wk${l}`] = [512, 512];
      inputs[`Wv${l}`] = [512, 512];
      inputs[`Wo${l}`] = [512, 512];
      nodes.push([`ln${l}`, "normalize", [cur], [`h${l}`], { kind: "layernorm", axes: [-1], hasWeight: false, hasBias: false }]);
      nodes.push([`q${l}`, "einsum", [`h${l}`, `Wq${l}`], [`Q${l}`], { equation: "sd,de->se" }]);
      nodes.push([`k${l}`, "einsum", [`h${l}`, `Wk${l}`], [`K${l}`], { equation: "sd,de->se" }]);
      nodes.push([`v${l}`, "einsum", [`h${l}`, `Wv${l}`], [`V${l}`], { equation: "sd,de->se" }]);
      nodes.push([`s${l}`, "einsum", [`Q${l}`, `K${l}`], [`S${l}`], { equation: "qe,ke->qk" }]);
      nodes.push([`sm${l}`, "softmax", [`S${l}`], [`P${l}`], { axis: -1 }]);
      nodes.push([`av${l}`, "einsum", [`P${l}`, `V${l}`], [`Z${l}`], { equation: "qk,ke->qe" }]);
      nodes.push([`o${l}`, "einsum", [`Z${l}`, `Wo${l}`], [`O${l}`], { equation: "se,ef->sf" }]);
      nodes.push([`r${l}`, "elementwise", [`O${l}`, cur], [`X${l + 1}`], { fn: "add", nary: 2 }]);
      cur = `X${l + 1}`;
    }
    const g = resolveGraph(G(inputs, nodes));
    expect(g.nodes.length).toBeGreaterThan(280);
    const t0 = performance.now();
    const res = propagateBackward(g, { tensorId: cur, region: fromBox(box([100, 101], [0, 512])) });
    const ms = performance.now() - t0;
    expect(res.tensors.size).toBeGreaterThan(200);
    // spec target is 50ms for ~1500 nodes; allow slack for CI machines
    expect(ms).toBeLessThan(500);
  });
});

describe("built-in style examples at miniature shapes", () => {
  it("attention miniature", () => {
    const graph: Graph = G(
      { X: [4, 6], Wq: [6, 6], Wk: [6, 6], Wv: [6, 6] },
      [
        ["q", "einsum", ["X", "Wq"], ["Q"], { equation: "sd,de->se" }],
        ["k", "einsum", ["X", "Wk"], ["K"], { equation: "sd,de->se" }],
        ["v", "einsum", ["X", "Wv"], ["V"], { equation: "sd,de->se" }],
        ["s", "einsum", ["Q", "K"], ["S"], { equation: "qe,ke->qk" }],
        ["p", "softmax", ["S"], ["P"], { axis: -1 }],
        ["z", "einsum", ["P", "V"], ["Z"], { equation: "qk,ke->qe" }],
      ]
    );
    checkGraph(graph, { perTensorElementCap: 6, boxSelections: 1 });
  });

  it("conv stack (receptive field growth)", () => {
    checkGraph(
      G({ X: [1, 1, 12], W1: [2, 1, 3], W2: [2, 2, 3] }, [
        ["c1", "conv", ["X", "W1"], ["Y1"], { stride: [2], pads: [[1, 1]], dilation: [1], groups: 1 }],
        ["c2", "conv", ["Y1", "W2"], ["Y2"], { stride: [2], pads: [[1, 1]], dilation: [1], groups: 1 }],
      ]),
      { perTensorElementCap: 10, boxSelections: 1 }
    );
  });

  it("cumsum cone", () => {
    checkGraph(G({ X: [7] }, [["c", "cumsum", ["X"], ["Y"], { axis: 0, reverse: false }]]));
  });
});
