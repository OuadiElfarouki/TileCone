import { describe, expect, it } from "vitest";
import { resolveGraph, Graph } from "../core/graph";
import { EXAMPLES } from "../examples/index";
import { parseDSL, toDSL } from "../parse/dsl";
import { graphToJSON, parseGraphJSON } from "../parse/json";
import { propagateBackward } from "../core/propagate";
import { box, fromBox } from "../core/region";
import { checkGraph } from "./harness";

function strip(g: Graph) {
  return JSON.parse(graphToJSON(g));
}

describe("DSL", () => {
  it("parses the spec's example", () => {
    const g = parseDSL(`params M=8 N=8 K=16

# a comment
input A [M, K] f16
input B [K, N] f16

C = einsum("mk,kn->mn", A, B)
D = softmax(C, axis=-1)
`);
    const rg = resolveGraph(g);
    expect(rg.tensors["C"].resolved).toEqual([8, 8]);
    expect(rg.tensors["D"].resolved).toEqual([8, 8]);
    expect(rg.tensors["A"].dtype).toBe("f16");
  });

  it("supports multi-output split and sugar names", () => {
    const g = parseDSL(`input X [6, 4] f32
A, B = split(X, axis=0, sizes=[2, 4])
C = add(A, A)
S = sum(C, axes=[1], keepdim=true)
`);
    const rg = resolveGraph(g);
    expect(rg.tensors["A"].resolved).toEqual([2, 4]);
    expect(rg.tensors["B"].resolved).toEqual([4, 4]);
    expect(rg.tensors["S"].resolved).toEqual([2, 1]);
  });

  it("accepts weight/param as input declarations and tags them", () => {
    const g = resolveGraph(parseDSL(`params M=4 K=6 N=8
input A [M, K] f16
weight B [K, N] f16
param Bias [N] f16
C = matmul(A, B)
D = add(C, Bias)
`));
    expect(g.tensors["A"].role).toBeUndefined(); // plain activation
    expect(g.tensors["B"].role).toBe("weight");
    expect(g.tensors["Bias"].role).toBe("weight");
    expect(g.tensors["D"].resolved).toEqual([4, 8]);
    // a weight is still just a graph input to the analysis
    expect(g.tensors["B"].producer).toBeUndefined();
  });

  it("reduce accepts the singular axis= spelling", () => {
    const g = resolveGraph(parseDSL(`input X [4, 6] f32
s = sum(X, axis=-1)
m = mean(X, axes=[0])
`));
    expect(g.tensors["s"].resolved).toEqual([4]);
    expect(g.tensors["m"].resolved).toEqual([6]);
  });

  it("names an unknown declaration keyword instead of demanding an =", () => {
    const bad = () => parseDSL("input X [4] f32\ntensor Y [4] f32\n");
    expect(bad).toThrow(/line 2/);
    expect(bad).toThrow(/unknown statement starting with "tensor"/);
    expect(bad).toThrow(/input\|weight\|param/);
  });

  it("says which axis attribute a reduce is missing", () => {
    expect(() => parseDSL("input X [4, 6] f32\ns = sum(X)\n")).toThrow(/sum\(\) needs an axis/);
  });

  it("reports line numbers on errors", () => {
    expect(() => parseDSL(`input X [4] f32\nY = bogus_op_name(X`)).toThrow(/line 2/);
  });

  it("round-trips JSON -> DSL -> JSON losslessly", () => {
    for (const ex of EXAMPLES) {
      const g1 = parseDSL(ex.dsl);
      resolveGraph(JSON.parse(graphToJSON(g1)) as Graph); // validates
      const g2 = parseDSL(toDSL(g1));
      expect(strip(g2), ex.name).toEqual(strip(g1));
    }
  });

  it("JSON loader validates schema", () => {
    expect(() => parseGraphJSON("{ not json")).toThrow(/invalid JSON/);
    expect(() => parseGraphJSON(`{"nodes": [{"id": 5}], "tensors": {}}`)).toThrow(/schema errors/);
    const g = parseGraphJSON(graphToJSON(parseDSL(EXAMPLES[0].dsl)));
    expect(resolveGraph(g).tensors["C"].resolved).toEqual([256, 256]);
  });
});

describe("built-in examples", () => {
  it("all examples parse, resolve, and propagate from their default selection", () => {
    for (const ex of EXAMPLES) {
      const g = resolveGraph(parseDSL(ex.dsl));
      if (ex.defaultSelection) {
        const { tensor, box: b } = ex.defaultSelection;
        const res = propagateBackward(g, { tensorId: tensor, region: fromBox(b.map(([lo, hi]) => ({ lo, hi }))) });
        expect(res.tensors.size, ex.name).toBeGreaterThan(1);
      }
    }
  });

  it("attention: one output token row pulls full K and V for every head", () => {
    const g = resolveGraph(parseDSL(EXAMPLES[2].dsl));
    const res = propagateBackward(g, {
      tensorId: "Out",
      region: fromBox(box([0, 1], [17, 18], [0, 128])),
    });
    // Kh/Vh: [B,H,S,D] fully needed
    expect(res.tensors.get("Kh")!.region.boxes).toEqual([box([0, 1], [0, 4], [0, 128], [0, 32])]);
    expect(res.tensors.get("Vh")!.region.boxes).toEqual([box([0, 1], [0, 4], [0, 128], [0, 32])]);
    // scores: only the query row 17, all keys
    expect(res.tensors.get("Scores")!.region.boxes).toEqual([
      box([0, 1], [0, 4], [17, 18], [0, 128]),
    ]);
    expect(res.tensors.get("Scores")!.region.exact).toBe(true);
  });

  it("examples validate against the oracle at miniature shapes", () => {
    // reshape trap + layernorm residual + cumsum are cheap enough to brute force as-is
    checkGraph(parseDSL(EXAMPLES[4].dsl), { perTensorElementCap: 16 });
    const miniLN = parseDSL(`params S=4 E=6
input X [S, E] f32
input W [E] f32
input Bb [E] f32
H = layernorm(X, W, Bb, axes=[-1])
Y = add(H, X)
`);
    checkGraph(miniLN, { perTensorElementCap: 12 });
    const miniCumsum = parseDSL(`params S=7
input X [S] f32
Y = cumsum(X, axis=0, reverse=false)
Z = cumsum(Y, axis=0, reverse=false)
`);
    checkGraph(miniCumsum);
    const miniAttn = parseDSL(`params B=1 H=2 S=4 D=3 E=6
input X  [B, S, E] f32
input Wq [E, E] f32
input Wk [E, E] f32
input Wv [E, E] f32
input Wo [E, E] f32
Qp = einsum("bse,ef->bsf", X, Wq)
Kp = einsum("bse,ef->bsf", X, Wk)
Vp = einsum("bse,ef->bsf", X, Wv)
Q4 = reshape(Qp, shape=[B, S, H, D])
K4 = reshape(Kp, shape=[B, S, H, D])
V4 = reshape(Vp, shape=[B, S, H, D])
Qh = transpose(Q4, perm=[0, 2, 1, 3])
Kh = transpose(K4, perm=[0, 2, 1, 3])
Vh = transpose(V4, perm=[0, 2, 1, 3])
Scores = einsum("bhqd,bhkd->bhqk", Qh, Kh)
P = softmax(Scores, axis=-1)
Z = einsum("bhqk,bhkd->bhqd", P, Vh)
Zt = transpose(Z, perm=[0, 2, 1, 3])
Zm = reshape(Zt, shape=[B, S, E])
Out = einsum("bse,ef->bsf", Zm, Wo)
`);
    checkGraph(miniAttn, { perTensorElementCap: 3, boxSelections: 1, forward: false });
    const miniConv = parseDSL(`params N=1 C=2 F1=2 F2=2 H=6 W=6
input X  [N, C, H, W] f32
input W1 [F1, C, 3, 3] f32
input W2 [F2, F1, 3, 3] f32
Y1 = conv(X, W1, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
Y2 = conv(Y1, W2, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
`);
    checkGraph(miniConv, { perTensorElementCap: 4, boxSelections: 1, forward: false });
    const miniGemm = parseDSL(`params M=4 N=4 K=5
input A [M, K] f32
input B [K, N] f32
C = matmul(A, B)
`);
    checkGraph(miniGemm);
  });
});
