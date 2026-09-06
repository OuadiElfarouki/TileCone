import { describe, expect, it } from "vitest";
import { resolveGraph, Graph } from "../core/graph";
import { EXAMPLES } from "../examples/index";
import { DSLError, parseDSL, toDSL } from "../parse/dsl";
import { graphToJSON, parseGraphJSON } from "../parse/json";
import { propagateBackward } from "../core/propagate";
import { box, fromBox } from "../core/region";
import { checkGraph } from "./harness";

const exampleNamed = (name: string) => {
  const example = EXAMPLES.find((candidate) => candidate.name === name);
  if (!example) throw new Error(`missing built-in example "${name}"`);
  return example;
};

function strip(g: Graph) {
  return JSON.parse(graphToJSON(g));
}

describe("DSL", () => {
  it("parses the spec's example", () => {
    const g = parseDSL(`M = 8
N = 8
K = 16

# a comment
A = Tensor(M, K, dtype=fp16)
B = Tensor(K, N, dtype=fp16)

C = einsum("mk,kn->mn", A, B)
D = softmax(C, axis=-1)
`);
    const rg = resolveGraph(g);
    expect(rg.tensors["C"].resolved).toEqual([8, 8]);
    expect(rg.tensors["D"].resolved).toEqual([8, 8]);
    expect(rg.tensors["A"].dtype).toBe("f16");
  });

  it("supports multi-output split and sugar names", () => {
    const g = parseDSL(`X = Tensor(6, 4, dtype=fp16)
A, B = split(X, axis=0, sizes=[2, 4])
C = add(A, A)
S = sum(C, axes=[1], keepdim=true)
`);
    const rg = resolveGraph(g);
    expect(rg.tensors["A"].resolved).toEqual([2, 4]);
    expect(rg.tensors["B"].resolved).toEqual([4, 4]);
    expect(rg.tensors["S"].resolved).toEqual([2, 1]);
    expect(rg.tensors["A"].dtype).toBe("f16");
    expect(rg.tensors["B"].dtype).toBe("f16");
    expect(rg.tensors["S"].dtype).toBe("f16");
  });

  it("uses Parameter for learned graph inputs", () => {
    const g = resolveGraph(parseDSL(`M = 4
K = 6
N = 8
A = Tensor(M, K, dtype=fp16)
B = Parameter(K, N, dtype=fp16)
Bias = Parameter(N, dtype=fp16)
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
    const g = resolveGraph(parseDSL(`X = Tensor(4, 6, dtype=fp32)
s = sum(X, axis=-1)
m = mean(X, axes=[0])
`));
    expect(g.tensors["s"].resolved).toEqual([4]);
    expect(g.tensors["m"].resolved).toEqual([6]);
  });

  it("rejects the removed declaration grammar", () => {
    for (const source of ["params M=4\n", "input X [4] f32\n", "weight W [4] f16\n"]) {
      const bad = () => parseDSL(source);
      expect(bad).toThrow(DSLError);
      expect(bad).toThrow(/expected an assignment/);
    }
  });

  it("says which axis attribute a reduce is missing", () => {
    expect(() => parseDSL("X = Tensor(4, 6, dtype=fp32)\ns = sum(X)\n")).toThrow(/sum\(\) needs an axis/);
  });

  it("reports line numbers on errors", () => {
    expect(() => parseDSL(`X = Tensor(4, dtype=fp32)\nY = bogus_op_name(X`)).toThrow(/line 2/);
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
    const g = parseGraphJSON(graphToJSON(parseDSL(exampleNamed("Plain GEMM").dsl)));
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
    const g = resolveGraph(parseDSL(exampleNamed("Multi-head attention").dsl));
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
    checkGraph(parseDSL(exampleNamed("Reshape trap").dsl), { perTensorElementCap: 16 });
    const miniLN = parseDSL(`S = 4
E = 6
X = Tensor(S, E, dtype=fp32)
W = Tensor(E, dtype=fp32)
Bb = Tensor(E, dtype=fp32)
H = layernorm(X, W, Bb, axes=[-1])
Y = add(H, X)
`);
    checkGraph(miniLN, { perTensorElementCap: 12 });
    const miniCumsum = parseDSL(`S = 7
X = Tensor(S, dtype=fp32)
Y = cumsum(X, axis=0, reverse=false)
Z = cumsum(Y, axis=0, reverse=false)
`);
    checkGraph(miniCumsum);
    const miniAttn = parseDSL(`B = 1
H = 2
S = 4
D = 3
E = 6
X = Tensor(B, S, E, dtype=fp32)
Wq = Tensor(E, E, dtype=fp32)
Wk = Tensor(E, E, dtype=fp32)
Wv = Tensor(E, E, dtype=fp32)
Wo = Tensor(E, E, dtype=fp32)
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
    const miniConv = parseDSL(`N = 1
C = 2
F1 = 2
F2 = 2
H = 6
W = 6
X = Tensor(N, C, H, W, dtype=fp32)
W1 = Tensor(F1, C, 3, 3, dtype=fp32)
W2 = Tensor(F2, F1, 3, 3, dtype=fp32)
Y1 = conv(X, W1, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
Y2 = conv(Y1, W2, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
`);
    checkGraph(miniConv, { perTensorElementCap: 4, boxSelections: 1, forward: false });
    const miniGemm = parseDSL(`M = 4
N = 4
K = 5
A = Tensor(M, K, dtype=fp32)
B = Tensor(K, N, dtype=fp32)
C = matmul(A, B)
`);
    checkGraph(miniGemm);
  });
});

/* The parser used to accept several things quietly. Each of these is a case
   where it changed the program instead of refusing it. */
describe("what the parser refuses", () => {
  const parseFails = (src: string) => {
    try {
      parseDSL(src);
    } catch (error) {
      return error as DSLError;
    }
    throw new Error("expected a DSLError, but the source parsed");
  };

  it("rejects trailing input after a constructor", () => {
    const error = parseFails("X = Tensor(4, 8, dtype=fp32) THIS_IS_GARBAGE\n");
    expect(error).toBeInstanceOf(DSLError);
    expect(error.detail).toContain("trailing input");
    expect(error.line).toBe(1);
  });

  it("rejects an internal dtype spelling from the removed grammar", () => {
    const error = parseFails("X = Tensor(4, 8, dtype=f32)\n");
    expect(error.detail).toContain("dtype must be one of");
  });

  /* The DSL and the IR spell dtypes differently, so the mapping is a seam with
     two sides and no test can check one alone: every external spelling must
     land on its internal name, and `toDSL` must put the external one back or a
     saved graph reopens as a source the parser rejects. */
  it.each([
    ["fp32", "f32"],
    ["fp16", "f16"],
    ["bf16", "bf16"],
    ["fp8", "f8"],
    ["int32", "i32"],
    ["int8", "i8"],
    ["bool", "bool"],
  ])("maps the dtype spelling %s to %s and back", (external, internal) => {
    const graph = parseDSL(`X = Tensor(4, dtype=${external})\n`);
    expect(graph.tensors.X.dtype).toBe(internal);
    expect(toDSL(graph)).toContain(`dtype=${external}`);
  });

  /* The removed grammar put the dtype last as a bare word: `input A [M, K] f16`.
     Read as a dimension that word is a symbol, and the mistake used to surface
     three phases later as `unbound symbolic dim "fp16"` - a true statement
     about a graph nobody meant to write. */
  it("names the mistake when a dtype is written where a dimension belongs", () => {
    expect(parseFails("X = Tensor(4, fp16)\n").detail).toContain("write dtype=fp16");
    expect(parseFails("X = Tensor(a=4, b=fp16)\n").detail).toContain("write dtype=fp16");
    // Only an exact spelling: a symbol may still be named after one.
    expect(parseDSL("fp16x = 4\nX = Tensor(fp16x, dtype=fp32)\n").tensors.X.shape).toEqual([
      "fp16x",
    ]);
  });

  /* The constructor always wins in call position, so a tensor bound to one of
     these names could never be read back. */
  it.each([
    ["Tensor = Tensor(4, dtype=fp32)\n"],
    ["X = Tensor(4, dtype=fp32)\nParameter = relu(X)\n"],
    ["X = Tensor(4, 4, dtype=fp32)\nA, Tensor = split(X, axis=0, sizes=[2, 2])\n"],
  ])("refuses a constructor name as a tensor name: %s", (source) => {
    expect(parseFails(source).detail).toContain("cannot name a tensor");
  });

  it("still allows a declaration with no dtype at all", () => {
    const graph = parseDSL("X = Tensor(4, 8)\n");
    expect(graph.tensors.X.dtype).toBe("f32");
  });
});

describe("numeric literals", () => {
  it("reads scientific notation in attributes", () => {
    // Asserted on the unresolved graph: no operation declares a float attribute
    // today, and whether one is accepted is the resolver's business, not the
    // lexer's.
    const graph = parseDSL('X = Tensor(4, dtype=fp32)\nY = identity(X, probe=1e-5)');
    expect(graph.nodes[0].attrs.probe).toBe(1e-5);
  });

  it("reads exponents and leading-dot decimals", () => {
    const graph = parseDSL('X = Tensor(4, dtype=fp32)\nY = identity(X, a=2.5E+3, b=.5, c=-1.5e-2)');
    expect(graph.nodes[0].attrs).toMatchObject({ a: 2500, b: 0.5, c: -0.015 });
  });

  it("reads scientific notation in dimensions and shapes", () => {
    const graph = parseDSL("N = 2e3\nX = Tensor(N, 1e2, dtype=fp32)\n");
    expect(graph.params.N).toBe(2000);
    expect(graph.tensors.X.shape).toEqual(["N", 100]);
  });
});

describe("toDSL keeps what the call does not already say", () => {
  it("writes out elementwise attributes beyond fn and nary", () => {
    // `fn` and `nary` are recovered from the call itself; anything else has to
    // be serialized, or expanding a composite and recompiling loses it.
    const graph = parseDSL("X = Tensor(4, dtype=fp32)\nY = relu(X, alpha=0.2)\n");
    expect(toDSL(graph)).toContain("relu(X, alpha=0.2)");
    expect(parseDSL(toDSL(graph)).nodes[0].attrs).toMatchObject({ alpha: 0.2 });
  });
});
