import { describe, expect, it } from "vitest";
import { compileDSL } from "../parse/compiler";
import {
  hasSymbolicShape,
  shapeLabel,
  shapeReadings,
  symbolicExtentLabel,
} from "../ui/shape-label";
import { resolveShape } from "../core/shapes";

const tensorIn = (dsl: string, id: string) => compileDSL(dsl).resolved.tensors[id];

describe("reading a shape as names or extents", () => {
  const NAMED = `params B=1 H=4 S=32 D=8
input X  [batch: B, seq: S, emb: H*D] f16
input Wq [emb: H*D, proj: H*D] f16
Qp = einsum("bse,ef->bsf", X, Wq)
Q4 = reshape(Qp, shape=[B, S, H, D])
`;

  it("names every axis of a declared tensor", () => {
    const tensor = tensorIn(NAMED, "X");
    expect(shapeLabel(tensor, "symbolic")).toBe("[batch × seq × emb]");
    expect(symbolicExtentLabel(tensor)).toBe("[B × S × H*D]");
    expect(shapeLabel(tensor, "numeric")).toBe("[1 × 32 × 32]");
  });

  it("carries names onto a produced tensor", () => {
    expect(shapeLabel(tensorIn(NAMED, "Qp"), "symbolic")).toBe("[batch × seq × proj]");
  });

  it("takes a reshape's split axes from the target shape the author wrote", () => {
    // Neither new axis was named and neither is a one-to-one group, but the
    // statement `shape=[B, S, H, D]` says what they are.
    expect(shapeLabel(tensorIn(NAMED, "Q4"), "symbolic")).toBe("[batch × seq × H × D]");
  });

  it("falls back to the extent for an axis with nothing to say", () => {
    // A literal target leaves nothing symbolic behind.
    const SPLIT = "input X [2, 12] f32\nY = reshape(X, shape=[2, 3, 4])\n";
    expect(shapeLabel(tensorIn(SPLIT, "Y"), "symbolic")).toBe("[2 × 3 × 4]");
  });

  it("falls back to the declared dimension when the axis is unnamed", () => {
    const GEMM = `params M=64 N=64 K=128
input A [M, K] f16
input B [K, N] f16
C = matmul(A, B)
`;
    expect(shapeLabel(tensorIn(GEMM, "A"), "symbolic")).toBe("[M × K]");
    // C is produced, so it has no declared shape — but matmul carries M and N
    // across the contraction, and K is the axis that disappears into it.
    expect(shapeLabel(tensorIn(GEMM, "C"), "symbolic")).toBe("[M × N]");
    expect(hasSymbolicShape(tensorIn(GEMM, "C"))).toBe(true);
    expect(hasSymbolicShape(tensorIn(GEMM, "A"))).toBe(true);
  });

  it("reads identically both ways when nothing is symbolic", () => {
    const PLAIN = "input X [2, 4] f32\nY = relu(X)\n";
    expect(shapeLabel(tensorIn(PLAIN, "X"), "symbolic")).toBe("[2 × 4]");
    expect(hasSymbolicShape(tensorIn(PLAIN, "X"))).toBe(false);
  });
});

/* Symbolic extents are a separate propagation from names: a name says what an
   axis is, a symbol says how wide it is, and operations lose them at different
   points. Everything an operation proposes is verified against the extents
   resolution actually produced, so a wrong mapping degrades to the number. */
describe("carrying symbolic extents downstream", () => {
  const SYM = `params M=8 K=4 N=6
input A [M, K] f32
input B [K, N] f32
C = matmul(A, B)
D = relu(C)
E = transpose(D, perm=[1, 0])
F = sum(E, axis=1)
`;

  it("carries a contraction's free axes and drops the contracted one", () => {
    expect(tensorIn(SYM, "C").symShape).toEqual(["M", "N"]);
  });

  it("carries through elementwise and transpose", () => {
    expect(tensorIn(SYM, "D").symShape).toEqual(["M", "N"]);
    expect(tensorIn(SYM, "E").symShape).toEqual(["N", "M"]);
  });

  it("drops the symbol of an axis a reduction collapses", () => {
    expect(tensorIn(SYM, "F").symShape).toEqual(["N"]);
  });

  it("states a concatenated axis as the sum that met there", () => {
    const CAT = `params P=6 T=2
input A [4, P] f32
input B [4, T] f32
C = concat(A, B, axis=1)
`;
    expect(tensorIn(CAT, "C").symShape).toEqual([4, "P+T"]);
    expect(shapeLabel(tensorIn(CAT, "C"), "symbolic")).toBe("[4 × P+T]");
  });

  it("can take an unchanged concat symbol from a later input", () => {
    const CAT = `params Rows=4 P=6 T=2
input A [4, P] f32
input B [Rows, T] f32
C = concat(A, B, axis=1)
`;
    expect(tensorIn(CAT, "C").symShape).toEqual(["Rows", "P+T"]);
  });

  it("carries only the unchanged extents through convolution and pooling", () => {
    const SPATIAL = `params N=2 Cin=3 Cout=5 H=8 W=8 KH=3 KW=3
input X [N, Cin, H, W] f32
input K [Cout, Cin, KH, KW] f32
Y = conv(X, K, stride=[1, 1], pads=[[1, 1], [1, 1]], dilation=[1, 1])
Z = pool(Y, kind=max, kernelShape=[2, 2], stride=[2, 2], pads=[[0, 0], [0, 0]])
`;
    expect(tensorIn(SPATIAL, "Y").symShape).toEqual(["N", "Cout", 8, 8]);
    expect(tensorIn(SPATIAL, "Z").symShape).toEqual(["N", "Cout", 4, 4]);
  });

  it("takes a gather axis extent from the indices tensor", () => {
    const GATHER = `params Rows=8 Cols=4 Picked=3
input D [Rows, Cols] f32
input I [Picked] i32
Y = gather(D, I, axis=0, indexValues=[1, 5, 2])
`;
    expect(tensorIn(GATHER, "Y").symShape).toEqual(["Picked", "Cols"]);
  });

  it("always evaluates to the extents that were actually inferred", () => {
    // The invariant the resolver enforces on every axis of every tensor.
    const { resolved } = compileDSL(SYM);
    for (const tensor of Object.values(resolved.tensors)) {
      expect(tensor.symShape).toHaveLength(tensor.resolved!.length);
      expect(resolveShape(tensor.symShape!, resolved.params)).toEqual(tensor.resolved);
    }
  });
});

/* The details list three readings, but only where they are three facts. A
   reading is dropped when a less symbolic one says the same thing, so the row
   that survives is named after what it actually is. */
describe("which readings a tensor's details are worth listing", () => {
  const readingsOf = (dsl: string, id: string) =>
    shapeReadings(tensorIn(dsl, id)).map((r) => `${r.label} ${r.value}`);

  it("lists all three when they are three different facts", () => {
    const NAMED = "params E=8\ninput X [emb: E, w: 4] f32\n";
    expect(readingsOf(NAMED, "X")).toEqual([
      "labels [emb × w]",
      "symbols [E × 4]",
      "extents [8 × 4]",
    ]);
  });

  it("collapses to one row for a tensor with nothing symbolic about it", () => {
    // Naming this row "labels" would claim an identity the tensor has not got.
    expect(readingsOf("input X [2, 4] f32\n", "X")).toEqual(["extents [2 × 4]"]);
  });

  it("keeps the symbolic row when only the names are missing", () => {
    const SYM = "params M=8 K=4\ninput A [M, K] f32\n";
    expect(readingsOf(SYM, "A")).toEqual(["symbols [M × K]", "extents [8 × 4]"]);
  });

  it("keeps the label row when the names are the only symbol", () => {
    expect(readingsOf("input X [rows: 2, cols: 4] f32\n", "X")).toEqual([
      "labels [rows × cols]",
      "extents [2 × 4]",
    ]);
  });
});
