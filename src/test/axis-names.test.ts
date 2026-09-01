import { describe, expect, it } from "vitest";
import { compileDSL, tryCompileDSL } from "../parse/compiler";
import { DSLError, parseDSL, toDSL } from "../parse/dsl";
import { parseGraphJSON } from "../parse/json";
import { resolveGraph } from "../core/graph";

const namesOf = (src: string, tensor: string) =>
  compileDSL(src).resolved.tensors[tensor].axisNames ?? null;

describe("declaring axis names", () => {
  it("labels axes alongside their dimensions", () => {
    const graph = parseDSL("input X [batch: B, seq: 128, emb: E] f16\n");
    expect(graph.tensors.X.axisNames).toEqual(["batch", "seq", "emb"]);
    expect(graph.tensors.X.shape).toEqual(["B", 128, "E"]);
  });

  it("leaves an unlabelled declaration unnamed", () => {
    const graph = parseDSL("input X [4, 8] f16\n");
    expect(graph.tensors.X.axisNames).toBeUndefined();
  });

  it("refuses a partly named declaration", () => {
    // Propagation produces holes legitimately; a hand-written half-naming is a
    // slip, so the two cases are treated differently on purpose.
    expect(() => parseDSL("input X [batch: 2, 8] f16\n")).toThrow(DSLError);
    expect(() => parseDSL("input X [batch: 2, 8] f16\n")).toThrow(/name every axis or none/);
  });

  it("refuses a repeated axis name", () => {
    expect(() => parseDSL("input X [d: 2, d: 8] f16\n")).toThrow(/duplicate axis name/);
  });

  it("round-trips through toDSL", () => {
    const source = "input X [batch: 2, seq: 8] f16\n";
    expect(toDSL(parseDSL(source))).toBe(source);
  });

  it("enforces declaration invariants at the JSON/resolver boundary too", () => {
    const graph = parseGraphJSON(JSON.stringify({
      nodes: [],
      tensors: {
        X: { id: "X", name: "X", shape: [2, 8], dtype: "f32", axisNames: ["batch", null] },
      },
      params: {},
    }));
    expect(() => resolveGraph(graph)).toThrow(/name every axis or none/);

    graph.tensors.X.axisNames = ["batch", "batch"];
    expect(() => resolveGraph(graph)).toThrow(/duplicate axis name/);
  });

  it("rejects an axis-name count that is not parallel to the input rank", () => {
    const graph = parseDSL("input X [2, 8] f32\n");
    graph.tensors.X.axisNames = ["batch"];
    expect(() => resolveGraph(graph)).toThrow(/1 axis names for rank 2/);
  });
});

describe("carrying axis names through operations", () => {
  const attention = `params B=1 H=4 S=128 D=32 E=128
input X  [batch: B, seq: S, emb: E] f16
input Wq [emb: E, proj: E] f16
Qp = einsum("bse,ef->bsf", X, Wq)
Q4 = reshape(Qp, shape=[B, S, H, D])
Qh = transpose(Q4, perm=[0, 2, 1, 3])
`;

  it("carries a label through an einsum contraction", () => {
    // `b` and `s` come from X, `f` from Wq; `e` is contracted away with its name.
    expect(namesOf(attention, "Qp")).toEqual(["batch", "seq", "proj"]);
  });

  it("names only the axes a reshape passes through one-to-one", () => {
    // [batch, seq, emb] -> [B, S, H, D]: batch and seq survive, but the split of
    // emb into heads produces two axes the source never named.
    expect(namesOf(attention, "Q4")).toEqual(["batch", "seq", undefined, undefined]);
  });

  it("moves names with their axes under a transpose", () => {
    expect(namesOf(attention, "Qh")).toEqual(["batch", undefined, "seq", undefined]);
  });

  it("drops a reduced axis and keeps it under keepdim", () => {
    const src = "input X [batch: 2, seq: 4, emb: 8] f16\n";
    expect(namesOf(src + "Y = sum(X, axis=1)\n", "Y")).toEqual(["batch", "emb"]);
    expect(namesOf(src + "Y = sum(X, axis=1, keepdim=true)\n", "Y")).toEqual([
      "batch",
      "seq",
      "emb",
    ]);
  });

  it("does not let a broadcast axis name the axis it was stretched against", () => {
    // A's axis 0 has extent 1 and is a different axis from B's 3-wide one, so
    // "one" must not end up naming it; the shared 4-wide axis does inherit.
    const names = namesOf(
      `input A [one: 1, wide: 4] f32
input B [rows: 3, cols: 4] f32
C = add(A, B)
`,
      "C"
    );
    expect(names).toEqual(["rows", "wide"]);
  });

  it("keeps the layout of every piece of a split", () => {
    const program = compileDSL(`input X [batch: 2, feat: 8] f32
A, B = split(X, axis=1, sizes=[3, 5])
`);
    expect(program.resolved.tensors.A.axisNames).toEqual(["batch", "feat"]);
    expect(program.resolved.tensors.B.axisNames).toEqual(["batch", "feat"]);
  });

  it("takes convolution output channels from the weight, not the activation", () => {
    const names = namesOf(
      `input X [batch: 2, in_channel: 3, height: 8, width: 8] f32
weight W [out_channel: 5, kernel_in: 3, kernel_h: 3, kernel_w: 3] f32
Y = conv(X, W, stride=[1, 1], pads=[[1, 1], [1, 1]], dilation=[1, 1])
`,
      "Y"
    );
    expect(names).toEqual(["batch", "out_channel", "height", "width"]);
  });

  it("takes a gathered axis name from index positions, not data coordinates", () => {
    const names = namesOf(
      `input D [vocab: 16, embedding: 8] f32
input I [token: 4] i32
Y = gather(D, I, axis=0, indexValues=[1, 5, 2, 7])
`,
      "Y"
    );
    expect(names).toEqual(["token", "embedding"]);
  });

  it("leaves outputs unnamed when nothing upstream was named", () => {
    expect(namesOf("input X [2, 4] f32\nY = relu(X)\n", "Y")).toBeNull();
  });

  it("does not make a previously valid graph fail", () => {
    const result = tryCompileDSL(`params M=4 K=8 N=2
input A [M, K] f16
weight B [K, N] f16
C = matmul(A, B)
`);
    expect(result.ok).toBe(true);
  });
});
