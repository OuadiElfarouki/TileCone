import { describe, expect, it } from "vitest";
import { compileDSL, tryCompileDSL } from "../parse/compiler";
import { toDSL } from "../parse/dsl";
import { executeQuery } from "../core/executor";
import { opLabel } from "../core/ops/index";
import { box, fromBox } from "../core/region";

/** The same three-layer chain, with the middle operation unknown or opaque. */
const chain = (middle: string) => `M = 4
K = 6
X = Tensor(rows=M, cols=K, dtype=fp16)
S = Parameter(cols=K, dtype=fp16)
H = ${middle}
Y = relu(H)
`;

const cone = (source: string, tensorId: string, region: ReturnType<typeof fromBox>) =>
  executeQuery(compileDSL(source).resolved, { tensorId, region, direction: "backward" }).backward!;

describe("an operation carried as a barrier", () => {
  /* The reason the op exists: one unrecognised node used to take the rest of
     the graph with it, and a cone that stops early is a subset of the truth. */
  it("keeps the graph whole where an unknown op truncates it", () => {
    const unknown = tryCompileDSL(chain("batch_norm(X, S)"));
    expect(unknown.ok).toBe(false);
    if (unknown.ok) throw new Error("unreachable");
    expect(unknown.diagnostics.map((d) => d.code)).toContain("SEM_UNKNOWN_OP");

    const barrier = compileDSL(chain('opaque(X, S, op="BatchNorm", shapes=[[M, K]])'));
    expect(barrier.resolved.topo.map((n) => n.op)).toEqual(["opaque", "elementwise"]);
    // The cone reaches the graph's inputs rather than stopping at the barrier.
    const reached = [...cone(chain('opaque(X, S, op="BatchNorm", shapes=[[M, K]])'), "Y",
      fromBox(box([0, 1], [0, 1]))).tensors.keys()];
    expect(reached.sort()).toEqual(["H", "S", "X", "Y"]);
  });

  it("reads every input in full, named by the operation it stands for", () => {
    const res = cone(
      chain('opaque(X, S, op="BatchNorm", shapes=[[M, K]])'),
      "Y",
      fromBox(box([0, 1], [2, 3]))
    );
    const onX = res.tensors.get("X")!.region;
    const onS = res.tensors.get("S")!.region;

    expect(onX.boxes).toEqual([box([0, 4], [0, 6])]);
    expect(onS.boxes).toEqual([box([0, 6])]);
    // Never presented as ground truth: this bounds an operation nobody described.
    expect([onX.exact, onS.exact]).toEqual([false, false]);
    expect(onX.reasons).toEqual(['opaque op "BatchNorm"']);
    // Everything before the barrier is still exact.
    expect(res.tensors.get("H")!.region.exact).toBe(true);
  });

  it("reaches every output in full from any input", () => {
    const forward = executeQuery(
      compileDSL(chain('opaque(X, S, op="BatchNorm", shapes=[[M, K]])')).resolved,
      { tensorId: "S", region: fromBox(box([1, 2])), direction: "forward" }
    ).forward!;
    const onY = forward.tensors.get("Y")!.region;
    expect(onY.boxes).toEqual([box([0, 4], [0, 6])]);
    expect(onY.exact).toBe(false);
  });

  it("requires one declared shape per output", () => {
    const bad = tryCompileDSL(chain('opaque(X, S, op="Split", shapes=[[M, K], [M, K]])'));
    expect(bad.ok).toBe(false);
    if (bad.ok) throw new Error("unreachable");
    expect(bad.diagnostics[0].message).toContain("2 output shape(s) for 1 output");
  });

  it("takes a declared dtype over promotion of its inputs", () => {
    const promoted = compileDSL(chain('opaque(X, S, op="Add", shapes=[[M, K]])'));
    expect(promoted.resolved.tensors.H.dtype).toBe("f16");

    const declared = compileDSL(
      chain('opaque(X, S, op="ArgMax", shapes=[[M, K]], dtypes=[int64])')
    );
    expect(declared.resolved.tensors.H.dtype).toBe("i64");
  });

  /* The reason one dtype for all outputs was not enough. TopK returns its
     values and its indices from one node, in different types, and bytes are
     measured per tensor from that tensor's own dtype - so a single declared
     type would misreport the footprint of whichever output it did not fit. */
  it("gives each output its own dtype", () => {
    const source = `M = 4
K = 6
X = Tensor(rows=M, cols=K, dtype=fp16)
V, I = opaque(X, op="TopK", shapes=[[M, 3], [M, 3]], dtypes=[fp16, int64])
`;
    const { resolved } = compileDSL(source);
    expect(resolved.tensors.V.dtype).toBe("f16");
    expect(resolved.tensors.I.dtype).toBe("i64");
    expect(resolved.tensors.I.resolved).toEqual([4, 3]);
  });

  it("lets an output fall back to promotion while its sibling is declared", () => {
    const source = `M = 4
K = 6
X = Tensor(rows=M, cols=K, dtype=fp16)
V, I = opaque(X, op="MaxPool", shapes=[[M, 3], [M, 3]], dtypes=[null, int64])
`;
    const { resolved } = compileDSL(source);
    expect(resolved.tensors.V.dtype).toBe("f16");
    expect(resolved.tensors.I.dtype).toBe("i64");
  });

  it("requires the declared dtypes to pair with the outputs", () => {
    const bad = tryCompileDSL(
      chain('opaque(X, S, op="TopK", shapes=[[M, K]], dtypes=[fp16, int64])')
    );
    expect(bad.ok).toBe(false);
    if (bad.ok) throw new Error("unreachable");
    expect(bad.diagnostics[0].message).toContain("2 output dtype(s) for 1 output");
  });

  it("qualifies a vendor operation by its domain, everywhere it is named", () => {
    const source = chain(
      'opaque(X, S, op="Attention", domain="com.microsoft", opset=1, shapes=[[M, K]])'
    );
    const { resolved } = compileDSL(source);
    expect(opLabel(resolved.nodes[0])).toBe("com.microsoft.Attention");

    // The same words on the approximation reason as on the card, so a reader
    // can trace a hatched region back to the operation that widened it.
    const res = cone(source, "Y", fromBox(box([0, 1], [0, 1])));
    expect(res.tensors.get("X")!.region.reasons).toEqual([
      'opaque op "com.microsoft.Attention"',
    ]);
  });

  it("leaves an operation in the default domain unqualified", () => {
    for (const domain of ['domain=""', 'domain="ai.onnx"', ""]) {
      const attrs = ['op="Resize"', domain, "shapes=[[M, K]]"].filter(Boolean).join(", ");
      const { resolved } = compileDSL(chain(`opaque(X, S, ${attrs})`));
      expect(opLabel(resolved.nodes[0])).toBe("Resize");
    }
  });

  /* Unusual, and representable on purpose: a converter meeting a constant-like
     node needs somewhere to put it, and the alternative to representing it is
     dropping it - which shortens the graph, which is the failure this op
     exists to prevent. */
  it("carries a node that reads nothing, given a dtype it cannot promote", () => {
    const source = `M = 4
K = 6
X = Tensor(rows=M, cols=K, dtype=fp16)
C = opaque(op="ConstantOfShape", shapes=[[M, K]], dtypes=[fp16])
Y = add(X, C)
`;
    const { resolved } = compileDSL(source);
    expect(resolved.tensors.C.dtype).toBe("f16");

    // It bounds nothing, because it reads nothing: the cone stops there
    // legitimately rather than by truncation.
    const res = cone(source, "Y", fromBox(box([0, 1], [0, 1])));
    expect(res.tensors.get("C")!.region.exact).toBe(true);
    expect([...res.tensors.keys()].sort()).toEqual(["C", "X", "Y"]);
  });

  it("refuses a node that reads nothing and declares no dtype, rather than inventing one", () => {
    const bad = tryCompileDSL(`M = 4
K = 6
C = opaque(op="ConstantOfShape", shapes=[[M, K]])
`);
    expect(bad.ok).toBe(false);
    if (bad.ok) throw new Error("unreachable");
    expect(bad.diagnostics[0].message).toContain("no declared dtype and no inputs to promote from");
  });

  /* The printer dropped the attribute's name when it recursed into a list, so
     a per-output dtype list came back out in canonical spellings that the
     grammar does not accept. A round trip is the only thing that catches it:
     both halves look right on their own. */
  it("round-trips per-output dtypes and provenance", () => {
    const source = `M = 4
K = 6
X = Tensor(rows=M, cols=K, dtype=fp16)
V, I = opaque(X, op="TopK", domain="com.microsoft", opset=17, sourceName="/head/TopK", shapes=[[M, 3], [M, 3]], dtypes=[null, int64])
`;
    const printed = toDSL(compileDSL(source).graph);
    expect(printed).toContain("dtypes=[null, int64]");
    expect(printed).toContain('sourceName="/head/TopK"');

    const again = compileDSL(printed).resolved;
    expect(again.tensors.V.dtype).toBe("f16");
    expect(again.tensors.I.dtype).toBe("i64");
    expect(again.nodes[0].attrs).toMatchObject({
      domain: "com.microsoft",
      opset: 17,
      sourceName: "/head/TopK",
      dtypes: [null, "i64"],
    });
  });

  it("round-trips through the printer, including a name no identifier could hold", () => {
    const source = chain('opaque(X, S, op="/model/layers.0/BatchNorm", shapes=[[M, K]])');
    const printed = toDSL(compileDSL(source).graph);
    expect(printed).toContain('op="/model/layers.0/BatchNorm"');
    expect(compileDSL(printed).resolved.nodes[0].attrs.op).toBe("/model/layers.0/BatchNorm");
  });
});
