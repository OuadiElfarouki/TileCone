import { describe, expect, it } from "vitest";
import { compileDSL, tryCompileDSL } from "../parse/compiler";
import { toDSL } from "../parse/dsl";
import { executeQuery } from "../core/executor";
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
      chain('opaque(X, S, op="ArgMax", shapes=[[M, K]], dtype=int32)')
    );
    expect(declared.resolved.tensors.H.dtype).toBe("i32");
  });

  it("round-trips through the printer, including a name no identifier could hold", () => {
    const source = chain('opaque(X, S, op="/model/layers.0/BatchNorm", shapes=[[M, K]])');
    const printed = toDSL(compileDSL(source).graph);
    expect(printed).toContain('op="/model/layers.0/BatchNorm"');
    expect(compileDSL(printed).resolved.nodes[0].attrs.op).toBe("/model/layers.0/BatchNorm");
  });
});
