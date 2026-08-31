import { describe, expect, it } from "vitest";
import { contributions } from "../core/contribution";
import { resolveGraph } from "../core/graph";
import { propagateForward } from "../core/propagate";
import { box, fromBox, Region } from "../core/region";
import { compileDSL } from "../parse/compiler";
import { G } from "./harness";

const MATMUL = `params S=8 K=12 N=8
input P [S, K] f16
input V [K, N] f16
O = matmul(P, V)
`;

/** Forward-propagate one tile and classify what it reaches. */
function report(dsl: string, seeds: [string, Region][]) {
  const { resolved } = compileDSL(dsl);
  const seedMap = new Map(seeds);
  const forwards = seeds.map(([tensorId, region]) =>
    propagateForward(resolved, { tensorId, region })
  );
  const merged = forwards[0];
  for (const other of forwards.slice(1))
    for (const [id, tr] of other.tensors)
      if (!merged.tensors.has(id)) merged.tensors.set(id, tr);
  return contributions(resolved, merged, seedMap);
}

const tile = (...pairs: [number, number][]) => fromBox(box(...pairs));

describe("partial contribution", () => {
  it("flags a tile that reaches a tensor without determining it", () => {
    // half of the contraction axis: O is influenced, but no element of it is
    // finished by this tile alone
    const { byTensor } = report(MATMUL, [["P", tile([0, 4], [0, 6])]]);
    const o = byTensor.get("O")!;
    expect(o.partial).toBe(true);
    expect(o.detail).toContain("accumulates over all 12 along axis 1 of P");
  });

  it("does not flag a tile that spans everything its consumer reads", () => {
    const { byTensor } = report(MATMUL, [["P", tile([0, 4], [0, 12])]]);
    expect(byTensor.get("O")).toMatchObject({ partial: false, detail: null });
  });

  it("is complete through an elementwise chain", () => {
    const dsl = `params S=8
input X [S] f32
Y = relu(X)
Z = relu(Y)
`;
    const { byTensor } = report(dsl, [["X", tile([2, 5])]]);
    expect(byTensor.get("Y")!.partial).toBe(false);
    expect(byTensor.get("Z")!.partial).toBe(false);
  });

  it("names the whole axis when that is what the consumer reads", () => {
    // a prefix scan reaches every later element, and each of those reads the
    // whole prefix -- so the honest statement is the axis, not a count
    const dsl = `params S=16
input X [S] f32
Y = cumsum(X, axis=0, reverse=false)
`;
    const { byTensor } = report(dsl, [["X", tile([4, 8])]]);
    const y = byTensor.get("Y")!;
    expect(y.partial).toBe(true);
    expect(y.detail).toBe("Y accumulates over all 16 along axis 0 of X");
  });

  it("counts the residue when no whole axis explains it", () => {
    // a conv halo: the neighbouring tiles supply the rest, and no axis is
    // consumed in full, so a count is the only true thing to say
    const dsl = `input X [1, 1, 16, 16] f32
input Kw [1, 1, 3, 3] f32
Y = conv(X, Kw, stride=[1, 1], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
`;
    const { byTensor } = report(dsl, [["X", tile([0, 1], [0, 1], [6, 10], [6, 10])]]);
    const y = byTensor.get("Y")!;
    expect(y.partial).toBe(true);
    expect(y.detail).toMatch(/^Y also needs \d+ elements of X outside this tile$/);
  });

  it("never reports the seed tensors themselves", () => {
    const { byTensor } = report(MATMUL, [["P", tile([0, 4], [0, 6])]]);
    expect(byTensor.has("P")).toBe(false);
  });

  it("treats two tiles on different tensors as one contribution", () => {
    // P alone leaves the K axis short; adding all of V does not change that,
    // because the residue is on P.
    const both = report(MATMUL, [
      ["P", tile([0, 4], [0, 12])],
      ["V", tile([0, 12], [0, 8])],
    ]);
    expect(both.byTensor.get("O")!.partial).toBe(false);

    const short = report(MATMUL, [
      ["P", tile([0, 4], [0, 6])],
      ["V", tile([0, 12], [0, 8])],
    ]);
    expect(short.byTensor.get("O")!.partial).toBe(true);
    expect(short.byTensor.get("O")!.detail).toContain("of P");
  });

  it("marks a verdict drawn from an over-approximated region", () => {
    const graph = G({ D: [6, 3], I: [4] }, [["n", "gather", ["D", "I"], ["Y"], { axis: 0 }]]);
    graph.tensors.I.dtype = "i32";
    const resolved = resolveGraph(graph);
    const seed = tile([0, 2], [0, 3]);
    const forward = propagateForward(resolved, { tensorId: "D", region: seed });
    const { byTensor } = contributions(resolved, forward, new Map([["D", seed]]));
    // the gather index is data-dependent, so the probe cannot be exact and the
    // flag has to say so rather than assert a partial contribution outright
    expect(byTensor.get("Y")!.exact).toBe(false);
  });

  it("reports nothing rather than probing an oversized cone", () => {
    const { resolved } = compileDSL(MATMUL);
    const seed = tile([0, 4], [0, 6]);
    const forward = propagateForward(resolved, { tensorId: "P", region: seed });
    const capped = contributions(resolved, forward, new Map([["P", seed]]), 0);
    expect(capped).toEqual({ byTensor: new Map(), capped: true });
  });
});
