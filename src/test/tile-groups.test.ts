import { describe, expect, it } from "vitest";
import { compileDSL } from "../parse/compiler";
import { groupPropResult, MAX_PER_BOX_PROPS, type BoxProp, type SelPart } from "../ui/store";
import { executeQuery } from "../core/executor";
import { box, count, fromBox } from "../core/region";
import type { ResolvedGraph } from "../core/graph";
import { analysisTensorId, groupAttribution } from "../ui/inspector-analysis";

/** Two tensors where one feeds the other, so their cones genuinely overlap. */
const chain = () =>
  compileDSL(`A = Tensor(64, 64, dtype=fp32)
B = Tensor(64, 64, dtype=fp32)
C = matmul(A, B)
D = relu(C)
`).resolved;

const propsFor = (resolved: ResolvedGraph, parts: SelPart[]): BoxProp[] =>
  parts.map((part) => {
    const r = executeQuery(resolved, {
      tensorId: part.tensorId,
      region: fromBox(part.box),
      direction: "both",
    });
    return { backward: r.backward, forward: r.forward };
  });

const byTensorOf = (resolved: ResolvedGraph, parts: SelPart[]) => {
  const out: Record<string, { backward: any; forward: any }> = {};
  for (const part of parts) {
    const r = executeQuery(resolved, {
      tensorId: part.tensorId,
      region: fromBox(part.box),
      direction: "both",
    });
    out[part.tensorId] = { backward: r.backward, forward: r.forward };
  }
  return out;
};

describe("tile groups", () => {
  const parts: SelPart[] = [
    { tensorId: "C", box: box([0, 16], [0, 16]) },
    { tensorId: "D", box: box([32, 48], [32, 48]) },
  ];

  it("keeps a selected merged group when tile focus clears", () => {
    const tiles = [parts[0], parts[0], parts[1]];
    expect(analysisTensorId(tiles, null, null)).toBe("D");
    expect(analysisTensorId(tiles, null, "C")).toBe("C");
    expect(analysisTensorId(tiles, 2, "C")).toBe("D");
    expect(analysisTensorId(tiles, null, "C")).toBe("C");
    expect(analysisTensorId([parts[1]], null, "C")).toBe("D");
  });

  it("excludes other groups from bars without renumbering their colors", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);
    const scoped = groupAttribution(perBox, parts, "D")!;
    expect(scoped).toHaveLength(2);
    expect(scoped[0]).toEqual({ backward: null, forward: null });
    expect(scoped[1]).toBe(perBox[1]);
    expect(groupAttribution(null, parts, "D")).toBeNull();
  });

  it("analyses one tensor's tiles and never the other's", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);

    const onC = groupPropResult(null, perBox, parts, new Set(), null, "C", "backward")!;
    const onD = groupPropResult(null, perBox, parts, new Set(), null, "D", "backward")!;

    // C's cone stops at A and B. D's reaches through C to A and B as well, but
    // over the region *its* tile needs - a different one.
    expect(count(onC.tensors.get("A")!.region)).toBe(16 * 64);
    expect(onC.tensors.has("D")).toBe(false);
    expect(onD.tensors.has("C")).toBe(true);
    // Same size band, different rows of A: C's tile needs rows 0:16, D's needs
    // 32:48. Merging them would claim a job that reads both, which is nobody's
    // kernel - and would charge the shared work through C only once.
    const rowsOf = (region: { boxes: { lo: number; hi: number }[][] }) =>
      region.boxes.map((b) => `${b[0].lo}:${b[0].hi}`);
    expect(rowsOf(onC.tensors.get("A")!.region)).toEqual(["0:16"]);
    expect(rowsOf(onD.tensors.get("A")!.region)).toEqual(["32:48"]);
  });

  it("honours hidden tiles within a group", () => {
    const resolved = chain();
    const twoOnC: SelPart[] = [
      { tensorId: "C", box: box([0, 16], [0, 16]) },
      { tensorId: "C", box: box([32, 48], [0, 16]) },
      { tensorId: "D", box: box([0, 8], [0, 8]) },
    ];
    const perBox = propsFor(resolved, twoOnC);

    const both = groupPropResult(null, perBox, twoOnC, new Set(), null, "C", "backward")!;
    const one = groupPropResult(null, perBox, twoOnC, new Set([1]), null, "C", "backward")!;

    expect(count(both.tensors.get("A")!.region)).toBe(2 * 16 * 64);
    expect(count(one.tensors.get("A")!.region)).toBe(16 * 64);
  });

  it("ignores a focused tile that belongs to another group", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);

    // Focus is on D (index 1) while the C group is being read: the C answer
    // must be C's tiles, not D's.
    const onC = groupPropResult(null, perBox, parts, new Set(), 1, "C", "backward")!;
    expect(onC.tensors.has("D")).toBe(false);
    expect(count(onC.tensors.get("A")!.region)).toBe(16 * 64);
  });

  it("stays grouped past the attribution cap, where per-tile cones are gone", () => {
    const resolved = chain();
    const byTensor = byTensorOf(resolved, parts);

    // `perBox` is null above MAX_PER_BOX_PROPS. The per-tensor query stands in,
    // so the analysis is coarser but still never mixes two tensors.
    expect(MAX_PER_BOX_PROPS).toBeGreaterThan(0);
    const onC = groupPropResult(byTensor, null, parts, new Set(), null, "C", "backward")!;
    expect(onC.tensors.has("D")).toBe(false);
    expect(groupPropResult(byTensor, null, parts, new Set(), null, "D", "backward")!.tensors.has("C"))
      .toBe(true);
  });

  it("has no answer without an active tensor", () => {
    const resolved = chain();
    const perBox = propsFor(resolved, parts);
    expect(groupPropResult(null, perBox, parts, new Set(), null, null, "backward")).toBeNull();
  });
});
