import { describe, expect, it } from "vitest";
import { compileDSL } from "../../src/parse/compiler";
import { computeMetrics, viewLayouts } from "../../src/core/metrics";
import { propagateBackward } from "../../src/core/propagate";
import { box, fromBox } from "../../src/core/region";

/* The attention head split and merge. The split is a reshape of a computed,
   contiguous tensor followed by a transpose: both views. The merge reshapes a
   transposed tensor, whose axes no single stride can join, so it is a copy. */
const attention = () =>
  compileDSL(`B = 1
S = 8
H = 2
D = 4
X = Tensor(B, S, H*D, dtype=fp32)
Q4 = reshape(X, shape=[B, S, H, D])
Qh = transpose(Q4, perm=[0, 2, 1, 3])
Y = relu(Qh)
Yt = transpose(Y, perm=[0, 2, 1, 3])
Ym = reshape(Yt, shape=[B, S, H*D])
`).resolved;

describe("views in the unfused scenario", () => {
  it("treats a reshape of contiguous data and a transpose as views, and a reshape of a transpose as a copy", () => {
    const resolved = attention();
    const { views, layouts } = viewLayouts(resolved);
    const producer = (tensor: string) => resolved.tensors[tensor].producer!.nodeId;
    expect(views.has(producer("Q4"))).toBe(true);
    expect(views.has(producer("Qh"))).toBe(true);
    expect(views.has(producer("Yt"))).toBe(true);
    expect(views.has(producer("Ym"))).toBe(false);
    expect(layouts.get("Qh")).toBe("strided");
    expect(layouts.get("Ym")).toBe("contiguous");
  });

  it("charges only the computing op and the copy", () => {
    const resolved = attention();
    const back = propagateBackward(resolved, {
      tensorId: "Ym",
      region: fromBox(box([0, 1], [0, 8], [0, 8])),
    });
    const metrics = computeMetrics(resolved, back);
    // relu reads 64 fp32 elements and writes 64; the merging reshape reads its
    // 64 and writes 64. The split and both transposes move nothing.
    expect(metrics.unfusedBytes.value).toBe(4 * 64 * 4);
  });

  it("keeps a transpose that only moves unit axes contiguous", () => {
    const resolved = compileDSL(`X = Tensor(1, 8, 4, dtype=fp32)
T = transpose(X, perm=[1, 0, 2])
R = reshape(T, shape=[32])
`).resolved;
    const { views } = viewLayouts(resolved);
    expect(views.has(resolved.tensors.R.producer!.nodeId)).toBe(true);
  });
});
