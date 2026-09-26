import { describe, expect, it } from "vitest";
import { compileDSL } from "../../src/parse/compiler";
import { computeMetrics, viewLayouts } from "../../src/core/metrics";
import { isRowMajor } from "../../src/core/ops/types";
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
    // Qh is Q4's buffer read in head-major order, not a new buffer.
    expect(layouts.get("Qh")).toEqual({ order: [0, 2, 1, 3] });
    expect(layouts.get("Ym")).toEqual({ order: [0, 1, 2] });
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

  it("brings a transpose followed by its inverse back to row-major", () => {
    const resolved = compileDSL(`X = Tensor(8, 4, dtype=fp32)
T = transpose(X, perm=[1, 0])
U = transpose(T, perm=[1, 0])
R = reshape(U, shape=[32])
`).resolved;
    const { views, layouts } = viewLayouts(resolved);
    expect(layouts.get("T")).toEqual({ order: [1, 0] });
    expect(layouts.get("U")).toEqual({ order: [0, 1] });
    // Row-major again, so the reshape is a view and nothing is copied.
    expect(views.has(resolved.tensors.R.producer!.nodeId)).toBe(true);
  });

  it("composes non-trivial permutations and only calls the result row-major when it is", () => {
    const resolved = compileDSL(`X = Tensor(2, 3, 4, dtype=fp32)
A = transpose(X, perm=[1, 2, 0])
B = transpose(A, perm=[2, 0, 1])
C = transpose(X, perm=[2, 0, 1])
D = transpose(C, perm=[2, 0, 1])
RB = reshape(B, shape=[24])
RD = reshape(D, shape=[24])
`).resolved;
    const { views, layouts } = viewLayouts(resolved);
    // [1, 2, 0] then [2, 0, 1] is the identity; applying [2, 0, 1] twice is not.
    expect(layouts.get("B")).toEqual({ order: [0, 1, 2] });
    expect(views.has(resolved.tensors.RB.producer!.nodeId)).toBe(true);
    expect(views.has(resolved.tensors.RD.producer!.nodeId)).toBe(false);
  });

  it("agrees with the buffer addresses a random transpose chain actually reads", () => {
    // Deterministic LCG so the corpus is the same on every run.
    let seed = 12345;
    const rand = (n: number) => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed % n;
    };
    const shuffle = (rank: number) => {
      const perm = Array.from({ length: rank }, (_, i) => i);
      for (let i = rank - 1; i > 0; i--) {
        const j = rand(i + 1);
        [perm[i], perm[j]] = [perm[j], perm[i]];
      }
      return perm;
    };
    let checked = 0;
    let rowMajorSeen = 0;
    for (let trial = 0; trial < 200; trial++) {
      const rank = 2 + rand(3);
      const shape = Array.from({ length: rank }, () => 1 + rand(3));
      const perms = Array.from({ length: 1 + rand(3) }, () => shuffle(rank));
      const lines = [`X = Tensor(${shape.join(", ")}, dtype=fp32)`];
      perms.forEach((perm, i) => lines.push(`T${i} = transpose(${i ? `T${i - 1}` : "X"}, perm=[${perm}])`));
      const resolved = compileDSL(lines.join("\n") + "\n").resolved;
      const last = `T${perms.length - 1}`;
      const finalShape = resolved.tensors[last].resolved!;
      // Truth: which original axis each final axis is, then the original
      // buffer address of every final element visited in row-major order.
      let source = shape.map((_, axis) => axis);
      for (const perm of perms) source = perm.map((axis) => source[axis]);
      const strides = shape.map((_, axis) => shape.slice(axis + 1).reduce((a, b) => a * b, 1));
      const addresses: number[] = [];
      const index = finalShape.map(() => 0);
      const total = finalShape.reduce((a, b) => a * b, 1);
      for (let n = 0; n < total; n++) {
        addresses.push(index.reduce((sum, i, axis) => sum + i * strides[source[axis]], 0));
        for (let axis = finalShape.length - 1; axis >= 0; axis--) {
          if (++index[axis] < finalShape[axis]) break;
          index[axis] = 0;
        }
      }
      const truth = addresses.every((address, n) => address === n);
      const layout = viewLayouts(resolved).layouts.get(last)!;
      expect(isRowMajor(layout, finalShape), `${shape} ${JSON.stringify(perms)}`).toBe(truth);
      checked++;
      if (truth) rowMajorSeen++;
    }
    // The corpus has to exercise both answers, or agreement proves little.
    expect(checked).toBe(200);
    expect(rowMajorSeen).toBeGreaterThan(20);
    expect(rowMajorSeen).toBeLessThan(180);
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
