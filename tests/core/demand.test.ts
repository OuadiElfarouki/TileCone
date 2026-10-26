import { describe, expect, it } from "vitest";
import { executeQuery } from "../../src/core/executor";
import { inputSharing } from "../../src/core/demand";
import { box, fromBox } from "../../src/core/region";
import { compileDSL } from "../../src/parse/compiler";

/** One output tile of a GEMM reads a full band of A and a full band of B. */
const GEMM = `M = 4\nN = 4\nK = 8\nA = Tensor(M, K, dtype=fp16)\nB = Tensor(K, N, dtype=fp16)\nC = matmul(A, B)\n`;

const coneOf = (graph: ReturnType<typeof compileDSL>["resolved"], tensorId: string, region: ReturnType<typeof fromBox>) =>
  executeQuery(graph, { tensorId, region, direction: "backward" }).backward!;

describe("input sharing across the enabled tiles", () => {
  it("separates duplicate graph-input demand from distinct demand", () => {
    const { resolved } = compileDSL(GEMM);
    // Two tiles side by side in the same tile-row of C.
    const cones = [
      coneOf(resolved, "C", fromBox(box([0, 2], [0, 2]))),
      coneOf(resolved, "C", fromBox(box([0, 2], [2, 4]))),
    ];
    const [a, b] = inputSharing(resolved, cones);

    // Both cones demand the same 2 x 8 band of A, so half of their summed
    // element demand is duplicate. This does not assume a hardware reload.
    const exact = (value: number) => ({ value, status: "exact", reasons: [] });
    expect(a).toEqual({
      tensorId: "A",
      selectedTiles: 2,
      readers: 2,
      summed: exact(2 * 2 * 8 * 2),
      distinct: exact(2 * 8 * 2),
      duplication: exact(2),
      duplicate: exact(2 * 8 * 2),
      exact: true,
      reasons: [],
    });
    // The two cones demand disjoint columns of B, so none of it is duplicate.
    expect([b.tensorId, b.duplicate.value, b.summed.value]).toEqual([
      "B",
      0,
      2 * 8 * 2 * 2,
    ]);
  });

  it("counts only tiles whose cone reaches a given input", () => {
    const { resolved } = compileDSL(`A = Tensor(2, dtype=fp16)
B = Tensor(2, dtype=fp16)
C = concat([A, B], axis=0)
`);
    const rows = inputSharing(resolved, [
      coneOf(resolved, "C", fromBox(box([0, 2]))),
      coneOf(resolved, "C", fromBox(box([2, 4]))),
    ]);

    expect(rows.map((row) => [row.tensorId, row.readers, row.selectedTiles]))
      .toEqual([["A", 1, 2], ["B", 1, 2]]);
  });

  it("keeps duplicate demand as an upper bound for widened footprints", () => {
    const { resolved } = compileDSL("X = Tensor(8, dtype=fp32)\nY = identity(X)\n");
    const first = coneOf(resolved, "Y", fromBox(box([0, 2])));
    const second = coneOf(resolved, "Y", fromBox(box([3, 5])));
    first.tensors.set("X", {
      region: { boxes: [box([0, 3])], exact: false, reasons: ["test widening"] },
      depth: 1,
    });
    second.tensors.set("X", {
      region: { boxes: [box([2, 5])], exact: false, reasons: ["test widening"] },
      depth: 1,
    });

    const [row] = inputSharing(resolved, [first, second]);
    expect(row.exact).toBe(false);
    expect(row.summed).toMatchObject({ value: 6 * 4, status: "upper" });
    expect(row.distinct).toMatchObject({ value: 5 * 4, status: "upper" });
    // A difference of bounds, still a bound: widening only raises membership.
    expect(row.duplicate).toMatchObject({ value: 1 * 4, status: "upper" });
    expect(row.duplication.status).toBe("approximate");
  });

  it("refuses to compare tiles that live on different tensors", () => {
    const { resolved } = compileDSL(GEMM);
    const cones = [
      coneOf(resolved, "C", fromBox(box([0, 2], [0, 2]))),
      coneOf(resolved, "B", fromBox(box([0, 2], [0, 2]))),
    ];
    expect(() => inputSharing(resolved, cones)).toThrow(/one tensor/);
  });
});
