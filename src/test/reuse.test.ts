import { describe, expect, it } from "vitest";
import { executeQuery } from "../core/executor";
import { estimateInputReuse, inputSharing } from "../core/reuse";
import { box, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";

/** One output tile of a GEMM reads a full band of A and a full band of B. */
const GEMM = `M = 4\nN = 4\nK = 8\nA = Tensor(M, K, dtype=fp16)\nB = Tensor(K, N, dtype=fp16)\nC = matmul(A, B)\n`;

const coneOf = (graph: ReturnType<typeof compileDSL>["resolved"], tensorId: string, region: ReturnType<typeof fromBox>) =>
  executeQuery(graph, { tensorId, region, direction: "backward" }).backward!;

describe("reuse estimation", () => {
  it("is exact when every tile fits under the sample cap", () => {
    const { resolved } = compileDSL("X = Tensor(4, dtype=fp32)\nY = identity(X)\n");
    const result = estimateInputReuse(resolved, {
      tensorId: "Y",
      region: fromBox(box([0, 2])),
    });

    expect(result).toEqual([
      {
        tensorId: "X",
        touches: 1,
        probes: 2,
        totalTiles: 2,
        estimatedTiles: 1,
        exhaustive: true,
        meanSharedFraction: 1,
        exact: true,
        reasons: [],
        // The tile below the anchor is out of bounds and is not resized to fit.
        neighbors: [{ axis: 0, delta: 1, sharedFraction: 0, exact: true }],
      },
    ]);
  });

  it("counts every tile that shares a band, and how much of it", () => {
    const { resolved } = compileDSL(GEMM);
    // C[0:2, 0:2] reads all of A's first two rows and all of B's first two columns.
    const [a, b] = estimateInputReuse(resolved, {
      tensorId: "C",
      region: fromBox(box([0, 2], [0, 2])),
    });

    // Four tiles of this size cover C; the two in the same tile-row read the
    // same rows of A, and the two in the same tile-column the same columns of B.
    expect(a.totalTiles).toBe(4);
    expect(a.exhaustive).toBe(true);
    expect([a.tensorId, a.estimatedTiles, a.meanSharedFraction]).toEqual(["A", 2, 1]);
    expect([b.tensorId, b.estimatedTiles, b.meanSharedFraction]).toEqual(["B", 2, 1]);
  });

  it("reports what a neighbouring tile on each axis still shares", () => {
    const { resolved } = compileDSL(GEMM);
    const [a, b] = estimateInputReuse(resolved, {
      tensorId: "C",
      region: fromBox(box([0, 2], [0, 2])),
    });

    // Stepping along C's columns keeps A's rows and moves off B's columns.
    expect(a.neighbors).toEqual([
      { axis: 0, delta: 1, sharedFraction: 0, exact: true },
      { axis: 1, delta: 1, sharedFraction: 1, exact: true },
    ]);
    expect(b.neighbors).toEqual([
      { axis: 0, delta: 1, sharedFraction: 1, exact: true },
      { axis: 1, delta: 1, sharedFraction: 0, exact: true },
    ]);
  });

  it("weights each sample by the stratum it was drawn from", () => {
    // 50 tiles, 12 probes: an unweighted count would report the hit rate times
    // the grid, which uneven strata bias. Every tile of this graph touches its
    // own slice of X and nothing else, so exactly one tile can be reported.
    const { resolved } = compileDSL("X = Tensor(100, dtype=fp32)\nY = identity(X)\n");
    const root = { tensorId: "Y", region: fromBox(box([0, 2])) };
    const [sampled] = estimateInputReuse(resolved, root, { sampleCap: 12, seed: 1234 });

    expect(sampled.probes).toBe(12);
    expect(sampled.totalTiles).toBe(50);
    expect(sampled.exhaustive).toBe(false);
    // The stratum holding tile 0 is four or five tiles wide, never one.
    expect(sampled.estimatedTiles).toBeGreaterThanOrEqual(4);
    expect(sampled.estimatedTiles).toBeLessThanOrEqual(5);
  });

  it("is reproducible when the estimate is sampled", () => {
    const { resolved } = compileDSL("X = Tensor(100, dtype=fp32)\nY = identity(X)\n");
    const root = { tensorId: "Y", region: fromBox(box([0, 2])) };
    const first = estimateInputReuse(resolved, root, { sampleCap: 12, seed: 1234 });

    expect(estimateInputReuse(resolved, root, { sampleCap: 12, seed: 1234 })).toEqual(first);
  });

  it("rejects a sample cap that cannot describe a sweep", () => {
    const { resolved } = compileDSL("X = Tensor(4, dtype=fp32)\nY = identity(X)\n");
    const root = { tensorId: "Y", region: fromBox(box([0, 2])) };
    expect(() => estimateInputReuse(resolved, root, { sampleCap: 0 })).toThrow(/sample cap/);
  });
});

describe("input sharing across the enabled tiles", () => {
  it("separates bytes read twice from bytes read once", () => {
    const { resolved } = compileDSL(GEMM);
    // Two tiles side by side in the same tile-row of C.
    const cones = [
      coneOf(resolved, "C", fromBox(box([0, 2], [0, 2]))),
      coneOf(resolved, "C", fromBox(box([0, 2], [2, 4]))),
    ];
    const [a, b] = inputSharing(resolved, cones);

    // Both tiles read the same 2 x 8 band of A: half of what they read is a re-read.
    expect(a).toEqual({
      tensorId: "A",
      tiles: 2,
      independentBytes: 2 * 2 * 8 * 2,
      unionBytes: 2 * 8 * 2,
      duplicateBytes: 2 * 8 * 2,
      exact: true,
      reasons: [],
    });
    // The two tiles read disjoint columns of B, so nothing is fetched twice.
    expect([b.tensorId, b.duplicateBytes, b.independentBytes]).toEqual([
      "B",
      0,
      2 * 8 * 2 * 2,
    ]);
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
