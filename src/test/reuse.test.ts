import { describe, expect, it } from "vitest";
import { executeQuery } from "../core/executor";
import { estimateInputReuseSweep, inputSharing, reuseReachAt } from "../core/reuse";
import { box, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";

/** One output tile of a GEMM reads a full band of A and a full band of B. */
const GEMM = `M = 4\nN = 4\nK = 8\nA = Tensor(M, K, dtype=fp16)\nB = Tensor(K, N, dtype=fp16)\nC = matmul(A, B)\n`;

/** A GEMM whose output is itself an operand, so a tile on it has all three
 *  relations: what it reads, what it feeds, and what it is combined with. */
const CHAIN = `${GEMM}W = Tensor(N, N, dtype=fp16)\nE = matmul(C, W)\n`;

const coneOf = (graph: ReturnType<typeof compileDSL>["resolved"], tensorId: string, region: ReturnType<typeof fromBox>) =>
  executeQuery(graph, { tensorId, region, direction: "backward" }).backward!;

/** These cases are about the aggregate rows; the probe trace has its own. */
const estimateInputReuse = (...args: Parameters<typeof estimateInputReuseSweep>) =>
  estimateInputReuseSweep(...args).estimates;

describe("reuse estimation", () => {
  it.each(["__proto__", "constructor", "toString"])(
    "preserves the tensor named %s across the worker boundary", (tensorId) => {
      const { resolved } = compileDSL(`${tensorId} = Tensor(4)\nY = relu(${tensorId})\n`);
      const sweep = estimateInputReuseSweep(resolved, {
        tensorId: "Y", region: fromBox(box([0, 2])),
      });
      const original = sweep.frames[0].surfaces.backward!;
      expect(Object.getPrototypeOf(original)).toBeNull();
      const transferred = structuredClone(sweep).frames[0].surfaces.backward!;
      expect(Object.keys(transferred)).toContain(tensorId);
      expect(reuseReachAt(transferred, tensorId)?.region.boxes).toEqual([box([0, 2])]);
      expect(reuseReachAt(transferred, tensorId)?.shared?.boxes).toEqual([box([0, 2])]);
    }
  );

  it("does not mistake inherited names for reached tensors after cloning", () => {
    const { resolved } = compileDSL(GEMM);
    const sweep = structuredClone(estimateInputReuseSweep(resolved, {
      tensorId: "C", region: fromBox(box([0, 2], [0, 2])),
    }));
    for (const tensorId of ["__proto__", "constructor", "toString"])
      expect(reuseReachAt(sweep.frames[0].surfaces.backward, tensorId)).toBeUndefined();
  });

  it("retains the real probes and shared input regions for playback", () => {
    const { resolved } = compileDSL(GEMM);
    const sweep = estimateInputReuseSweep(resolved, {
      tensorId: "C",
      region: fromBox(box([0, 2], [0, 2])),
    });

    expect(sweep.frames.map((frame) => frame.box)).toEqual([
      box([0, 2], [0, 2]),
      box([0, 2], [2, 4]),
      box([2, 4], [0, 2]),
      box([2, 4], [2, 4]),
    ]);
    const met = (frame: (typeof sweep.frames)[number]) =>
      Object.entries(frame.surfaces.backward!)
        .filter(([tensorId, reach]) => reach.shared && !resolved.tensors[tensorId].producer)
        .map(([tensorId]) => tensorId);
    expect(sweep.frames.map((frame) => [frame.weight, met(frame)])).toEqual([
      [1, ["A", "B"]],
      [1, ["A"]],
      [1, ["B"]],
      [1, []],
    ]);
    expect(sweep.frames[1].surfaces.backward!.A.shared!.boxes).toEqual([box([0, 2], [0, 8])]);
  });

  /* The relations are what the canvas paints, so an unasked one has to be
     absent rather than empty - the two read differently on a card. A tile on
     the intermediate has all three: it reads A and B, it feeds E, and it is
     combined with W at the second contraction. */
  it("traces only the relations it was asked for", () => {
    const { resolved } = compileDSL(CHAIN);
    const anchor = { tensorId: "C", region: fromBox(box([0, 2], [0, 2])) };

    const plain = estimateInputReuseSweep(resolved, anchor);
    expect(plain.surfaces).toEqual(["backward"]);
    expect(plain.frames[0].surfaces.forward).toBeUndefined();
    expect(plain.frames[0].surfaces.entangled).toBeUndefined();

    const all = estimateInputReuseSweep(resolved, anchor, {
      surfaces: ["backward", "forward", "entangled"],
    });
    expect(all.surfaces).toEqual(["backward", "forward", "entangled"]);
    expect(Object.keys(all.frames[0].surfaces.forward!)).toContain("E");
    expect(Object.keys(all.frames[0].surfaces.entangled!)).toEqual(["W"]);
    // The estimate is the same work either way; only the paint data grew.
    expect(all.estimates).toEqual(plain.estimates);
  });

  /* The backward walk happens either way, because the estimate is made of it.
     Reporting it per tensor is paint, and paint nobody asked for is not sent. */
  it("still estimates with nothing painted", () => {
    const { resolved } = compileDSL(CHAIN);
    const anchor = { tensorId: "C", region: fromBox(box([0, 2], [0, 2])) };
    const bare = estimateInputReuseSweep(resolved, anchor, { surfaces: [] });

    expect(bare.surfaces).toEqual([]);
    expect(bare.frames.every((frame) => Object.keys(frame.surfaces).length === 0)).toBe(true);
    expect(bare.estimates).toEqual(estimateInputReuseSweep(resolved, anchor).estimates);
  });

  it("is exact when every tile fits under the sample cap", () => {
    const { resolved } = compileDSL("X = Tensor(4, dtype=fp32)\nY = identity(X)\n");
    const result = estimateInputReuse(resolved, {
      tensorId: "Y",
      region: fromBox(box([0, 2])),
    });

    expect(result).toEqual([
      {
        tensorId: "X",
        probes: 2,
        totalTiles: 2,
        estimatedTiles: 1,
        exhaustive: true,
        meanSharedFraction: 1,
        geometryExact: true,
        reasons: [],
        // The tile below the anchor is out of bounds and is not resized to fit.
        neighbors: [{ axis: 0, delta: 1, sharedFraction: 0, exact: true, reasons: [] }],
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
      { axis: 0, delta: 1, sharedFraction: 0, exact: true, reasons: [] },
      { axis: 1, delta: 1, sharedFraction: 1, exact: true, reasons: [] },
    ]);
    expect(b.neighbors).toEqual([
      { axis: 0, delta: 1, sharedFraction: 1, exact: true, reasons: [] },
      { axis: 1, delta: 1, sharedFraction: 0, exact: true, reasons: [] },
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

  it("treats a sampled result as an estimate that may undershoot", () => {
    const { resolved } = compileDSL("X = Tensor(100, dtype=fp32)\nY = identity(X)\n");
    const [sampled] = estimateInputReuse(resolved, {
      tensorId: "Y",
      region: fromBox(box([0, 2])),
    }, { sampleCap: 12, seed: 0 });

    // Only tile zero overlaps. Seed zero samples another tile from the first
    // four-tile stratum, proving that sampling is not an upper bound.
    expect(sampled.exhaustive).toBe(false);
    expect(sampled.geometryExact).toBe(true);
    expect(sampled.estimatedTiles).toBe(0);
  });

  it("rejects a sample cap that cannot describe a sweep", () => {
    const { resolved } = compileDSL("X = Tensor(4, dtype=fp32)\nY = identity(X)\n");
    const root = { tensorId: "Y", region: fromBox(box([0, 2])) };
    expect(() => estimateInputReuse(resolved, root, { sampleCap: 0 })).toThrow(/sample cap/);
  });
});

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
    expect(a).toEqual({
      tensorId: "A",
      selectedTiles: 2,
      contributingTiles: 2,
      summedDemandBytes: 2 * 2 * 8 * 2,
      distinctDemandBytes: 2 * 8 * 2,
      duplicateDemandBytes: 2 * 8 * 2,
      geometryExact: true,
      reasons: [],
    });
    // The two cones demand disjoint columns of B, so none of it is duplicate.
    expect([b.tensorId, b.duplicateDemandBytes, b.summedDemandBytes]).toEqual([
      "B",
      0,
      2 * 8 * 2 * 2,
    ]);
  });

  it("counts only tiles whose cone reaches a given input", () => {
    const { resolved } = compileDSL(`A = Tensor(2, dtype=fp16)
B = Tensor(2, dtype=fp16)
C = concat(A, B, axis=0)
`);
    const rows = inputSharing(resolved, [
      coneOf(resolved, "C", fromBox(box([0, 2]))),
      coneOf(resolved, "C", fromBox(box([2, 4]))),
    ]);

    expect(rows.map((row) => [row.tensorId, row.contributingTiles, row.selectedTiles]))
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
    expect(row.geometryExact).toBe(false);
    expect(row.summedDemandBytes).toBe(6 * 4);
    expect(row.distinctDemandBytes).toBe(5 * 4);
    expect(row.duplicateDemandBytes).toBe(1 * 4);
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
