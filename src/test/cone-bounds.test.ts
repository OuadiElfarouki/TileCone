import { describe, expect, it } from "vitest";
import { compileDSL } from "../parse/compiler";
import { computeMetrics } from "../core/metrics";
import { mergeProps, propagateBackward, PropResult } from "../core/propagate";
import { box, fromBox } from "../core/region";
import { coneBounds } from "../ui/inspector-analysis";
import type { ResolvedGraph } from "../core/graph";

const gemm = () =>
  compileDSL(`A = Tensor(128, 128, dtype=fp32)
B = Tensor(128, 128, dtype=fp32)
C = matmul(A, B)
`).resolved;

const tileOf = (resolved: ResolvedGraph, r: [number, number], c: [number, number]) =>
  propagateBackward(resolved, { tensorId: "C", region: fromBox(box(r, c)) });

/** The fused/unfused range over a set of per-tile cones, as the panel forms it. */
function boundsOf(resolved: ResolvedGraph, tiles: PropResult[]) {
  const merged = computeMetrics(resolved, mergeProps(tiles)!);
  const b = coneBounds(resolved, merged, tiles);
  return {
    fused: b.fused.flops / b.fused.bytes,
    unfused: b.unfused!.flops / b.unfused!.bytes,
  };
}

describe("cone bounds", () => {
  it("separates adjacent tiles that share an operand band", () => {
    const resolved = gemm();
    const { fused, unfused } = boundsOf(resolved, [
      tileOf(resolved, [0, 16], [0, 16]),
      tileOf(resolved, [0, 16], [16, 32]),
    ]);

    // Both tiles read the same A[0:16, :] band. Sharing it once is the whole
    // difference between the two ends.
    expect(fused).toBeCloseTo(4.923, 3);
    expect(unfused).toBeCloseTo(3.765, 3);
    expect(fused).toBeGreaterThan(unfused);
  });

  it("collapses when the tiles share nothing", () => {
    const resolved = gemm();
    const { fused, unfused } = boundsOf(resolved, [
      tileOf(resolved, [0, 16], [0, 16]),
      tileOf(resolved, [64, 80], [64, 80]),
    ]);

    // Disjoint rows and columns, so there is no band to share and the range
    // has nothing to express.
    expect(fused).toBeCloseTo(unfused, 9);
    expect(fused).toBeCloseTo(3.765, 3);
  });

  it("puts one tile's own two readings at the two ends", () => {
    const resolved = compileDSL(`X = Tensor(64, 64, dtype=fp32)
Y = relu(X)
Z = relu(Y)
`).resolved;
    const tile = propagateBackward(resolved, {
      tensorId: "Z",
      region: fromBox(box([0, 16], [0, 16])),
    });
    const merged = computeMetrics(resolved, tile);
    const b = coneBounds(resolved, merged, [tile]);

    // With a single tile there is nothing to share, so the range is exactly
    // the op-fusion range that `computeMetrics` reports for that one cone.
    expect(b.fused.flops / b.fused.bytes).toBeCloseTo(merged.fusedIntensity, 9);
    expect(b.unfused!.flops / b.unfused!.bytes).toBeCloseTo(merged.unfusedIntensity, 9);
  });

  it("charges overlapping tiles twice at the worst end and once at the best", () => {
    const resolved = gemm();
    const tiles = [tileOf(resolved, [0, 16], [0, 16]), tileOf(resolved, [0, 16], [8, 24])];
    const merged = computeMetrics(resolved, mergeProps(tiles)!);
    const b = coneBounds(resolved, merged, tiles);

    // The shared output columns are real work done twice if the tiles are
    // computed as separate jobs, and once if they are merged.
    expect(b.unfused!.flops).toBeGreaterThan(b.fused.flops);
    expect(b.fused.flops).toBe(merged.flops);
  });

  it("reports no unfused bound when per-tile cones were not traced", () => {
    const resolved = gemm();
    const merged = computeMetrics(resolved, tileOf(resolved, [0, 16], [0, 16]));

    // Past the attribution cap the merged cone cannot be taken apart again.
    expect(coneBounds(resolved, merged, null).unfused).toBeNull();
  });
});
