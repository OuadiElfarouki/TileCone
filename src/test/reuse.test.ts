import { describe, expect, it } from "vitest";
import { estimateInputReuse, sampledTileIndices } from "../core/reuse";
import { box, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";

describe("reuse estimation", () => {
  it("is exact when every tile fits under the sample cap", () => {
    const { resolved } = compileDSL("input X [4] f32\nY = identity(X)\n");
    const result = estimateInputReuse(resolved, {
      tensorId: "Y",
      region: fromBox(box([0, 2])),
    });

    expect(result).toEqual([
      { tensorId: "X", touches: 1, probes: 2, totalTiles: 2, estimatedTiles: 1 },
    ]);
  });

  it("uses reproducible, non-repeating stratified samples", () => {
    const first = sampledTileIndices(10_000, 48, 1234);
    expect(sampledTileIndices(10_000, 48, 1234)).toEqual(first);
    expect(new Set(first).size).toBe(48);
    expect(sampledTileIndices(10_000, 48, 5678)).not.toEqual(first);
  });
});
