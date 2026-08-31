import { describe, expect, it } from "vitest";
import { estimateInputReuse } from "../core/reuse";
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

  it("is reproducible when the estimate is sampled", () => {
    const { resolved } = compileDSL("input X [100] f32\nY = identity(X)\n");
    const root = { tensorId: "Y", region: fromBox(box([0, 2])) };
    const first = estimateInputReuse(resolved, root, { sampleCap: 12, seed: 1234 });

    expect(estimateInputReuse(resolved, root, { sampleCap: 12, seed: 1234 })).toEqual(first);
    expect(first[0].probes).toBe(12);
    expect(first[0].totalTiles).toBe(50);
  });
});
