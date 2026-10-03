import { describe, expect, it } from "vitest";
import { sharedDemand } from "../../src/core/demand";
import { box } from "../../src/core/region";
import { compileDSL } from "../../src/parse/compiler";
import { demandDetail, demandSummary } from "../../src/view/demand";
import { FIGURE_MARK } from "../../src/view/format";
import { axisName } from "../../src/view/tensor/shape-label";

describe("a shared-demand row in words", () => {
  const { resolved } = compileDSL("X = Tensor(8, dtype=fp32)\nY = relu(X)\n");
  const X = resolved.tensors.X;
  const region = (lo: number, hi: number, exact = true) =>
    ({ boxes: [box([lo, hi])], exact, reasons: exact ? [] : ["widened"] });

  it("states duplication and duplicate bytes, or that there are none", () => {
    expect(demandSummary(sharedDemand(X, [region(0, 4), region(2, 6)]))).toBe("1.33× demand · 8 B duplicate");
    expect(demandSummary(sharedDemand(X, [region(0, 4), region(4, 8)]))).toBe("no duplicate demand");
    expect(demandDetail(sharedDemand(X, [region(0, 4), region(2, 6)]), "2 of 2 tiles"))
      .toBe("2 of 2 tiles demand 32 B in total; 24 B distinct");
  });

  it("marks a widened row as every figure is marked", () => {
    const widened = demandSummary(sharedDemand(X, [region(0, 4, false), region(2, 6)]));
    expect(widened).toBe(`${FIGURE_MARK.approximate}1.33× demand · ${FIGURE_MARK.upper}8 B duplicate`);
  });
});

describe("an axis's name", () => {
  it("is the source's own name, else its position", () => {
    expect(axisName({ axisNames: ["batch", undefined] }, 0)).toBe("batch");
    expect(axisName({ axisNames: ["batch", undefined] }, 1)).toBe("ax1");
    expect(axisName({}, 2)).toBe("ax2");
  });
});
