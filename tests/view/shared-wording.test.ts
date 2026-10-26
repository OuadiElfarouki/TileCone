import { describe, expect, it } from "vitest";
import { sharedDemand } from "../../src/core/demand";
import { executeQuery } from "../../src/core/executor";
import { operandReads } from "../../src/core/multiplicity";
import { box, fromBox, type Region } from "../../src/core/region";
import { compileDSL } from "../../src/parse/compiler";
import { demandDetail, demandSummary } from "../../src/view/demand";
import { readOnceLine, readsDetail, readsOperation, readsSummary } from "../../src/view/multiplicity";
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

describe("a read-multiplicity row in words", () => {
  const readsOf = (dsl: string, tensorId: string, region: Region, read: string) => {
    const { resolved } = compileDSL(dsl);
    const back = executeQuery(resolved, { tensorId, region, direction: "backward" }).backward!;
    return operandReads(resolved, back.tensors).find((row) => row.tensorId === read)!;
  };
  const GEMM = "A = Tensor(4, 8, dtype=fp16)\nB = Tensor(8, 6, dtype=fp16)\nC = matmul(A, B)\n";

  it("names a uniform count, a mixed one, and a single use", () => {
    const tile = readsOf(GEMM, "C", fromBox(box([0, 2], [0, 3])), "A");
    expect(readsSummary(tile)).toBe("3 uses per element · 32 B");
    expect(readsDetail(tile)).toBe("48 uses of 16 elements, 3.00 per element on average."
      + " A use is one output element of the operation depending on an operand element; it is not a memory load.");
    const corner = { boxes: [box([0, 1], [0, 3]), box([1, 2], [0, 1])], exact: true, reasons: [] };
    expect(readsSummary(readsOf(GEMM, "C", corner, "A"))).toBe("16 B at up to 3 uses per element · 16 B used once");
    expect(readsSummary(readsOf(GEMM, "C", fromBox(box([1, 2], [2, 3])), "A"))).toBe("used once · 16 B");
  });

  it("gives the range when nothing is read once but counts differ", () => {
    // The shared corner of the two bands is read by 4 outputs, the rest by 2.
    const shared = readsOf("A = Tensor(4, 4)\nC = matmul(A, A)\n", "C", fromBox(box([0, 2], [0, 2])), "A");
    expect(readsSummary(shared)).toBe("2–4 uses per element · 48 B");
    expect(readsOperation(shared, (id) => id)).toBe("matmul → C");
  });

  it("names single-use operands by what they become, and their operations in the detail", () => {
    const { resolved } = compileDSL("X = Tensor(4, 8)\nY = permute_dims(X, [1, 0])\nZ = relu(Y)\n");
    const back = executeQuery(resolved, { tensorId: "Z", region: fromBox(box([0, 2], [0, 2])), direction: "backward" }).backward!;
    const line = readOnceLine(operandReads(resolved, back.tensors), (id) => id);
    expect(line.text).toBe("X → Y, Y → Z");
    expect(line.detail).toBe("X via permute_dims → Y\nY via relu → Z");
  });

  it("marks a count over an unknown index as every figure is marked", () => {
    const dsl = "W = Tensor(10, 4)\nIdx = Tensor(3, dtype=int32)\nE = take(W, Idx, axis=0)\n";
    const row = readsOf(dsl, "E", fromBox(box([0, 2], [0, 4])), "W");
    expect(readsSummary(row)).toBe(`${FIGURE_MARK.upper}2 uses per element · ${FIGURE_MARK.upper}160 B`);
    expect(readsDetail(row)).toBe(
      `${FIGURE_MARK.upper}80 uses of ${FIGURE_MARK.upper}40 elements, ${FIGURE_MARK.approximate}2.00 per element on average.` +
        " A use is one output element of the operation depending on an operand element; it is not a memory load."
    );
  });
});

describe("an axis's name", () => {
  it("is the source's own name, else its position", () => {
    expect(axisName({ axisNames: ["batch", undefined] }, 0)).toBe("batch");
    expect(axisName({ axisNames: ["batch", undefined] }, 1)).toBe("ax1");
    expect(axisName({}, 2)).toBe("ax2");
  });
});
