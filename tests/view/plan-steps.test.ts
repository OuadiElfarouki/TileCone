import { describe, expect, it } from "vitest";
import { compileDSL } from "../../src/parse/compiler";
import { tilePlan } from "../../src/core/plan/plan";
import { stagingOf, type Staged } from "../../src/core/plan/steps";
import { formatBytes } from "../../src/view/format";
import { budgetVerdict, parseBytes, parseChunk, reducedAxisNames, withChunk } from "../../src/view/plan-steps";

const names = (src: string, tensorId: string, tile: number[]) => {
  const plan = tilePlan(compileDSL(src).resolved, { [tensorId]: tile });
  return reducedAxisNames(plan, stagingOf(plan, tensorId) as Staged);
};

describe("what a reduced axis is called", () => {
  it("is the name the source gave it on an operand", () => {
    expect(names(`X = Tensor(batch=2, seq=8, emb=16)\nW = Parameter(emb=16, out=4)\nY = einsum("bse,ef->bsf", X, W)`, "Y", [1, 4, 2])).toEqual(["emb"]);
  });

  it("is the symbol that sizes it, where it has no name", () => {
    expect(names(`M = 4\nK = 8\nA = Tensor(M, K)\nB = Tensor(K, 6)\nC = matmul(A, B)`, "C", [2, 3])).toEqual(["K"]);
  });

  it("is the operation's own label where the source wrote only a number", () => {
    expect(names(`A = Tensor(4, 8)\nB = Tensor(8, 6)\nC = matmul(A, B)`, "C", [2, 3])).toEqual(["k"]);
  });

  it("falls back to labels for axes whose preferred names are the same", () => {
    // Both reduced axes are sized by N, so the symbol names neither.
    expect(names(`N = 4\nA = Tensor(3, N, N)\nB = Tensor(N, N, 5)\nC = einsum("ijk,jkl->il", A, B)`, "C", [3, 5])).toEqual(["j", "k"]);
  });
});

describe("a chunk as typed", () => {
  it("is a whole number within the axis, or the whole axis", () => {
    expect(parseChunk("32", 512)).toBe(32);
    expect(parseChunk(" 1 ", 512)).toBe(1);
    for (const whole of ["", "all", "ALL", "whole", "512"]) expect(parseChunk(whole, 512)).toBe("whole");
    for (const bad of ["0", "513", "-4", "3.5", "k", "32 64"]) expect(parseChunk(bad, 512)).toBeNull();
  });

  it("sets or clears one axis of a declaration and keeps the rest", () => {
    expect(withChunk(undefined, "k", 32)).toEqual({ chunk: { k: 32 } });
    expect(withChunk({ chunk: { k: 32 } }, "k", "whole")).toBeNull();
    expect(withChunk({ chunk: { j: 2, k: 32 } }, "k", "whole")).toEqual({ chunk: { j: 2 } });
    expect(withChunk({ chunk: { k: 32 }, accumulator: "f16", order: ["k"] }, "k", "whole")).toEqual({ accumulator: "f16", order: ["k"] });
    expect(withChunk({ accumulator: "f16" }, "k", 8)).toEqual({ chunk: { k: 8 }, accumulator: "f16" });
  });
});

describe("a byte count as typed", () => {
  it("reads binary units and bare counts", () => {
    expect(parseBytes("99 KB")).toBe(99 * 1024);
    expect(parseBytes("99KiB")).toBe(99 * 1024);
    expect(parseBytes("1.5 mb")).toBe(1.5 * 1024 * 1024);
    expect(parseBytes("4096")).toBe(4096);
    expect(parseBytes(" 4096 B ")).toBe(4096);
    for (const bad of ["", "0", "-1 KB", "99 XB", "KB", "1e3", "0.2"]) expect(parseBytes(bad)).toBeNull();
  });

  it("reads back what the panel prints", () => {
    for (const n of [512, 4096, 16384, 99 * 1024, 3 << 20]) expect(parseBytes(formatBytes(n))).toBe(n);
  });
});

describe("a payload against the budget", () => {
  it("says within or over only when the figure can establish it", () => {
    const exact = (value: number) => ({ value, status: "exact" });
    expect(budgetVerdict(exact(100), undefined)).toBeNull();
    expect(budgetVerdict(exact(100), 100)).toBe("within");
    expect(budgetVerdict(exact(101), 100)).toBe("over");
    // An upper bound under the limit proves within; over the limit it proves nothing.
    expect(budgetVerdict({ value: 90, status: "upper" }, 100)).toBe("within");
    expect(budgetVerdict({ value: 120, status: "upper" }, 100)).toBe("unknown");
    expect(budgetVerdict({ value: 120, status: "lower" }, 100)).toBe("over");
    expect(budgetVerdict({ value: 90, status: "lower" }, 100)).toBe("unknown");
    expect(budgetVerdict({ value: 90, status: "approximate" }, 100)).toBe("unknown");
    expect(budgetVerdict({ value: null, status: "unknown" }, 100)).toBe("unknown");
  });
});
