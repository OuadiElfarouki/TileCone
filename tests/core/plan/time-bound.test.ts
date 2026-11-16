import { describe, expect, it } from "vitest";
import { figure, type FigureStatus } from "../../../src/core/metrics";
import { planReport } from "../../../src/core/plan/interfaces";
import { tilePlan } from "../../../src/core/plan/plan";
import { planTimeBound } from "../../../src/core/plan/time-bound";
import { compileDSL } from "../../../src/parse/compiler";

const work = () => {
  const graph = compileDSL(`A = Tensor(2, 3, dtype=float32)
B = Tensor(3, 4, dtype=float32)
Y = matmul(A, B)
`).resolved;
  return planReport(tilePlan(graph, { Y: [2, 4] })).total!;
};
const rates = { flopsPerSecond: 12, bytesPerSecond: 10 };

describe("plan time bound", () => {
  it("bounds a small matmul's declared work and traffic with explicit assumptions", () => {
    const w = work();
    expect(w.flops.value).toBe(48);
    expect(w.readDistinct.value).toBe(72);
    expect(w.written.value).toBe(32);
    expect(planTimeBound(w, rates)).toEqual({
      seconds: { value: 10.4, status: "lower", reasons: [] },
      scope: "declared-work-and-traffic",
      assumptions: {
        countedArithmeticExecuted: true,
        countedDistinctReadsAndWritesCrossBandwidthBoundary: true,
        ratesAreUpperLimits: true,
      },
    });
  });

  it("is absent until both rates are supplied, and rejects any invalid supplied rate", () => {
    expect(planTimeBound(work())).toBeNull();
    expect(planTimeBound(work(), { flopsPerSecond: 1 })).toBeNull();
    expect(planTimeBound(work(), { bytesPerSecond: 1 })).toBeNull();
    for (const invalid of [0, -1, NaN, Infinity, -Infinity]) {
      expect(() => planTimeBound(work(), { flopsPerSecond: invalid })).toThrow(RangeError);
      expect(() => planTimeBound(work(), { ...rates, bytesPerSecond: invalid })).toThrow(RangeError);
    }
  });

  it("uses the compute term when it dominates", () => {
    expect(planTimeBound(work(), { flopsPerSecond: 12, bytesPerSecond: 1000 })!.seconds)
      .toMatchObject({ value: 4, status: "lower" });
  });

  it("declines byte totals outside safe integer arithmetic", () => {
    const w = { ...work(), readDistinct: figure(Number.MAX_SAFE_INTEGER, "exact"), written: figure(2, "exact") };
    expect(planTimeBound(w, rates)!.seconds).toMatchObject({ value: null, status: "unknown" });
    expect(planTimeBound(w, rates)!.seconds.reasons).toContain("figure exceeds safe integer range");
  });

  it("accepts lower inputs and refuses every inadmissible source status", () => {
    const w = work();
    expect(planTimeBound({ ...w, flops: figure(24, "lower") }, rates)!.seconds.status).toBe("lower");
    for (const field of ["flops", "readDistinct", "written"] as const) {
      for (const status of ["upper", "approximate", "unknown"] as FigureStatus[]) {
        const result = planTimeBound({ ...w, [field]: figure(1, status, ["source"]) }, rates)!;
        expect(result.seconds).toMatchObject({ value: null, status: "unknown" });
        expect(result.seconds.reasons).toContain("source");
      }
    }
  });
});
