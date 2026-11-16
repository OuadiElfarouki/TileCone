import { describe, expect, it } from "vitest";
import { figure, type FigureStatus } from "../../../src/core/metrics";
import { planReport } from "../../../src/core/plan/interfaces";
import { tilePlan } from "../../../src/core/plan/plan";
import { PlanError } from "../../../src/core/plan/tile-family";
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
  it("reports both terms and the limiting modeled lower term", () => {
    const result = planTimeBound(work(), rates)!;
    expect(result.seconds).toEqual(figure(10.4, "lower"));
    expect(result.terms.compute).toEqual(figure(4, "lower"));
    expect(result.terms.memory).toEqual(figure(10.4, "lower"));
    expect(result).toMatchObject({ completeness: "complete", limiting: "memory", scope: "declared-work-and-traffic", assumptions: {
      countedArithmeticExecuted: true, countedDistinctReadsAndWritesCrossBandwidthBoundary: true, ratesAreUpperLimits: true,
    } });
    expect(planTimeBound(work(), { flopsPerSecond: 12, bytesPerSecond: 1000 })!.limiting).toBe("compute");
    expect(planTimeBound(work(), { flopsPerSecond: 12, bytesPerSecond: 26 })!.limiting).toBe("tied");
  });

  it("accepts each rate independently and rejects invalid supplied rates", () => {
    expect(planTimeBound(work())).toBeNull();
    expect(planTimeBound(work(), {})).toBeNull();
    const compute = planTimeBound(work(), { flopsPerSecond: 12 })!;
    expect(compute).toMatchObject({ seconds: { value: 4, status: "lower" }, terms: { memory: null, memorySummed: null }, completeness: "partial", limiting: "undetermined", assumptions: { countedArithmeticExecuted: true, countedDistinctReadsAndWritesCrossBandwidthBoundary: false } });
    const memory = planTimeBound(work(), { bytesPerSecond: 10 })!;
    expect(memory).toMatchObject({ seconds: { value: 10.4, status: "lower" }, terms: { compute: null }, completeness: "partial", limiting: "undetermined", assumptions: { countedArithmeticExecuted: false, countedDistinctReadsAndWritesCrossBandwidthBoundary: true } });
    for (const invalid of [0, -1, NaN, Infinity, -Infinity]) {
      for (const invalidRates of [{ flopsPerSecond: invalid }, { ...rates, bytesPerSecond: invalid }]) {
        expect(() => planTimeBound(work(), invalidRates)).toThrow(PlanError);
        try { planTimeBound(work(), invalidRates); } catch (error) { expect(error).toMatchObject({ code: "PLAN_BUDGET" }); }
      }
    }
  });

  it("retains a known lower term when the other term is inadmissible", () => {
    const w = work();
    expect(planTimeBound({ ...w, flops: figure(24, "lower") }, rates)!.terms.compute).toEqual(figure(2, "lower"));
    for (const field of ["flops", "readDistinct", "written"] as const) {
      for (const status of ["upper", "approximate", "unknown"] as FigureStatus[]) {
        const result = planTimeBound({ ...w, [field]: figure(1, status, ["source"]) }, rates)!;
        const unknown = field === "flops" ? result.terms.compute : result.terms.memory;
        expect(unknown).toMatchObject({ value: null, status: "unknown", reasons: expect.arrayContaining(["source"]) });
        expect(result.seconds).toMatchObject({ value: field === "flops" ? 10.4 : 4, status: "lower" });
        expect(result).toMatchObject({ completeness: "partial", limiting: "undetermined" });
      }
    }
    expect(planTimeBound({ ...w, flops: figure(1, "unknown") }, { flopsPerSecond: 1 })!.seconds.status).toBe("unknown");
  });

  it("handles overflow without discarding the other term", () => {
    const w = { ...work(), readDistinct: figure(Number.MAX_SAFE_INTEGER, "exact"), written: figure(2, "exact") };
    const result = planTimeBound(w, rates)!;
    expect(result.terms.memory).toMatchObject({ value: null, status: "unknown", reasons: expect.arrayContaining(["figure exceeds safe integer range"]) });
    expect(result.seconds).toMatchObject({ value: 4, status: "lower" });
    expect(planTimeBound(work(), { flopsPerSecond: Number.MIN_VALUE })!.seconds.status).toBe("unknown");
  });

  it("reports summed traffic as a separate scenario without altering the bound or input", () => {
    const w = work();
    const before = structuredClone(w);
    const changed = { ...w, read: figure(1000, "exact") };
    const original = planTimeBound(w, rates)!;
    const result = planTimeBound(changed, rates)!;
    expect(result.terms.memorySummed).toEqual({ seconds: figure(103.2, "exact"), scenario: "summed-logical-reads-and-writes" });
    expect(result.seconds).toEqual(original.seconds);
    expect(w).toEqual(before);
  });
});
