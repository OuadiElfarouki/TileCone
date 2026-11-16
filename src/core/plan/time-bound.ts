import { addFigures, figure, type Figure } from "../metrics";
import type { Work } from "./interfaces";
import { PlanError } from "./tile-family";

export type TimeBoundRates = {
  flopsPerSecond?: number;
  bytesPerSecond?: number;
};

export type PlanTimeBound = {
  seconds: Figure;
  terms: {
    compute: Figure | null;
    memory: Figure | null;
    memorySummed: { seconds: Figure; scenario: "summed-logical-reads-and-writes" } | null;
  };
  completeness: "complete" | "partial";
  limiting: "compute" | "memory" | "tied" | "undetermined";
  scope: "declared-work-and-traffic";
  assumptions: {
    countedArithmeticExecuted: boolean;
    countedDistinctReadsAndWritesCrossBandwidthBoundary: boolean;
    ratesAreUpperLimits: true;
  };
};

/**
 * Conditional lower bound under the declared model, not a physical runtime
 * guarantee: counted arithmetic must execute, and logical distinct reads and
 * writes must cross the specified bandwidth boundary. Logical reads can include
 * cache reads. The assumption flags indicate which channels were requested.
 *
 * Each supplied rate must be finite and positive (otherwise PLAN_BUDGET is
 * thrown). No supplied rates returns null. An absent channel has a null term;
 * an upper, approximate, unknown, or non-finite input gives an unknown term.
 * `seconds` is the maximum available lower term, or unknown if none is known.
 * Completeness requires both terms to be known. Limiting identifies the largest
 * modeled lower term only when complete; it does not prove a physical bottleneck,
 * particularly when the input figures themselves are lower bounds.
 *
 * memorySummed reports a separate summed logical reads plus writes scenario. Its
 * figure retains the input arithmetic's status; it is neither a physical runtime
 * bracket nor an input to seconds. It makes no boundary-crossing claim.
 */
export function planTimeBound(work: Work, rates?: TimeBoundRates): PlanTimeBound | null {
  for (const rate of [rates?.flopsPerSecond, rates?.bytesPerSecond]) {
    if (rate !== undefined && (!Number.isFinite(rate) || rate <= 0))
      throw new PlanError("PLAN_BUDGET", "Time-bound rates must be finite and positive");
  }
  const computeRate = rates?.flopsPerSecond;
  const memoryRate = rates?.bytesPerSecond;
  if (computeRate === undefined && memoryRate === undefined) return null;

  const lowerTerm = (input: Figure, rate: number): Figure => {
    const value = input.value === null ? NaN : input.value / rate;
    return (input.status === "exact" || input.status === "lower") && Number.isFinite(value)
      ? figure(value, "lower", input.reasons)
      : figure(0, "unknown", [...input.reasons, "cannot establish a time bound from this figure"]);
  };
  const compute = computeRate === undefined ? null : lowerTerm(work.flops, computeRate);
  const memory = memoryRate === undefined ? null : lowerTerm(addFigures(work.readDistinct, work.written), memoryRate);
  const summed = memoryRate === undefined ? null : addFigures(work.read, work.written);
  const summedValue = summed?.value === null || summed === null ? NaN : summed.value / memoryRate!;
  const memorySummed = summed === null ? null : {
    seconds: Number.isFinite(summedValue)
      ? figure(summedValue, summed.status, summed.reasons)
      : figure(0, "unknown", [...summed.reasons, "cannot calculate summed traffic scenario"]),
    scenario: "summed-logical-reads-and-writes" as const,
  };
  const known = [compute, memory].filter((term): term is Figure & { value: number } => term !== null && term.value !== null);
  const complete = compute?.value != null && memory?.value != null;
  const seconds = known.length
    ? figure(Math.max(...known.map(term => term.value)), "lower", [compute, memory].flatMap(term => term?.reasons ?? []))
    : figure(0, "unknown", [compute, memory].flatMap(term => term?.reasons ?? []));
  const limiting = !complete ? "undetermined"
    : compute.value > memory.value ? "compute"
    : memory.value > compute.value ? "memory" : "tied";
  return {
    seconds,
    terms: { compute, memory, memorySummed },
    completeness: complete ? "complete" : "partial",
    limiting,
    scope: "declared-work-and-traffic",
    assumptions: {
      countedArithmeticExecuted: computeRate !== undefined,
      countedDistinctReadsAndWritesCrossBandwidthBoundary: memoryRate !== undefined,
      ratesAreUpperLimits: true,
    },
  };
}
