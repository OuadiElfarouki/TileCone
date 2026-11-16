import { addFigures, figure, type Figure } from "../metrics";
import type { Work } from "./interfaces";

export type TimeBoundRates = {
  flopsPerSecond?: number;
  bytesPerSecond?: number;
};

export type PlanTimeBound = {
  seconds: Figure;
  scope: "declared-work-and-traffic";
  assumptions: {
    countedArithmeticExecuted: true;
    countedDistinctReadsAndWritesCrossBandwidthBoundary: true;
    ratesAreUpperLimits: true;
  };
};

/**
 * Bound the declared work and logical traffic under the returned assumptions.
 * Logical distinct reads may include cache reads; this is not an unconditional
 * physical runtime bound. Missing either rate returns null. Any supplied rate
 * must be finite and positive, or a RangeError is thrown. Upper, approximate,
 * or unknown input figures cannot establish a lower bound, so seconds is unknown.
 */
export function planTimeBound(work: Work, rates?: TimeBoundRates): PlanTimeBound | null {
  for (const rate of [rates?.flopsPerSecond, rates?.bytesPerSecond]) {
    if (rate !== undefined && (!Number.isFinite(rate) || rate <= 0))
      throw new RangeError("Time-bound rates must be finite and positive");
  }
  if (rates?.flopsPerSecond === undefined || rates.bytesPerSecond === undefined) return null;

  const transferred = addFigures(work.readDistinct, work.written);
  const inputs = [work.flops, transferred];
  const reasons = inputs.flatMap((f) => f.reasons);
  const admissible = inputs.every((f) => f.status === "exact" || f.status === "lower");
  const seconds = admissible
    ? Math.max(work.flops.value! / rates.flopsPerSecond,
        transferred.value! / rates.bytesPerSecond)
    : NaN;
  return {
    seconds: Number.isFinite(seconds)
      ? figure(seconds, "lower", reasons)
      : figure(0, "unknown", [...reasons, "cannot establish a time bound from these figures"]),
    scope: "declared-work-and-traffic",
    assumptions: {
      countedArithmeticExecuted: true,
      countedDistinctReadsAndWritesCrossBandwidthBoundary: true,
      ratesAreUpperLimits: true,
    },
  };
}
