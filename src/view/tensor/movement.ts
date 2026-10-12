import { hasUnitStride, stepCoordinate } from "../../core/coordinates";
import { boundingBox, iv, type Box } from "../../core/region";

/**
 * Move boxes as one rigid frame through valid origins, in row-major order.
 * Snapped origins are multiples of `units`, plus a flush far-edge placement.
 * An off-grid current origin is an extra stop: it stays put on untouched axes
 * and the first step aligns it in the requested direction. No tile is resized.
 */
export function nudgeBoxes(
  boxes: Box[],
  shape: readonly number[],
  units: readonly number[],
  axis: number,
  sign: -1 | 1,
  multiplier = 1
): Box[] {
  const frame = boundingBox({ boxes, exact: true, reasons: [] });
  if (!frame || axis < 0 || axis >= shape.length) return boxes;
  const origins = frame.map((interval) => interval.lo);
  const limits = frame.map((interval, i) => shape[i] - (interval.hi - interval.lo));
  const inserted = origins.map((lo, i) => Number(lo % units[i] !== 0 && lo !== limits[i]));
  const ceilSteps = (value: number, unit: number) => Number((BigInt(value) + BigInt(unit) - 1n) / BigInt(unit));
  const grid = limits.map((limit, i) => ceilSteps(limit, units[i]) + 1 + inserted[i]);
  const coord = origins.map((lo, i) => ceilSteps(lo, units[i]));
  // Align before taking a fast stride, as ordinary axis nudges do.
  const steps = sign * (1 + (multiplier - 1) * (1 - inserted[axis]));
  const next = stepCoordinate(coord, grid, axis, steps, hasUnitStride(shape, axis));
  const delta = next.map((digit, i) => {
    const regular = Math.min(limits[i], (digit - inserted[i] * Number(digit > coord[i])) * units[i]);
    const origin = regular + Number(digit === coord[i]) * (origins[i] - regular);
    return origin - origins[i];
  });
  if (delta.every((d) => d === 0)) return boxes;
  return boxes.map((box) => box.map((interval, i) => iv(interval.lo + delta[i], interval.hi + delta[i])));
}
