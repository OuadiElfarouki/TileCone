/** An axis has stride one when every faster extent is one (row-major storage). */
export const hasUnitStride = (shape: readonly number[], axis: number): boolean =>
  axis >= 0 && axis < shape.length && shape.slice(axis + 1).every((extent) => extent === 1);

/**
 * Step a coordinate in a mixed-radix domain. Without carry only `axis` is
 * active; with carry its slower axes participate too. The outer ends clamp.
 * BigInt keeps a single step exact even when the domain exceeds 2^53 cells.
 */
export function stepCoordinate(
  coord: readonly number[],
  grid: readonly number[],
  axis: number,
  delta: number,
  carry: boolean
): number[] {
  if (axis < 0 || axis >= coord.length) return [...coord];
  if (!Number.isSafeInteger(axis) || !Number.isSafeInteger(delta) || grid.length !== coord.length ||
      grid.some((size, i) => !Number.isSafeInteger(size) || size < 1 ||
        !Number.isSafeInteger(coord[i]) || coord[i] < 0 || coord[i] >= size))
    throw new RangeError("coordinate steps require integer coordinates within positive radices");
  const active = grid.map((_, i) => Number(i === axis || (carry && i < axis)));
  const radices = grid.map((size, i) => BigInt(1 + active[i] * (size - 1)));
  const ordinal = coord.reduce((n, digit, i) => n * radices[i] + BigInt(active[i] * digit), 0n);
  const last = radices.reduce((n, size) => n * size, 1n) - 1n;
  const requested = ordinal + BigInt(delta);
  let at = requested < 0n ? 0n : requested > last ? last : requested;
  const next = [...coord];
  for (let i = grid.length - 1; i >= 0; i--) {
    const digit = Number(at % radices[i]);
    at /= radices[i];
    next[i] += active[i] * (digit - coord[i]);
  }
  return next;
}
