import { describe, expect, it } from "vitest";
import { hasUnitStride, stepCoordinate } from "../../src/core/coordinates";
import { tileCoord, tileFamily, tileOrdinal } from "../../src/core/plan/tile-family";

describe("mixed-radix movement", () => {
  it("carries and borrows across rows and batches only after the faster axes finish", () => {
    const grid = [2, 3, 4];
    expect(stepCoordinate([0, 1, 3], grid, 2, 1, true)).toEqual([0, 2, 0]);
    expect(stepCoordinate([0, 2, 3], grid, 2, 1, true)).toEqual([1, 0, 0]);
    expect(stepCoordinate([1, 0, 0], grid, 2, -1, true)).toEqual([0, 2, 3]);
    expect(stepCoordinate([0, 1, 2], grid, 2, 8, true)).toEqual([1, 0, 2]);
  });

  it("clamps at the tensor ends rather than cycling the whole tensor", () => {
    expect(stepCoordinate([0, 0], [3, 4], 1, -8, true)).toEqual([0, 0]);
    expect(stepCoordinate([2, 3], [3, 4], 1, 8, true)).toEqual([2, 3]);
    expect(stepCoordinate([0, 1], [3, 4], 1, -8, true)).toEqual([0, 0]);
  });

  it("isolates non-contiguous axes and preserves the coordinates of other axes", () => {
    expect(stepCoordinate([0, 2, 1], [2, 3, 4], 1, 1, false)).toEqual([0, 2, 1]);
    expect(stepCoordinate([0, 1, 3], [2, 3, 4], 1, -1, false)).toEqual([0, 0, 3]);
    expect(hasUnitStride([2, 3, 4], 1)).toBe(false);
    expect(hasUnitStride([2, 3, 4], 2)).toBe(true);
    expect(hasUnitStride([2, 3, 1], 1)).toBe(true);
    expect(stepCoordinate([0, 2, 0], [2, 3, 1], 1, 1, true)).toEqual([1, 0, 0]);
  });

  it("agrees with the tile-family listing used by replay", () => {
    const f = tileFamily("X", [3, 5, 7], [1, 2, 3]);
    for (let i = 0; i < f.count; i++) {
      const coord = tileCoord(f, i);
      for (const delta of [-8, -1, 0, 1, 8]) {
        const next = stepCoordinate(coord, f.grid, 2, delta, true);
        expect(tileOrdinal(f, next)).toBe(Math.max(0, Math.min(f.count - 1, i + delta)));
      }
    }
  });

  it("keeps adjacent steps exact past the safe total-element range", () => {
    const grid = [1_000_000_000, 1_000_000_000, 1_000_000_000];
    const origin = [123_456_789, 987_654_321, 999_999_999];
    const next = stepCoordinate(origin, grid, 2, 1, true);
    expect(next).toEqual([123_456_789, 987_654_322, 0]);
    expect(stepCoordinate(next, grid, 2, -1, true)).toEqual(origin);
  });

  it("has no scalar movement and rejects invalid radices or fractional steps", () => {
    expect(stepCoordinate([], [], -1, 1, true)).toEqual([]);
    expect(() => stepCoordinate([0], [0], 0, 1, true)).toThrow(RangeError);
    expect(() => stepCoordinate([0], [2], 0, 0.5, true)).toThrow(RangeError);
  });
});
