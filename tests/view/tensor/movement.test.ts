import { describe, expect, it } from "vitest";
import { box, type Box } from "../../../src/core/region";
import { nudgeBoxes } from "../../../src/view/tensor/movement";

describe("row-major tile origins", () => {
  it("moves an element from the end of a plane into the next batch", () => {
    const shape = [2, 3, 4], units = [1, 1, 1];
    const start = [box([0, 1], [2, 3], [3, 4])];
    const next = nudgeBoxes(start, shape, units, 2, 1);
    expect(next).toEqual([box([1, 2], [0, 1], [0, 1])]);
    expect(nudgeBoxes(next, shape, units, 2, -1)).toEqual(start);
  });

  it("keeps extents intact at shortened tails and includes a flush-edge stop", () => {
    const shape = [4, 5], units = [2, 2];
    const start = [box([0, 2], [2, 4])];
    const edge = nudgeBoxes(start, shape, units, 1, 1);
    expect(edge).toEqual([box([0, 2], [3, 5])]);
    expect(nudgeBoxes(edge, shape, units, 1, 1)).toEqual([box([2, 4], [0, 2])]);
    expect(nudgeBoxes([box([2, 4], [0, 2])], shape, units, 1, -1)).toEqual(edge);
  });

  it("aligns an off-grid fast axis first and preserves off-grid slower axes until carry", () => {
    const start = [box([1, 3], [1, 3])];
    const shape = [6, 6], units = [2, 2];
    expect(nudgeBoxes(start, shape, units, 1, 1, 8)).toEqual([box([1, 3], [2, 4])]);
    expect(nudgeBoxes(start, shape, units, 1, -1, 8)).toEqual([box([1, 3], [0, 2])]);
    expect(nudgeBoxes([box([1, 3], [4, 6])], shape, units, 1, 1)).toEqual([box([2, 4], [0, 2])]);
  });

  it("translates a multi-part frame rigidly and does not spend history at the outer end", () => {
    const parts = [box([0, 1], [2, 3]), box([1, 2], [3, 4])];
    const next = nudgeBoxes(parts, [4, 4], [1, 1], 1, 1);
    expect(next).toEqual([box([1, 2], [0, 1]), box([2, 3], [1, 2])]);
    const last = [box([2, 4], [2, 4])];
    expect(nudgeBoxes(last, [4, 4], [1, 1], 1, 1)).toBe(last);
  });

  it("does not wrap a box that spans every valid placement", () => {
    const full = [box([0, 3], [0, 4])];
    expect(nudgeBoxes(full, [3, 4], [1, 1], 1, 1)).toBe(full);
    expect(nudgeBoxes([], [3, 4], [1, 1], 1, 1)).toEqual([]);
  });

  it("matches an independently enumerated valid-origin sequence across varied ranks and extents", () => {
    let seed = 42;
    const random = (n: number) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
    const product = (axes: number[][]): number[][] => axes.reduce<number[][]>((prefixes, stops) =>
      prefixes.flatMap((prefix) => stops.map((stop) => [...prefix, stop])), [[]]);
    for (let trial = 0; trial < 500; trial++) {
      const shape = Array.from({ length: 1 + random(4) }, () => 1 + random(5));
      const extents = shape.map((n) => 1 + random(n));
      const origins = shape.map((n, i) => random(n - extents[i] + 1));
      const units = shape.map(() => 1 + random(3));
      const start: Box = origins.map((lo, i) => ({ lo, hi: lo + extents[i] }));
      const stops = shape.map((n, i) => {
        const limit = n - extents[i];
        const points = [origins[i], limit];
        for (let p = 0; p < limit; p += units[i]) points.push(p);
        return [...new Set(points)].sort((a, b) => a - b);
      });
      const sequence = product(stops);
      const index = sequence.findIndex((point) => point.every((lo, i) => lo === origins[i]));
      const last = shape.length - 1;
      const offGrid = origins[last] % units[last] !== 0 && origins[last] !== shape[last] - extents[last];
      for (const sign of [-1, 1] as const) {
        for (const multiplier of [1, 8]) {
          const at = Math.max(0, Math.min(sequence.length - 1, index + sign * (offGrid ? 1 : multiplier)));
          const expected = sequence[at].map((lo, i) => ({ lo, hi: lo + extents[i] }));
          expect(nudgeBoxes([start], shape, units, shape.length - 1, sign, multiplier)).toEqual([expected]);
        }
      }
    }
  });
});
