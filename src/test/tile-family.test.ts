import { describe, expect, it } from "vitest";
import { Box, count } from "../core/region";
import {
  isTile,
  PlanError,
  tileBox,
  tileCoord,
  tileFamily,
  tileOrdinal,
  tiles,
  tilesMeeting,
  tileVolume,
} from "../core/plan/tile-family";
import { randInt, rng } from "./harness";

const randomShape = (r: () => number, rank: number) =>
  Array.from({ length: rank }, () => randInt(r, 1, 9));
const randomTile = (r: () => number, shape: number[]) =>
  shape.map((extent) => randInt(r, 1, extent + 3)); // sometimes wider than the axis

const meets = (a: Box, b: Box) => a.every((iv, axis) => iv.lo < b[axis].hi && b[axis].lo < iv.hi);

describe("a tile family covers its tensor", () => {
  it("places every element in exactly one tile, with shortened tails", () => {
    const r = rng(11);
    for (let trial = 0; trial < 200; trial++) {
      const shape = randomShape(r, randInt(r, 0, 4));
      const f = tileFamily("T", shape, randomTile(r, shape));
      const boxes = [...tiles(f)].map((coord) => tileBox(f, coord));
      const volume = shape.reduce((n, e) => n * e, 1);

      expect(boxes.length).toBe(f.count);
      // Disjoint and complete: the volumes sum to the tensor, and so does their union.
      expect(boxes.reduce((n, b) => n + count({ boxes: [b], exact: true, reasons: [] }), 0)).toBe(volume);
      expect(count({ boxes, exact: true, reasons: [] })).toBe(volume);
      for (const b of boxes)
        b.forEach(({ lo, hi }, axis) => {
          expect(lo).toBeLessThan(hi);
          expect(hi).toBeLessThanOrEqual(shape[axis]);
        });
    }
  });

  it("shortens the last tile on an axis the extent does not divide", () => {
    const f = tileFamily("T", [10, 4], [4, 4]);
    expect(f.grid).toEqual([3, 1]);
    expect(tileBox(f, [2, 0])).toEqual([{ lo: 8, hi: 10 }, { lo: 0, hi: 4 }]);
    expect(tileVolume(f, [2, 0])).toBe(8);
  });

  it("gives one tile per axis when the tile is wider than the axis", () => {
    const f = tileFamily("T", [3, 5], [8, 8]);
    expect(f.count).toBe(1);
    expect(tileBox(f, [0, 0])).toEqual([{ lo: 0, hi: 3 }, { lo: 0, hi: 5 }]);
  });

  it("has a single empty coordinate at rank 0", () => {
    const f = tileFamily("s", [], []);
    expect(f.count).toBe(1);
    expect([...tiles(f)]).toEqual([[]]);
    expect(tileBox(f, [])).toEqual([]);
    expect([...tilesMeeting(f, [])]).toEqual([[]]);
  });

  it("enumerates row-major, and ordinals round-trip", () => {
    const f = tileFamily("T", [5, 7, 3], [2, 3, 2]);
    const all = [...tiles(f)];
    all.forEach((coord, i) => {
      expect(tileOrdinal(f, coord)).toBe(i);
      expect(tileCoord(f, i)).toEqual(coord);
      expect(isTile(f, coord)).toBe(true);
    });
    expect(isTile(f, [3, 0, 0])).toBe(false);
    expect(isTile(f, [0, 0])).toBe(false);
    expect(isTile(f, [0, 0.5, 0])).toBe(false);
  });
});

describe("the tiles a box meets", () => {
  it("are exactly the tiles sharing an element with it", () => {
    const r = rng(12);
    for (let trial = 0; trial < 300; trial++) {
      const shape = randomShape(r, randInt(r, 1, 4));
      const f = tileFamily("T", shape, randomTile(r, shape));
      const box: Box = shape.map((extent) => {
        const lo = randInt(r, 0, extent);
        return { lo, hi: randInt(r, lo + 1, extent + 1) };
      });
      const expected = [...tiles(f)].filter((coord) => meets(tileBox(f, coord), box));
      expect([...tilesMeeting(f, box)]).toEqual(expected);
    }
  });

  it("are none for an empty box", () => {
    const f = tileFamily("T", [8, 8], [2, 2]);
    expect([...tilesMeeting(f, [{ lo: 3, hi: 3 }, { lo: 0, hi: 8 }])]).toEqual([]);
  });

  it("refuses a box outside the tensor", () => {
    const f = tileFamily("T", [8, 8], [2, 2]);
    expect(() => [...tilesMeeting(f, [{ lo: 0, hi: 9 }, { lo: 0, hi: 1 }])]).toThrow(/outside/);
    expect(() => [...tilesMeeting(f, [{ lo: 0, hi: 1 }])]).toThrow(/rank/);
  });
});

describe("tile validation", () => {
  const code = (fn: () => unknown) => {
    try {
      fn();
    } catch (e) {
      return (e as PlanError).code;
    }
    return null;
  };

  it("requires one positive integer extent per axis", () => {
    expect(code(() => tileFamily("T", [4, 4], [2]))).toBe("PLAN_TILE");
    expect(code(() => tileFamily("T", [4, 4], [2, 0]))).toBe("PLAN_TILE");
    expect(code(() => tileFamily("T", [4, 4], [2, -1]))).toBe("PLAN_TILE");
    expect(code(() => tileFamily("T", [4, 4], [2, 1.5]))).toBe("PLAN_TILE");
  });

  it("refuses a family whose tile count is past the safe integer range", () => {
    const huge = 2 ** 30;
    expect(code(() => tileFamily("T", [huge, huge], [1, 1]))).toBe("PLAN_SIZE");
  });
});
