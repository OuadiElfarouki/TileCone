import { describe, expect, it } from "vitest";
import {
  Box,
  Region,
  boundingBox,
  box,
  canonicalize,
  contains,
  count,
  empty,
  fromBox,
  full,
  intersect,
  iv,
  points,
  removeBoxAt,
  sortRegion,
  subtractBox,
  translateRegion,
  union,
} from "../core/region";
import { rng, randInt } from "./harness";

function flatSet(r: Region, shape: number[]): Set<number> {
  const s = new Set<number>();
  for (const p of points(r)) {
    let f = 0;
    for (let i = 0; i < shape.length; i++) f = f * shape[i] + p[i];
    s.add(f);
  }
  return s;
}

function randBox(r: () => number, shape: number[]): Box {
  return shape.map((e) => {
    const lo = randInt(r, 0, e);
    return iv(lo, randInt(r, lo + 1, e + 1));
  });
}

describe("region algebra", () => {
  it("basic constructors", () => {
    expect(empty(2).boxes).toHaveLength(0);
    expect(count(full([3, 4]))).toBe(12);
    expect(count(fromBox(box([1, 3], [0, 2])))).toBe(4);
  });

  it("canonicalize drops empties and merges adjacent", () => {
    const r = canonicalize({
      boxes: [box([0, 2], [5, 5]), box([0, 2], [0, 3]), box([0, 2], [3, 6])],
      exact: true,
      reasons: [],
    });
    expect(r.boxes).toHaveLength(1);
    expect(r.boxes[0]).toEqual(box([0, 2], [0, 6]));
    expect(r.exact).toBe(true);
  });

  it("canonicalize is idempotent (randomized)", () => {
    const r = rng(1);
    const shape = [6, 6, 4];
    for (let t = 0; t < 100; t++) {
      const boxes: Box[] = [];
      const n = randInt(r, 1, 6);
      for (let i = 0; i < n; i++) boxes.push(randBox(r, shape));
      const c1 = canonicalize({ boxes, exact: true, reasons: [] });
      const c2 = canonicalize(c1);
      expect(sortRegion(c2)).toEqual(sortRegion(c1));
      // canonicalize preserves the element set
      expect(flatSet(c1, shape)).toEqual(flatSet({ boxes, exact: true, reasons: [] }, shape));
    }
  });

  it("canonical boxes are disjoint, so count never double-counts (randomized)", () => {
    const r = rng(2);
    const shape = [7, 5];
    for (let t = 0; t < 200; t++) {
      const boxes: Box[] = [];
      const n = randInt(r, 1, 5);
      for (let i = 0; i < n; i++) boxes.push(randBox(r, shape));
      const reg: Region = { boxes, exact: true, reasons: [] };
      expect(count(reg)).toBe(flatSet(reg, shape).size);
    }
  });

  it("union is commutative and matches set union (randomized)", () => {
    const r = rng(3);
    const shape = [6, 6];
    for (let t = 0; t < 100; t++) {
      const a = fromBox(randBox(r, shape));
      const b = fromBox(randBox(r, shape));
      const ab = union(a, b);
      const ba = union(b, a);
      expect(flatSet(ab, shape)).toEqual(flatSet(ba, shape));
      const expected = new Set([...flatSet(a, shape), ...flatSet(b, shape)]);
      expect(flatSet(ab, shape)).toEqual(expected);
    }
  });

  it("intersect matches set intersection (randomized)", () => {
    const r = rng(4);
    const shape = [6, 6];
    for (let t = 0; t < 100; t++) {
      const a = fromBox(randBox(r, shape));
      const b = fromBox(randBox(r, shape));
      const got = flatSet(intersect(a, b), shape);
      const sa = flatSet(a, shape);
      const expected = new Set([...flatSet(b, shape)].filter((x) => sa.has(x)));
      expect(got).toEqual(expected);
    }
  });

  it("subtractBox partitions correctly (randomized)", () => {
    const r = rng(5);
    const shape = [6, 5, 4];
    for (let t = 0; t < 100; t++) {
      const a = randBox(r, shape);
      const b = randBox(r, shape);
      const pieces = subtractBox(a, b);
      const setA = flatSet(fromBox(a), shape);
      const setB = flatSet(fromBox(b), shape);
      const expected = new Set([...setA].filter((x) => !setB.has(x)));
      const got = flatSet({ boxes: pieces, exact: true, reasons: [] }, shape);
      expect(got).toEqual(expected);
      // pieces disjoint
      let total = 0;
      for (const p of pieces) total += flatSet(fromBox(p), shape).size;
      expect(total).toBe(expected.size);
    }
  });

  it("contains agrees with membership", () => {
    const r = union(fromBox(box([0, 2], [0, 2])), fromBox(box([3, 5], [3, 5])));
    expect(contains(r, [1, 1])).toBe(true);
    expect(contains(r, [2, 2])).toBe(false);
    expect(contains(r, [4, 3])).toBe(true);
  });

  it("box count cap produces bounding box marked inexact", () => {
    const boxes: Box[] = [];
    for (let i = 0; i < 600; i++) boxes.push(box([i * 2, i * 2 + 1]));
    const r = canonicalize({ boxes, exact: true, reasons: [] });
    expect(r.exact).toBe(false);
    expect(r.reasons).toContain("box count cap");
    expect(r.boxes).toHaveLength(1);
    expect(r.boxes[0]).toEqual(box([0, 1199]));
  });

  it("boundingBox encloses everything", () => {
    const r = union(fromBox(box([1, 2], [5, 9])), fromBox(box([4, 8], [0, 3])));
    expect(boundingBox(r)).toEqual(box([1, 8], [0, 9]));
  });
});

describe("region editing operations", () => {
  const shape = [8, 8];

  it("translate shifts and preserves element count", () => {
    const r = fromBox(box([2, 4], [1, 3]));
    const moved = translateRegion(r, 0, 2, shape);
    expect(moved.boxes).toEqual([box([4, 6], [1, 3])]);
    expect(count(moved)).toBe(count(r));
  });

  it("translate clamps at the edge instead of eroding", () => {
    const r = fromBox(box([6, 8], [0, 2]));
    const moved = translateRegion(r, 0, 5, shape);
    expect(moved.boxes).toEqual([box([6, 8], [0, 2])]); // already flush against the edge
    const partial = translateRegion(fromBox(box([5, 7], [0, 2])), 0, 5, shape);
    expect(partial.boxes).toEqual([box([6, 8], [0, 2])]); // moved as far as it fits
    expect(count(partial)).toBe(4);
  });

  it("translate moves a multi-box region rigidly", () => {
    const r = union(fromBox(box([0, 2], [0, 2])), fromBox(box([4, 6], [4, 6])));
    const moved = translateRegion(r, 1, 2, shape);
    expect(count(moved)).toBe(count(r));
    expect(sortRegion(moved).boxes).toEqual([box([0, 2], [2, 4]), box([4, 6], [6, 8])]);
  });

  it("removeBoxAt drops exactly one box", () => {
    const r = union(fromBox(box([0, 2], [0, 2])), fromBox(box([5, 7], [5, 7])));
    expect(r.boxes).toHaveLength(2);
    const left = removeBoxAt(r, 0);
    expect(left.boxes).toHaveLength(1);
    expect(count(left)).toBe(4);
  });

  it("editing operations keep regions canonical (disjoint boxes)", () => {
    const r = rng(9);
    for (let t = 0; t < 60; t++) {
      let reg = union(fromBox(randBox(r, shape)), fromBox(randBox(r, shape)));
      reg = translateRegion(reg, 0, randInt(r, -3, 4), shape);
      reg = removeBoxAt(reg, 0);
      expect(count(reg)).toBe(flatSet(reg, shape).size);
    }
  });
});
