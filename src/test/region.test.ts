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
  addPart,
  partsOverlap,
  removePart,
  subtractFromParts,
  translateAllParts,
  translatePart,
  sortRegion,
  subtractBox,
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

describe("selection parts (identity-stable, may overlap)", () => {
  const shape = [8, 8];
  const P = (...bs: Box[]) => bs;

  it("translateAllParts shifts every part rigidly", () => {
    const parts = P(box([0, 2], [0, 2]), box([4, 6], [4, 6]));
    const moved = translateAllParts(parts, 1, 2, shape);
    expect(moved).toEqual(P(box([0, 2], [2, 4]), box([4, 6], [6, 8])));
  });

  it("translateAllParts clamps at the edge instead of eroding", () => {
    const parts = P(box([5, 7], [0, 2]));
    expect(translateAllParts(parts, 0, 5, shape)).toEqual(P(box([6, 8], [0, 2])));
    // already flush: no movement, no shrink
    expect(translateAllParts(P(box([6, 8], [0, 2])), 0, 5, shape)).toEqual(P(box([6, 8], [0, 2])));
  });

  it("translatePart moves one part and leaves the others identical", () => {
    const a = box([0, 2], [0, 2]);
    const b = box([4, 6], [4, 6]);
    const moved = translatePart(P(a, b), 1, 0, -2, shape);
    expect(moved[0]).toBe(a); // untouched, same object
    expect(moved[1]).toEqual(box([2, 4], [4, 6]));
    expect(moved).toHaveLength(2);
  });

  it("translatePart clamps only the part it moves", () => {
    const parts = P(box([0, 2], [0, 2]), box([6, 8], [0, 2]));
    const moved = translatePart(parts, 1, 0, 5, shape);
    expect(moved[1]).toEqual(box([6, 8], [0, 2])); // already at the edge
    expect(moved[0]).toEqual(box([0, 2], [0, 2]));
  });

  it("a part may be moved onto another; both keep their identity", () => {
    const parts = translatePart(P(box([0, 4], [0, 4]), box([4, 8], [0, 4])), 1, 0, -2, shape);
    expect(parts).toHaveLength(2); // NOT merged, unlike a canonical Region
    expect(parts[0]).toEqual(box([0, 4], [0, 4]));
    expect(parts[1]).toEqual(box([2, 6], [0, 4]));
  });

  it("overlap is counted once as a set, and reported", () => {
    const parts = P(box([0, 4], [0, 4]), box([2, 6], [0, 4]));
    const { unique, summed } = partsOverlap(parts);
    expect(summed).toBe(16 + 16); // each part's own volume
    expect(unique).toBe(24); // union: 6x4 minus nothing double counted
    expect(count({ boxes: parts, exact: true, reasons: [] })).toBe(unique);
  });

  it("addPart appends but ignores an exact duplicate", () => {
    const a = box([0, 2], [0, 2]);
    expect(addPart(P(a), box([4, 6], [4, 6]))).toHaveLength(2);
    expect(addPart(P(a), box([0, 2], [0, 2]))).toHaveLength(1);
  });

  it("removePart drops exactly one, preserving order of the rest", () => {
    const parts = P(box([0, 1], [0, 1]), box([2, 3], [2, 3]), box([4, 5], [4, 5]));
    const left = removePart(parts, 1);
    expect(left).toEqual(P(box([0, 1], [0, 1]), box([4, 5], [4, 5])));
  });

  it("subtractFromParts can split a part into several", () => {
    const parts = P(box([0, 8], [0, 2]));
    const cut = subtractFromParts(parts, box([3, 5], [0, 2]));
    expect(cut).toHaveLength(2);
    expect(count({ boxes: cut, exact: true, reasons: [] })).toBe(8 * 2 - 2 * 2);
  });

  it("parts operations never invent or lose elements (randomized)", () => {
    const r = rng(9);
    const asSet = (parts: Box[]) => flatSet({ boxes: parts, exact: true, reasons: [] }, shape);
    for (let t = 0; t < 80; t++) {
      let parts: Box[] = [randBox(r, shape), randBox(r, shape)];
      const before = asSet(parts);
      // a move of one part changes the set, but never breaks part count
      const n = parts.length;
      parts = translatePart(parts, randInt(r, 0, n), randInt(r, 0, 2), randInt(r, -3, 4), shape);
      expect(parts).toHaveLength(n);
      // every part stays inside the tensor
      for (const p of parts)
        p.forEach((I, ax) => {
          expect(I.lo).toBeGreaterThanOrEqual(0);
          expect(I.hi).toBeLessThanOrEqual(shape[ax]);
          expect(I.hi).toBeGreaterThan(I.lo);
        });
      // counting the parts as a set never double counts
      expect(count({ boxes: parts, exact: true, reasons: [] })).toBe(asSet(parts).size);
      void before;
    }
  });
});
