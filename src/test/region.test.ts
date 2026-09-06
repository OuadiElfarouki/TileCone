import { describe, expect, it } from "vitest";
import {
  Box,
  Region,
  boundingBox,
  box,
  canonicalize,
  count,
  coversAxisFully,
  empty,
  fromBox,
  full,
  intersect,
  iv,
  points,
  addPart,
  disjointify,
  partsOverlap,
  regionOverlap,
  subtractFromParts,
  translateAllParts,
  translatePart,
  sortRegion,
  subtractBox,
  subtract,
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

  it("does not under-approximate when subtracting an inexact region", () => {
    const minuend = fromBox(box([0, 10]));
    const representedSuperset: Region = {
      boxes: [box([0, 10])],
      exact: false,
      reasons: ["test bound"],
    };

    const difference = subtract(minuend, representedSuperset);
    expect(difference.boxes).toEqual(minuend.boxes);
    expect(difference.exact).toBe(false);
    expect(difference.reasons).toEqual(expect.arrayContaining(["test bound", "inexact subtraction"]));
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

describe("a full-axis pull is seen however the boxes are arranged", () => {
  // What `matmul(D, D)` produces: slot 0 asks for whole rows, slot 1 for whole
  // columns. Both bands survive whole, so both spanning boxes are present.
  const rowsAndCols = union(
    fromBox(box([0, 32], [0, 256])),
    fromBox(box([0, 256], [0, 32]))
  );

  it("keeps each operand band as one box", () => {
    expect(rowsAndCols.boxes).toHaveLength(2);
    expect(rowsAndCols.boxes.some((b) => b[0].hi - b[0].lo === 256)).toBe(true);
    expect(rowsAndCols.boxes.some((b) => b[1].hi - b[1].lo === 256)).toBe(true);
    expect(coversAxisFully(rowsAndCols, 0, 256)).toBe(true);
    expect(coversAxisFully(rowsAndCols, 1, 256)).toBe(true);
  });

  it("still proves containment rather than trusting a spanning box", () => {
    // Two boxes cover axis 1 between them across the rows they share, and
    // neither spans it. They differ on both axes, so no merge fuses them.
    // `boxes.some(b => b spans)` would miss this; the union does not.
    const between = union(fromBox(box([0, 4], [0, 130])), fromBox(box([2, 6], [120, 256])));
    expect(between.boxes).toHaveLength(2);
    expect(between.boxes.some((b) => b[1].hi - b[1].lo === 256)).toBe(false);
    expect(coversAxisFully(between, 1, 256)).toBe(true);
  });

  it("refuses an axis no line covers", () => {
    // one element short on both axes: nothing spans either
    const short = union(fromBox(box([0, 32], [0, 255])), fromBox(box([0, 255], [0, 32])));
    expect(coversAxisFully(short, 0, 256)).toBe(false);
    expect(coversAxisFully(short, 1, 256)).toBe(false);
  });

  it("agrees with the single-box case it replaces", () => {
    expect(coversAxisFully(fromBox(box([0, 4], [0, 16])), 1, 16)).toBe(true);
    expect(coversAxisFully(fromBox(box([0, 4], [0, 15])), 1, 16)).toBe(false);
    expect(coversAxisFully(empty(2), 0, 16)).toBe(false);
  });

  it("refuses to prove full-axis coverage from an inexact bound", () => {
    const bound: Region = {
      boxes: [box([0, 4], [0, 16])],
      exact: false,
      reasons: ["test bound"],
    };
    expect(coversAxisFully(bound, 1, 16)).toBe(false);
  });
});


describe("boxes are kept whole, and measured on the set", () => {
  // The shape this representation exists for: one tensor in two operand slots.
  const bands = union(fromBox(box([64, 128], [0, 256])), fromBox(box([0, 256], [32, 96])));

  it("names the two bands rather than three fragments", () => {
    expect(bands.boxes).toHaveLength(2);
    // Each is a band an operand actually reads, stated as an offset and extent.
    expect(bands.boxes).toContainEqual(box([64, 128], [0, 256]));
    expect(bands.boxes).toContainEqual(box([0, 256], [32, 96]));
  });

  it("counts the shared corner once", () => {
    // 64*256 + 256*64 would be 32768; the bands share a 64x64 square.
    expect(count(bands)).toBe(28672);
    const { unique, summed } = regionOverlap(bands);
    expect(unique).toBe(28672);
    expect(summed).toBe(32768);
    expect(summed - unique).toBe(64 * 64);
  });

  it("reports no overlap when the boxes are disjoint", () => {
    const apart = union(fromBox(box([0, 4], [0, 4])), fromBox(box([8, 12], [8, 12])));
    const { unique, summed } = regionOverlap(apart);
    expect(summed - unique).toBe(0);
  });

  it("still merges boxes that agree on every axis but one", () => {
    expect(union(fromBox(box([0, 4], [0, 8])), fromBox(box([4, 9], [0, 8]))).boxes)
      .toEqual([box([0, 9], [0, 8])]);
    // overlapping, not merely adjacent
    expect(union(fromBox(box([0, 6], [0, 8])), fromBox(box([4, 9], [0, 8]))).boxes)
      .toEqual([box([0, 9], [0, 8])]);
  });

  it("drops a box contained in another", () => {
    expect(union(fromBox(box([0, 10], [0, 10])), fromBox(box([2, 4], [2, 4]))).boxes)
      .toEqual([box([0, 10], [0, 10])]);
  });

  it("runs the two simplifications to a fixpoint", () => {
    // Merging the first two produces [0,8]x[0,8], which then swallows the third.
    const r = canonicalize({
      boxes: [box([0, 4], [0, 8]), box([4, 8], [0, 8]), box([2, 6], [1, 7])],
      exact: true,
      reasons: [],
    });
    expect(r.boxes).toEqual([box([0, 8], [0, 8])]);
  });

  it("falls back to a marked bounding box past the cap", () => {
    const many = Array.from({ length: 40 }, (_, i) => box([i * 2, i * 2 + 1], [0, 4]));
    const r = canonicalize({ boxes: many, exact: true, reasons: [] }, 8);
    expect(r.boxes).toHaveLength(1);
    expect(r.exact).toBe(false);
    expect(r.reasons).toContain("box count cap");
  });
});

describe("disjointify is a partition of the same set", () => {
  const cases: Box[][] = [
    [box([64, 128], [0, 256]), box([0, 256], [32, 96])],
    [box([0, 4], [0, 10]), box([0, 10], [0, 4])],
    [box([0, 8], [2, 4], [0, 8]), box([0, 8], [0, 8], [2, 4])],
    [box([0, 1], [0, 1]), box([2, 3], [2, 3])],
    [box([0, 6]), box([3, 9]), box([8, 12])],
  ];

  const key = (p: number[]) => p.join(",");

  it("holds every element exactly once, and the same elements as the boxes", () => {
    for (const boxes of cases) {
      const stored = canonicalize({ boxes, exact: true, reasons: [] });
      const flat = disjointify(stored);
      // no element is in two disjoint boxes
      const summed = flat.boxes.reduce(
        (a, b) => a + b.reduce((v, i) => v * (i.hi - i.lo), 1),
        0
      );
      expect(summed).toBe(count(stored));
      // and the two forms enumerate the same set
      const a = new Set([...points(stored)].map(key));
      const b = new Set([...points(flat)].map(key));
      expect(a).toEqual(b);
      expect(a.size).toBe(count(stored));
    }
  });

  it("leaves the stored form alone", () => {
    const stored = canonicalize({ boxes: cases[0], exact: true, reasons: [] });
    const before = JSON.parse(JSON.stringify(stored.boxes));
    disjointify(stored);
    expect(stored.boxes).toEqual(before);
  });
});
