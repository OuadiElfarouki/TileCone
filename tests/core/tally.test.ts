import { describe, expect, it } from "vitest";
import { Box, box, iv } from "../../src/core/region";
import {
  addTallies,
  measureTally,
  separableTally,
  tallyAt,
  tallyLevels,
  type Tally,
  uniformTally,
} from "../../src/core/tally";
import { randInt, rng } from "../corpus/harness";
import { unflatIndex } from "../corpus/oracle";

/** Every element of `shape` with its count, by summing pieces pointwise. */
function pointwise(t: Tally, shape: number[]): number[] {
  const n = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: n }, (_, f) => tallyAt(t, unflatIndex(f, shape)));
}

function randomBox(shape: number[], r: () => number): Box {
  return shape.map((e) => {
    const lo = randInt(r, 0, e);
    return iv(lo, randInt(r, lo + 1, e + 1));
  });
}

describe("a tally", () => {
  it("detects overflow in both multiplication and accumulation", () => {
    const tally = (pieces: Tally["pieces"]): Tally => ({ pieces, exact: true, reasons: [] });
    expect(measureTally(tally([{ box: box([0, 262145], [0, 262145]), count: 262145 }]))).toMatchObject({
      total: null, elements: 262145 ** 2, min: 262145, max: 262145,
    });
    const count = 2 ** 52;
    expect(measureTally(tally([
      { box: box([0, 1]), count }, { box: box([1, 2]), count: count - 1 },
    ])).total).toBe(Number.MAX_SAFE_INTEGER);
    expect(measureTally(tally([
      { box: box([0, 2]), count: count / 2 }, { box: box([2, 4]), count: count / 2 },
    ])).total).toBeNull();
  });

  it("does not sweep unsafe overlapping counts through cancellation", () => {
    const t: Tally = {
      pieces: [{ box: box([0, 2]), count: 2 ** 52 }, { box: box([1, 3]), count: 2 ** 52 }],
      exact: true, reasons: [],
    };
    expect(measureTally(t)).toEqual({ total: null, elements: null, once: null, repeated: null, min: null, max: null });
  });

  it("marks unsafe volumes unknown without losing safe counts", () => {
    const t: Tally = {
      pieces: [{ box: box([0, 262145], [0, 262145], [0, 262145]), count: 1 }],
      exact: true, reasons: [],
    };
    expect(measureTally(t)).toEqual({ total: null, elements: null, once: null, repeated: null, min: 1, max: 1 });
  });

  it("adds the counts of overlapping pieces", () => {
    const t = addTallies([
      { pieces: [{ box: box([0, 4]), count: 1 }], exact: true, reasons: [] },
      { pieces: [{ box: box([2, 6]), count: 3 }], exact: true, reasons: [] },
    ]);
    expect(pointwise(t, [7])).toEqual([1, 1, 4, 4, 3, 3, 0]);
  });

  it("counts an element of an overlapping region once per count, not once per box", () => {
    const t = uniformTally({ boxes: [box([0, 4], [0, 2]), box([2, 6], [0, 2])], exact: true, reasons: [] }, 2);
    expect(pointwise(t, [6, 2])).toEqual(Array(12).fill(2));
    expect(measureTally(t)).toMatchObject({ total: 24, elements: 12, once: 0, repeated: 12, min: 2, max: 2 });
  });

  it("measures the same distribution brute force does, at every rank", () => {
    const r = rng(31);
    for (const shape of [[11], [5, 7], [4, 3, 5]]) {
      for (let k = 0; k < 40; k++) {
        const pieces = Array.from({ length: randInt(r, 1, 7) }, () => ({
          box: randomBox(shape, r),
          count: randInt(r, 1, 4),
        }));
        const t: Tally = { pieces, exact: true, reasons: [] };
        const values = pointwise(t, shape).filter((v) => v > 0);
        expect(measureTally(t)).toEqual({
          total: values.reduce((a, b) => a + b, 0),
          elements: values.length,
          once: values.filter((v) => v === 1).length,
          repeated: values.filter((v) => v > 1).length,
          min: Math.min(...values),
          max: Math.max(...values),
        });
      }
    }
  });

  it("is a product of per-axis counts when built from steps", () => {
    const t = separableTally(
      [
        [{ lo: 0, hi: 3, count: 1 }, { lo: 1, hi: 2, count: 1 }],
        [{ lo: 0, hi: 2, count: 2 }],
      ],
      64
    );
    expect(t.exact).toBe(true);
    expect(pointwise(t, [3, 2])).toEqual([2, 2, 4, 4, 2, 2]);
  });

  it("past its piece cap becomes an upper bound at every element, never lower", () => {
    const r = rng(5);
    for (let k = 0; k < 30; k++) {
      const perAxis = [9, 7].map((n) =>
        Array.from({ length: randInt(r, 2, 8) }, () => {
          const lo = randInt(r, 0, n);
          return { lo, hi: randInt(r, lo + 1, n + 1), count: randInt(r, 1, 3) };
        })
      );
      const exact = separableTally(perAxis, 10_000);
      const capped = separableTally(perAxis, 4);
      expect(capped.pieces.length).toBeLessThanOrEqual(4);
      const truth = pointwise(exact, [9, 7]);
      const bound = pointwise(capped, [9, 7]);
      bound.forEach((v, f) => expect(v).toBeGreaterThanOrEqual(truth[f]));
      if (capped.pieces.length < exact.pieces.length) {
        expect(capped.exact).toBe(false);
        expect(capped.reasons).toContain("multiplicity cap");
      }
    }
  });

  it("is empty with no elements, and measures as such", () => {
    expect(measureTally(separableTally([[], [{ lo: 0, hi: 2, count: 1 }]], 8))).toEqual({
      total: 0, elements: 0, once: 0, repeated: 0, min: 0, max: 0,
    });
  });
});

/* The lens draws a tally from these pieces, so each element must land in
   exactly one of them, holding the sum the overlapping pieces give it. */
describe("a tally's disjoint levels", () => {
  it("covers every counted element once, at its summed count, on random tallies", () => {
    const r = rng(20261016);
    for (let trial = 0; trial < 120; trial++) {
      const shape = Array.from({ length: randInt(r, 1, 4) }, () => randInt(r, 1, 7));
      const t: Tally = {
        pieces: Array.from({ length: randInt(r, 1, 6) }, () => ({ box: randomBox(shape, r), count: randInt(r, 1, 5) })),
        exact: true,
        reasons: [],
      };
      const levels = tallyLevels(t);
      expect(levels.complete).toBe(true);
      const drawn = pointwise({ pieces: levels.pieces, exact: true, reasons: [] }, shape);
      expect(drawn).toEqual(pointwise(t, shape));
      // Disjoint: no element is covered by two level pieces.
      const covered = pointwise({ pieces: levels.pieces.map((p) => ({ box: p.box, count: 1 })), exact: true, reasons: [] }, shape);
      expect(Math.max(0, ...covered)).toBeLessThanOrEqual(1);
      // Lowest first, so a painter drawing in order leaves the highest on top.
      expect(levels.pieces.map((p) => p.count)).toEqual([...levels.pieces.map((p) => p.count)].sort((a, b) => a - b));
    }
  });

  it("draws nothing rather than part of a count past its piece budget", () => {
    const t: Tally = {
      pieces: Array.from({ length: 8 }, (_, i) => ({ box: box([i, i + 2], [0, 4]), count: 1 + i })),
      exact: true,
      reasons: [],
    };
    expect(tallyLevels(t, 4)).toEqual({ pieces: [], complete: false });
    expect(tallyLevels(t).complete).toBe(true);
  });

  it("returns nothing for an empty tally", () => {
    expect(tallyLevels({ pieces: [], exact: true, reasons: [] })).toEqual({ pieces: [], complete: true });
  });
});
