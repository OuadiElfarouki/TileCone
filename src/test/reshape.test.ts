import { describe, expect, it } from "vitest";
import { groupAxes, linearRangeToBoxes, reshapeMapBox } from "../core/ops/reshape";
import { box, count, fromBox, points, sortRegion, Box } from "../core/region";
import { checkGraph, G, rng, randInt } from "./harness";
import { flatIndex } from "./oracle";

function strides(sh: number[]): number[] {
  const s = new Array(sh.length).fill(1);
  for (let i = sh.length - 2; i >= 0; i--) s[i] = s[i + 1] * sh[i + 1];
  return s;
}

/** Brute-force image of a box under reshape via linear indices. */
function bruteMap(b: Box, from: number[], to: number[]): Set<number> {
  const out = new Set<number>();
  for (const p of points(fromBox(b))) {
    let lin = 0;
    strides(from).forEach((s, i) => (lin += p[i] * s));
    out.add(lin); // linear index doubles as flat index of `to`
  }
  return out;
}

function regionFlat(r: ReturnType<typeof fromBox>, shape: number[]): Set<number> {
  const out = new Set<number>();
  for (const p of points(r)) out.add(flatIndex(p, shape));
  return out;
}

describe("axis grouping", () => {
  it("finds coarsest common grouping", () => {
    expect(groupAxes([2, 3, 4], [6, 4])).toEqual([
      { from: [2, 3], to: [6] },
      { from: [4], to: [4] },
    ]);
    expect(groupAxes([4, 4], [16])).toEqual([{ from: [4, 4], to: [16] }]);
    expect(groupAxes([16], [2, 8])).toEqual([{ from: [16], to: [2, 8] }]);
    expect(groupAxes([2, 3], [2, 3])).toEqual([
      { from: [2], to: [2] },
      { from: [3], to: [3] },
    ]);
  });
  it("rejects mismatched element counts", () => {
    expect(() => groupAxes([2, 3], [5])).toThrow();
  });
});

describe("linearRangeToBoxes", () => {
  it("covers exactly the range (randomized)", () => {
    const r = rng(7);
    for (let t = 0; t < 200; t++) {
      const rank = randInt(r, 1, 4);
      const ext = Array.from({ length: rank }, () => randInt(r, 2, 6));
      const n = ext.reduce((a, b) => a * b, 1);
      const a = randInt(r, 0, n);
      const b = randInt(r, a, n + 1);
      const boxes = linearRangeToBoxes(a, b, ext);
      const st = strides(ext);
      const got = new Set<number>();
      for (const bx of boxes)
        for (const p of points(fromBox(bx))) {
          let lin = 0;
          st.forEach((s, i) => (lin += p[i] * s));
          got.add(lin);
        }
      expect(got.size).toBe(b - a);
      for (let x = a; x < b; x++) expect(got.has(x)).toBe(true);
    }
  });
});

describe("reshapeMapBox", () => {
  const shapes: [number[], number[]][] = [
    [[4, 4], [16]],
    [[16], [2, 8]],
    [[16], [4, 4]],
    [[2, 3, 4], [6, 4]],
    [[6, 4], [2, 3, 4]],
    [[2, 3, 4], [4, 6]], // regrouping across boundaries
    [[2, 2, 2, 2], [4, 4]],
    [[5, 6], [3, 10]],
    [[2, 3, 4], [24]],
    [[1, 6, 4], [6, 4, 1]],
  ];

  it("exact regions equal brute force; inexact are supersets (randomized boxes)", () => {
    const r = rng(11);
    for (const [from, to] of shapes) {
      for (let t = 0; t < 40; t++) {
        const b: Box = from.map((e) => {
          const lo = randInt(r, 0, e);
          return { lo, hi: randInt(r, lo + 1, e + 1) };
        });
        const mapped = reshapeMapBox(b, from, to);
        const truth = bruteMap(b, from, to);
        const got = regionFlat(mapped, to);
        if (mapped.exact) {
          expect(got, `exact ${from}->${to} box ${JSON.stringify(b)}`).toEqual(truth);
        } else {
          for (const x of truth) expect(got.has(x)).toBe(true);
        }
      }
    }
  });

  it("Tier-1 style cases produce a single exact box", () => {
    // (B,S,H*D) -> (B,S,H,D): pure split
    const r1 = reshapeMapBox(box([0, 2], [1, 3], [4, 8]), [2, 4, 12], [2, 4, 3, 4]);
    expect(r1.exact).toBe(true);
    expect(r1.boxes).toHaveLength(1);
    expect(r1.boxes[0]).toEqual(box([0, 2], [1, 3], [1, 2], [0, 4]));
    // (B,S,H,D) -> (B*S,H,D): merge, box full on inner merged axis
    const r2 = reshapeMapBox(box([0, 2], [0, 4], [1, 2], [0, 4]), [2, 4, 3, 4], [8, 3, 4]);
    expect(r2.exact).toBe(true);
    expect(r2.boxes).toHaveLength(1);
    expect(r2.boxes[0]).toEqual(box([0, 8], [1, 2], [0, 4]));
  });

  it("the reshape trap: (4,4) -> (16,) -> (2,8), straddling selection", () => {
    // Selecting flat range [6, 10) of the (16,) view = source rows 1..2 partially
    const r = reshapeMapBox(box([6, 10]), [16], [4, 4]);
    expect(r.exact).toBe(true);
    const flat = regionFlat(r, [4, 4]);
    expect(flat).toEqual(new Set([6, 7, 8, 9]));
    expect(r.boxes.length).toBe(2); // NOT a single box
  });

  it("adjointness: forward image and backward preimage agree", () => {
    const r = rng(13);
    for (const [from, to] of shapes) {
      for (let t = 0; t < 10; t++) {
        const b: Box = from.map((e) => {
          const lo = randInt(r, 0, e);
          return { lo, hi: randInt(r, lo + 1, e + 1) };
        });
        const fwd = reshapeMapBox(b, from, to);
        if (!fwd.exact) continue;
        // map back and verify we recover exactly the original set
        let backSet = new Set<number>();
        for (const fb of fwd.boxes) {
          const back = reshapeMapBox(fb, to, from);
          if (!back.exact) {
            backSet = null as never;
            break;
          }
          for (const x of regionFlat(back, from)) backSet.add(x);
        }
        if (backSet === (null as never)) continue;
        expect(backSet).toEqual(regionFlat(fromBox(b), from));
      }
    }
  });
});

describe("reshape end-to-end through propagation", () => {
  it("reshape trap graph", () => {
    checkGraph(
      G({ X: [4, 4] }, [
        ["n0", "reshape", ["X"], ["F"], { shape: [16] }],
        ["n1", "reshape", ["F"], ["Y"], { shape: [2, 8] }],
      ])
    );
  });
  it("split/merge attention-style reshapes", () => {
    checkGraph(
      G({ X: [2, 3, 8] }, [
        ["n0", "reshape", ["X"], ["Y"], { shape: [2, 3, 2, 4] }],
        ["n1", "reshape", ["Y"], ["Z"], { shape: [6, 2, 4] }],
      ]),
      { perTensorElementCap: 16 }
    );
  });
  it("count sanity", () => {
    const r = reshapeMapBox(box([0, 3]), [16], [4, 4]);
    expect(count(sortRegion(r))).toBe(3);
  });
});
