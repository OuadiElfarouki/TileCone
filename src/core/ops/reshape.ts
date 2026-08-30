import { z } from "zod";
import { Box, Interval, Region, boundingBox, canonicalize, iv, MAX_BOXES } from "../region";
import { resolveShape } from "../shapes";
import { OpSpec, uniformDTypeOutputs } from "./types";

export const MAX_RESHAPE_RUNS = 4096;

/**
 * Reshape provenance, implemented per IDEA.md §3.2 but unified:
 *
 * Tier 1 (axis factorization) appears here as the GROUPING step — the coarsest
 * common decomposition into independent axis groups with equal products. 1:1
 * groups copy intervals directly and product-of-groups keeps everything exact
 * and box-shaped for the common split/merge cases.
 *
 * Tier 2 (linear run decomposition) runs INSIDE each non-trivial group: the box
 * restricted to the group is decomposed into contiguous row-major runs, each run
 * is converted to boxes in the other shape (peel head / full middle / peel tail),
 * and the results are unioned.
 *
 * Tier 3: when a group's run count exceeds the budget, that group falls back to
 * the bounding box of its linear range with `exact: false`.
 */

type AxisGroup = { from: number[]; to: number[] }; // extents on each side, equal products

export function groupAxes(fromShape: number[], toShape: number[]): AxisGroup[] {
  const groups: AxisGroup[] = [];
  let i = 0,
    j = 0;
  while (i < fromShape.length || j < toShape.length) {
    let pi = i < fromShape.length ? fromShape[i] : 1;
    let pj = j < toShape.length ? toShape[j] : 1;
    const gi = i < fromShape.length ? [fromShape[i]] : [];
    const gj = j < toShape.length ? [toShape[j]] : [];
    i++;
    j++;
    while (pi !== pj) {
      if (pi < pj) {
        if (i >= fromShape.length) throw new Error("reshape: element count mismatch");
        pi *= fromShape[i];
        gi.push(fromShape[i]);
        i++;
      } else {
        if (j >= toShape.length) throw new Error("reshape: element count mismatch");
        pj *= toShape[j];
        gj.push(toShape[j]);
        j++;
      }
    }
    // absorb trailing extent-1 axes into the current group
    while (i < fromShape.length && fromShape[i] === 1 && (j >= toShape.length || toShape[j] !== 1)) {
      gi.push(1);
      i++;
    }
    while (j < toShape.length && toShape[j] === 1 && (i >= fromShape.length || fromShape[i] !== 1)) {
      gj.push(1);
      j++;
    }
    groups.push({ from: gi, to: gj });
  }
  return groups;
}

function strides(extents: number[]): number[] {
  const s = new Array(extents.length).fill(1);
  for (let i = extents.length - 2; i >= 0; i--) s[i] = s[i + 1] * extents[i + 1];
  return s;
}

/** Decompose a contiguous linear range [a, b) into row-major boxes over `ext`. */
export function linearRangeToBoxes(a: number, b: number, ext: number[]): Box[] {
  if (a >= b) return [];
  if (ext.length === 0) return [[]];
  if (ext.length === 1) return [[iv(a, b)]];
  const inner = ext.slice(1).reduce((x, y) => x * y, 1);
  const qa = Math.floor(a / inner);
  const ra = a - qa * inner;
  const qb = Math.floor(b / inner);
  const rb = b - qb * inner;
  const tail = ext.slice(1);
  const out: Box[] = [];
  if (qa === (rb > 0 ? qb : qb - 1)) {
    // entire range inside one leading slice
    for (const t of linearRangeToBoxes(ra, ra + (b - a), tail)) out.push([iv(qa, qa + 1), ...t]);
    return out;
  }
  const headEnd = ra > 0 ? qa + 1 : qa;
  if (ra > 0) for (const t of linearRangeToBoxes(ra, inner, tail)) out.push([iv(qa, qa + 1), ...t]);
  if (qb > headEnd) out.push([iv(headEnd, qb), ...tail.map((e) => iv(0, e))]);
  if (rb > 0) for (const t of linearRangeToBoxes(0, rb, tail)) out.push([iv(qb, qb + 1), ...t]);
  return out;
}

/** Contiguous linear runs covered by `box` over row-major `ext`, as [start, end) pairs. */
function boxToRuns(box: Box, ext: number[], budget: number): [number, number][] | null {
  const st = strides(ext);
  // s = first axis (from the end) before the maximal full suffix
  let s = box.length - 1;
  while (s >= 0 && box[s].lo === 0 && box[s].hi === ext[s]) s--;
  if (s < 0) return [[0, ext.reduce((a, b) => a * b, 1)]];
  // runs iterate over all prefix coords on axes < s; axis s spans contiguously
  let nRuns = 1;
  for (let ax = 0; ax < s; ax++) nRuns *= box[ax].hi - box[ax].lo;
  if (nRuns > budget) return null;
  const runs: [number, number][] = [];
  const idx = box.slice(0, s).map((I) => I.lo);
  while (true) {
    let base = 0;
    for (let ax = 0; ax < s; ax++) base += idx[ax] * st[ax];
    runs.push([base + box[s].lo * st[s], base + box[s].hi * st[s]]);
    let ax = s - 1;
    while (ax >= 0) {
      idx[ax]++;
      if (idx[ax] < box[ax].hi) break;
      idx[ax] = box[ax].lo;
      ax--;
    }
    if (ax < 0) break;
  }
  return runs;
}

/** Map a box through a reshape: from `fromShape` coordinates into `toShape` coordinates. */
export function reshapeMapBox(b: Box, fromShape: number[], toShape: number[]): Region {
  const groups = groupAxes(fromShape, toShape);
  // Per group: a small Region over the group's `to` axes.
  const perGroup: { boxes: Box[]; exact: boolean; reasons: string[] }[] = [];
  let fi = 0;
  let budget = MAX_RESHAPE_RUNS;
  for (const g of groups) {
    const gBox = b.slice(fi, fi + g.from.length);
    fi += g.from.length;
    if (g.from.length === 1 && g.to.length === 1) {
      perGroup.push({ boxes: [[{ ...gBox[0] }]], exact: true, reasons: [] });
      continue;
    }
    const runs = boxToRuns(gBox, g.from, budget);
    if (runs === null) {
      // Tier 3: bounding box of the group's linear extent
      const st = strides(g.from);
      let lo = 0,
        hi = 0;
      gBox.forEach((I, ax) => {
        lo += I.lo * st[ax];
        hi += (I.hi - 1) * st[ax];
      });
      const bb = boundingBox({
        boxes: linearRangeToBoxes(lo, hi + 1, g.to),
        exact: false,
        reasons: [],
      })!;
      perGroup.push({ boxes: [bb], exact: false, reasons: ["reshape run cap exceeded"] });
      continue;
    }
    budget = Math.max(64, budget - runs.length);
    const boxes: Box[] = [];
    for (const [a, e] of runs) boxes.push(...linearRangeToBoxes(a, e, g.to));
    const r = canonicalize({ boxes, exact: true, reasons: [] }, MAX_BOXES);
    perGroup.push({ boxes: r.boxes, exact: r.exact, reasons: r.reasons });
  }
  // Combine groups: cartesian product. Collapse the biggest groups first if it explodes.
  const counts = perGroup.map((g) => g.boxes.length);
  let product = counts.reduce((a, c) => a * c, 1);
  let exact = perGroup.every((g) => g.exact);
  const reasons = new Set(perGroup.flatMap((g) => g.reasons));
  while (product > MAX_BOXES) {
    let worst = 0;
    for (let k = 1; k < perGroup.length; k++)
      if (perGroup[k].boxes.length > perGroup[worst].boxes.length) worst = k;
    if (perGroup[worst].boxes.length <= 1) break;
    const bb = boundingBox({ boxes: perGroup[worst].boxes, exact: false, reasons: [] })!;
    perGroup[worst] = { boxes: [bb], exact: false, reasons: [] };
    exact = false;
    reasons.add("reshape box product cap");
    product = perGroup.reduce((a, g) => a * g.boxes.length, 1);
  }
  const interleaved: Interval[][][] = perGroup.map((g) => g.boxes as Interval[][]);
  let combos: Box[] = [[]];
  for (const list of interleaved) {
    const next: Box[] = [];
    for (const c of combos) for (const gb of list) next.push([...c, ...gb.map((I) => ({ ...I }))]);
    combos = next;
  }
  return canonicalize({ boxes: combos, exact, reasons: [...reasons] });
}

type ReshapeAttrs = { shape: (string | number)[] };

export const reshapeOp: OpSpec = {
  name: "reshape",
  attrSchema: z.object({ shape: z.array(z.union([z.string(), z.number().int().min(1)])) }),
  arity: { inputs: 1, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("reshape"),
  inferShapes: (inShapes, attrs, params) => {
    const target = resolveShape((attrs as ReshapeAttrs).shape, params ?? {});
    const inN = inShapes[0].reduce((a, b) => a * b, 1);
    const outN = target.reduce((a, b) => a * b, 1);
    if (inN !== outN) throw new Error(`reshape: ${inN} elements -> ${outN}`);
    return [target];
  },
  backward: (_s, outBox, ctx) => [reshapeMapBox(outBox, ctx.outShapes[0], ctx.inShapes[0])],
  forward: (_s, inBox, ctx) => [reshapeMapBox(inBox, ctx.inShapes[0], ctx.outShapes[0])],
  oracleDeps: (_s, outIndex, ctx) => {
    const outSt = strides(ctx.outShapes[0]);
    const inSt = strides(ctx.inShapes[0]);
    let lin = 0;
    outIndex.forEach((v, ax) => (lin += v * outSt[ax]));
    const idx = inSt.map((s, ax) => {
      const v = Math.floor(lin / s);
      lin -= v * s;
      void ax;
      return v;
    });
    return [[idx]];
  },
  flopsFor: () => 0,
};
