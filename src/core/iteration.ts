/**
 * An operation's iteration domain: the index space its arithmetic ranges over,
 * and how each operand and the output are addressed from it.
 *
 * For `einsum("ik,kj->ij", A, B)` the domain has three axes, `i` and `j`
 * (output) and `k` (reduced). `A` is addressed by `(i, k)`, `B` by `(k, j)` and
 * the output by `(i, j)`. One point of the domain is one term of the
 * contraction, and an iteration box - one interval per loop axis - is a set of
 * terms.
 *
 * The relations an operation is asked about are projections of such a box:
 *
 * - `backward` of an output box is the box with every reduced axis taken whole,
 *   projected onto each operand;
 * - `coaccess` of an operand box is the box that operand fixes, with every axis
 *   it does not carry taken whole, projected onto another operand;
 * - read multiplicity is the size of a projection's fibre: the box's extent
 *   over the output axes an operand does not carry;
 * - a box completes the output elements it reaches when it spans every reduced
 *   axis, and otherwise contributes part of each.
 *
 * A box narrower than a reduced axis is what none of those hooks can state: one
 * step of a staged reduction. That is the reason the domain is a type of its
 * own rather than something each hook rebuilds.
 *
 * The domain is concrete, like `Region`: extents are numbers and a projection
 * is a union of boxes.
 */

import { Box, Interval, Region, canonicalize, empty, fromBox, iv } from "./region";

export type LoopAxis = Readonly<{
  /** The operation's own name for the axis: an einsum label. */
  label: string;
  extent: number;
  /** `output` axes index the result; `reduced` axes are summed away. */
  role: "output" | "reduced";
}>;

/** One loop axis per tensor axis, as an index into `IterationDomain.axes`. A
 * loop axis named twice is read on its diagonal. */
export type Access = readonly number[];

export type IterationDomain = Readonly<{
  axes: readonly LoopAxis[];
  /** Parallel to the operation's inputs. */
  inputs: readonly Access[];
  /** Names each output-role axis exactly once. */
  output: Access;
}>;

/** Every term of the operation. */
export function fullIterationBox(d: IterationDomain): Box {
  return d.axes.map((axis) => iv(0, axis.extent));
}

/** The terms that compute an output box: its own intervals on the output axes,
 * every reduced axis whole. */
export function iterationBoxOfOutput(d: IterationDomain, outBox: Box): Box {
  const box: Interval[] = d.axes.map((axis) => iv(0, axis.extent));
  d.output.forEach((axis, tensorAxis) => (box[axis] = outBox[tensorAxis]));
  return box;
}

/**
 * The terms that read some element of an operand box: the box's intervals on
 * the loop axes the operand carries, every other axis whole. A loop axis the
 * operand names twice is read where its tensor axes agree, so it takes the
 * intersection of their intervals. `null` when no term reads the box.
 */
export function iterationBoxOfInput(d: IterationDomain, slot: number, inBox: Box): Box | null {
  const box: Interval[] = d.axes.map((axis) => iv(0, axis.extent));
  const seen = new Set<number>();
  const access = d.inputs[slot];
  for (let tensorAxis = 0; tensorAxis < access.length; tensorAxis++) {
    const axis = access[tensorAxis];
    const cur = inBox[tensorAxis];
    box[axis] = seen.has(axis)
      ? iv(Math.max(box[axis].lo, cur.lo), Math.min(box[axis].hi, cur.hi))
      : cur;
    seen.add(axis);
  }
  for (const axis of seen) if (box[axis].hi <= box[axis].lo) return null;
  return box;
}

/** The output elements an iteration box contributes to. */
export function projectOutput(d: IterationDomain, iterBox: Box): Box {
  return d.output.map((axis) => iterBox[axis]);
}

/**
 * The elements of operand `slot` that the terms of an iteration box read.
 *
 * Exact as one box unless the operand names a loop axis twice. Then the true
 * set is a diagonal: enumerated while it has at most `diagEnum` positions, and
 * past that covered by the sub-squares of a blocked staircase, marked inexact.
 */
export function projectInput(
  d: IterationDomain,
  slot: number,
  iterBox: Box,
  diagEnum: number
): Region {
  const access = d.inputs[slot];
  const base: Box = access.map((axis) => iterBox[axis]);
  const counts = new Map<number, number>();
  access.forEach((axis) => counts.set(axis, (counts.get(axis) ?? 0) + 1));
  const repeated = [...counts.entries()].filter(([, c]) => c > 1).map(([axis]) => axis);
  if (repeated.length === 0) return fromBox(base);

  let combos = 1;
  for (const axis of repeated) combos *= Math.max(0, iterBox[axis].hi - iterBox[axis].lo);
  if (combos === 0) return empty(access.length);
  if (combos > diagEnum) {
    // Too many diagonal positions to name individually, but the staircase is
    // still worth keeping: chunk each repeated axis into blocks and emit one
    // box per combination of blocks. Each box is the sub-square a block spans,
    // which contains that block's diagonal cells, so the union is a superset -
    // and for a 300-wide diagonal it is a few hundred elements, not 90,000.
    const perAxis = Math.max(1, Math.floor(Math.pow(diagEnum, 1 / repeated.length)));
    const blocks = repeated.map((axis) => {
      const I = iterBox[axis];
      const width = I.hi - I.lo;
      const groups = Math.min(perAxis, width);
      const size = Math.ceil(width / groups);
      const out: Interval[] = [];
      for (let start = I.lo; start < I.hi; start += size)
        out.push(iv(start, Math.min(start + size, I.hi)));
      return out;
    });
    const boxes: Box[] = [];
    const walk = (ri: number, assign: Map<number, Interval>) => {
      if (ri === repeated.length) {
        boxes.push(access.map((axis, tensorAxis) => assign.get(axis) ?? base[tensorAxis]));
        return;
      }
      for (const block of blocks[ri]) {
        assign.set(repeated[ri], block);
        walk(ri + 1, assign);
      }
      assign.delete(repeated[ri]);
    };
    walk(0, new Map());
    return canonicalize({ boxes, exact: false, reasons: ["diagonal einsum"] });
  }
  const boxes: Box[] = [];
  const rec = (ri: number, assign: Map<number, number>) => {
    if (ri === repeated.length) {
      boxes.push(
        access.map((axis, tensorAxis) =>
          assign.has(axis) ? iv(assign.get(axis)!, assign.get(axis)! + 1) : base[tensorAxis]
        )
      );
      return;
    }
    const axis = repeated[ri];
    for (let v = iterBox[axis].lo; v < iterBox[axis].hi; v++) {
      assign.set(axis, v);
      rec(ri + 1, assign);
    }
    assign.delete(axis);
  };
  rec(0, new Map());
  return canonicalize({ boxes, exact: true, reasons: [] });
}

/**
 * How many output elements of an iteration box read each element of operand
 * `slot` that the box reads: the box's extent over the output axes the operand
 * does not carry. Reduced axes do not count, because an output that meets an
 * element in several terms still reads it once.
 */
export function fibreOf(d: IterationDomain, slot: number, iterBox: Box): number {
  const carried = new Set(d.inputs[slot]);
  let readers = 1;
  d.axes.forEach((axis, i) => {
    if (axis.role === "output" && !carried.has(i))
      readers *= Math.max(0, iterBox[i].hi - iterBox[i].lo);
  });
  return readers;
}

/** Whether the box holds every term of the output elements it reaches. A box
 * that does not is a partial contribution to each of them. */
export function spansReduced(d: IterationDomain, iterBox: Box): boolean {
  return d.axes.every(
    (axis, i) => axis.role === "output" || (iterBox[i].lo <= 0 && iterBox[i].hi >= axis.extent)
  );
}

/** The number of terms in an iteration box. */
export function iterationVolume(iterBox: Box): number {
  let n = 1;
  for (const I of iterBox) n *= Math.max(0, I.hi - I.lo);
  return n;
}
