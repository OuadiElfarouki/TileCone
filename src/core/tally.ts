/**
 * A count per element of one tensor, as weighted boxes.
 *
 * A region says which elements are in a set; a tally says how many times each
 * one is counted. The question it was written for is read multiplicity - how
 * many output elements of an operation read each element of an operand - and
 * a region cannot carry that: a union keeps reachability and drops the count.
 *
 * Representation: a list of pieces, each a box and a positive count, and the
 * value at an element is the **sum** of the counts of the pieces containing
 * it. Pieces may overlap, and overlap is how counts add up: a convolution's
 * halo is one shifted box per kernel tap, and the elements two taps both
 * reach are the ones read twice. So the rule here is the opposite of the one
 * in `region.ts`. There, summing box volumes double-counts; here, summing
 * `count * volume` is exactly the total, because each overlap is meant to be
 * counted once per piece.
 *
 * That is also why this is a separate type rather than a region with weights
 * attached. `canonicalize` drops a box contained in another and merges
 * neighbours into one box, both of which are union-preserving and both of
 * which would silently lose a count.
 *
 * `exact: false` means the tally is an upper bound at every element, and so
 * its support is a superset of the true support - the same one-sided contract
 * a region has, stated per element.
 */

import { Box, Region, disjointify, isEmpty, iv } from "./region";

export type Piece = Readonly<{ box: Box; count: number }>;

export type Tally = Readonly<{
  pieces: readonly Piece[];
  exact: boolean;
  reasons: readonly string[];
}>;

/** One step of a count along a single axis: every index in `[lo, hi)` counts `count`. */
export type Step = Readonly<{ lo: number; hi: number; count: number }>;

const mergeReasons = (...lists: (readonly string[])[]): string[] =>
  [...new Set(lists.flat())].sort();

function boxVolume(b: Box): number {
  let v = 1;
  for (const i of b) v *= Math.max(0, i.hi - i.lo);
  return v;
}

export function emptyTally(): Tally {
  return { pieces: [], exact: true, reasons: [] };
}

/**
 * Every element of `region` counted `n` times.
 *
 * The region is split into disjoint boxes first, since an element covered by
 * two of its boxes is still one element counted `n` times, not `2n`. A split
 * past its budget is a coarser superset, which keeps the tally an upper bound.
 */
export function uniformTally(region: Region, n: number): Tally {
  if (n <= 0 || isEmpty(region)) return { ...emptyTally(), exact: region.exact, reasons: region.reasons };
  const parts = disjointify(region);
  return {
    pieces: parts.boxes.map((box) => ({ box, count: n })),
    exact: parts.exact,
    reasons: [...parts.reasons],
  };
}

/**
 * One axis's steps as a step function: disjoint, sorted, no zero steps, and
 * adjacent steps of equal count merged. Steps given here may overlap, and add
 * where they do.
 */
function normalizeSteps(steps: readonly Step[]): Step[] {
  const events = new Map<number, number>();
  for (const s of steps) {
    if (s.hi <= s.lo || s.count === 0) continue;
    events.set(s.lo, (events.get(s.lo) ?? 0) + s.count);
    events.set(s.hi, (events.get(s.hi) ?? 0) - s.count);
  }
  const coordinates = [...events.keys()].sort((a, b) => a - b);
  const out: { lo: number; hi: number; count: number }[] = [];
  let running = 0;
  for (let i = 0; i < coordinates.length - 1; i++) {
    running += events.get(coordinates[i])!;
    if (running === 0) continue;
    const lo = coordinates[i];
    const hi = coordinates[i + 1];
    const last = out[out.length - 1];
    if (last && last.hi === lo && last.count === running) last.hi = hi;
    else out.push({ lo, hi, count: running });
  }
  return out;
}

/**
 * Halve an axis's step count by merging neighbouring pairs into their hull at
 * the larger count. Every index keeps at least its true count - the gap between
 * two steps, which counted zero, now counts the larger of the two - so the
 * result is an upper bound.
 */
function coarsenSteps(steps: Step[]): Step[] {
  const out: Step[] = [];
  for (let i = 0; i < steps.length; i += 2) {
    const a = steps[i];
    const b = steps[i + 1];
    out.push(b ? { lo: a.lo, hi: b.hi, count: Math.max(a.count, b.count) } : a);
  }
  return out;
}

/**
 * A count that is a product of one count per axis: the value at an index is
 * the product of each axis's step function at that index's coordinate.
 *
 * This is the common shape of read multiplicity. An operation that reads along
 * each axis independently - a broadcast, a pad, a strided window, a gather on
 * one axis - is counted per axis, and the counts multiply. Past `maxPieces`
 * the axis with the most steps is coarsened until the product fits, and the
 * result is then an upper bound rather than the count.
 */
export function separableTally(
  perAxis: readonly (readonly Step[])[],
  maxPieces: number,
  exact = true,
  reasons: readonly string[] = []
): Tally {
  let axes = perAxis.map(normalizeSteps);
  if (axes.some((steps) => steps.length === 0)) return { pieces: [], exact, reasons: [...reasons] };
  let capped = false;
  const product = () => axes.reduce((n, steps) => n * steps.length, 1);
  while (product() > maxPieces) {
    let widest = 0;
    axes.forEach((steps, axis) => {
      if (steps.length > axes[widest].length) widest = axis;
    });
    if (axes[widest].length <= 1) break;
    axes = axes.map((steps, axis) => (axis === widest ? coarsenSteps(steps) : steps));
    capped = true;
  }
  let pieces: { box: Box; count: number }[] = [{ box: [], count: 1 }];
  for (const steps of axes) {
    const next: { box: Box; count: number }[] = [];
    for (const p of pieces)
      for (const s of steps) next.push({ box: [...p.box, iv(s.lo, s.hi)], count: p.count * s.count });
    pieces = next;
  }
  return {
    pieces,
    exact: exact && !capped,
    reasons: capped ? mergeReasons(reasons, ["multiplicity cap"]) : [...reasons],
  };
}

/** The sum of several tallies on one tensor: counts add element by element. */
export function addTallies(tallies: readonly Tally[]): Tally {
  return {
    pieces: tallies.flatMap((t) => t.pieces),
    exact: tallies.every((t) => t.exact),
    reasons: mergeReasons(...tallies.map((t) => t.reasons)),
  };
}

export function markTallyInexact(t: Tally, ...reasons: string[]): Tally {
  return { pieces: t.pieces, exact: false, reasons: mergeReasons(t.reasons, reasons) };
}

/** The elements counted at least once. */
export function supportOf(t: Tally): Region {
  return { boxes: t.pieces.map((p) => p.box), exact: t.exact, reasons: [...t.reasons] };
}

/** @internal The count at one element; a test oracle, linear in the pieces. */
export function tallyAt(t: Tally, index: readonly number[]): number {
  let n = 0;
  for (const p of t.pieces)
    if (p.box.every((I, ax) => index[ax] >= I.lo && index[ax] < I.hi)) n += p.count;
  return n;
}

export type TallyMeasure = {
  // null means the count exceeds the safe integer range.
  /** The sum of the count over every element: `count * volume` summed over pieces. */
  total: number | null;
  /** Elements counted at least once. */
  elements: number | null;
  /** Elements counted exactly once. */
  once: number | null;
  /** Elements counted more than once. */
  repeated: number | null;
  /** The smallest and largest count on an element counted at all; 0 when none is. */
  min: number | null;
  max: number | null;
};

/**
 * Volume per summed count, accumulated into `out` scaled by `scale`.
 *
 * The same recursive dimensional sweep `count` uses for a union: choose the
 * axis with the fewest endpoints, and measure each slab between them over the
 * remaining axes with the pieces active there. Where `count` asks whether any
 * box is active, this asks what the active counts sum to, so it never builds
 * the disjoint level sets it measures.
 */
function levelVolumes(pieces: readonly Piece[], axes: number[], scale: number, out: Map<number, number>): void {
  if (pieces.length === 0) return;
  const add = (value: number, volume: number) => {
    if (value === 0 || volume === 0) return;
    out.set(value, (out.get(value) ?? 0) + volume * scale);
  };
  if (axes.length === 0) {
    add(pieces.reduce((n, p) => n + p.count, 0), 1);
    return;
  }
  if (pieces.length === 1) {
    let volume = 1;
    for (const axis of axes) volume *= Math.max(0, pieces[0].box[axis].hi - pieces[0].box[axis].lo);
    add(pieces[0].count, volume);
    return;
  }
  if (axes.length === 1) {
    const axis = axes[0];
    const events = new Map<number, number>();
    for (const p of pieces) {
      const I = p.box[axis];
      if (I.hi <= I.lo) continue;
      events.set(I.lo, (events.get(I.lo) ?? 0) + p.count);
      events.set(I.hi, (events.get(I.hi) ?? 0) - p.count);
    }
    const coordinates = [...events.keys()].sort((a, b) => a - b);
    let running = 0;
    for (let i = 0; i < coordinates.length - 1; i++) {
      running += events.get(coordinates[i])!;
      add(running, coordinates[i + 1] - coordinates[i]);
    }
    return;
  }

  let sweepAxis = axes[0];
  let fewest = Infinity;
  for (const axis of axes) {
    const endpoints = new Set<number>();
    for (const p of pieces) {
      endpoints.add(p.box[axis].lo);
      endpoints.add(p.box[axis].hi);
    }
    if (endpoints.size < fewest) {
      fewest = endpoints.size;
      sweepAxis = axis;
    }
  }
  const events = new Map<number, { add: Piece[]; remove: Piece[] }>();
  const at = (c: number) => {
    let e = events.get(c);
    if (!e) events.set(c, (e = { add: [], remove: [] }));
    return e;
  };
  for (const p of pieces) {
    if (p.box[sweepAxis].hi <= p.box[sweepAxis].lo) continue;
    at(p.box[sweepAxis].lo).add.push(p);
    at(p.box[sweepAxis].hi).remove.push(p);
  }
  const coordinates = [...events.keys()].sort((a, b) => a - b);
  const remaining = axes.filter((axis) => axis !== sweepAxis);
  const active = new Set<Piece>();
  let previous = coordinates[0];
  for (const c of coordinates) {
    if (c > previous && active.size) levelVolumes([...active], remaining, scale * (c - previous), out);
    const e = events.get(c)!;
    for (const p of e.remove) active.delete(p);
    for (const p of e.add) active.add(p);
    previous = c;
  }
}

/** How a tally's counts are distributed over the elements it covers. */
export function measureTally(t: Tally): TallyMeasure {
  const pieces = t.pieces.filter((p) => p.count > 0 && boxVolume(p.box) > 0);
  const safe = (n: number): number | null => Number.isSafeInteger(n) ? n : null;
  let total: number | null = 0;
  let counts = 0;
  for (const p of pieces) {
    if (total !== null) total = safe(total + p.count * boxVolume(p.box));
    counts += p.count;
  }
  // Sweep events subtract counts as pieces leave. If their sums could round,
  // cancellation could turn an unsafe count into a plausible small one.
  if (!Number.isSafeInteger(counts))
    return { total: null, elements: null, once: null, repeated: null, min: null, max: null };
  const rank = pieces[0]?.box.length ?? 0;
  const levels = new Map<number, number>();
  levelVolumes(pieces, Array.from({ length: rank }, (_, axis) => axis), 1, levels);
  let elements = 0;
  let min = Infinity;
  let max = 0;
  for (const [value, volume] of levels) {
    elements += volume;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const once = safe(levels.get(1) ?? 0);
  const measuredElements = safe(elements);
  return {
    total,
    elements: measuredElements,
    once,
    repeated: measuredElements === null || once === null ? null : measuredElements - once,
    min: elements ? min : 0,
    max,
  };
}
