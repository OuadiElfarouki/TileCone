/**
 * Region algebra: unions of half-open axis-aligned boxes in tensor index space.
 *
 * Representation choice: a region is a union of BOXES that may overlap.
 * `canonicalize` normalizes it only in union-preserving ways - it drops empties,
 * drops boxes contained in another, and merges boxes that agree on every axis
 * but one. It never splits a box on overlap.
 *
 * The earlier representation did split, so that box volumes could be summed
 * directly. It was abandoned because splitting destroys the only structure the
 * boxes carry. An operation reading one tensor in two operand slots - the
 * `matmul(A, A)` case - contributes a row band and a column band; splitting
 * clips the overlap out of one of them and leaves three fragments, two of which
 * correspond to nothing anyone reads. A box should stay the unit it looks
 * like: an offset and an extent, the thing a kernel loads. A cross is not that.
 *
 * The word is "box" throughout, never "tile": in this product a tile is the
 * reading lattice drawn on a card, an independent unit. A box is a range of
 * elements and owes the lattice nothing.
 *
 * Overlap therefore has to be handled where quantities are measured rather than
 * where they are produced. Cardinality uses an exact recursive sweep without
 * constructing fragments; consumers that need drawable non-overlapping boxes
 * use `disjointify`. The rule is one sentence, and it is the one thing a new
 * consumer must know:
 *
 *     Cardinality is measured on the set. Boxes describe it; they do not
 *     count it. Summing box volumes double-counts the overlap.
 *
 * `regionOverlap` exposes that difference, because the elements two boxes share
 * are elements genuinely read twice.
 */

export type Interval = { lo: number; hi: number }; // half-open: lo <= i < hi
export type Box = Interval[]; // length === tensor rank
export type Region = {
  boxes: Box[];
  exact: boolean; // false => conservative over-approximation (strict superset allowed)
  reasons: string[]; // why it became inexact; empty when exact
};

export const MAX_BOXES = 256;

export function iv(lo: number, hi: number): Interval {
  return { lo, hi };
}

export function box(...pairs: [number, number][]): Box {
  return pairs.map(([lo, hi]) => iv(lo, hi));
}

/** The index syntax every readout shares: a bare index when the interval covers
 * one element, `lo:hi` otherwise. Delimiters belong to the caller - the
 * inspector field wraps it in `[]`, a slice expression prefixes the tensor name,
 * the hover readout uses `()` - but the terms are spelled once, because
 * `ui/selection-range.ts` parses this syntax back and a second speller would
 * drift from the parser rather than merely from another printer. */
export function formatBoxIndices(b: Box): string {
  return b.map((i) => (i.hi - i.lo === 1 ? `${i.lo}` : `${i.lo}:${i.hi}`)).join(", ");
}

export function empty(rank: number): Region {
  void rank;
  return { boxes: [], exact: true, reasons: [] };
}

export function full(shape: number[]): Region {
  return { boxes: [shape.map((n) => iv(0, n))], exact: true, reasons: [] };
}

export function fromBox(b: Box): Region {
  return canonicalize({ boxes: [b], exact: true, reasons: [] });
}

function isEmptyBox(b: Box): boolean {
  return b.some((i) => i.hi <= i.lo);
}

export function isEmpty(r: Region): boolean {
  return r.boxes.length === 0;
}

function boxVolume(b: Box): number {
  let v = 1;
  for (const i of b) v *= Math.max(0, i.hi - i.lo);
  return v;
}

function intersectBoxes(a: Box, b: Box): Box | null {
  const out: Box = [];
  for (let i = 0; i < a.length; i++) {
    const lo = Math.max(a[i].lo, b[i].lo);
    const hi = Math.min(a[i].hi, b[i].hi);
    if (hi <= lo) return null;
    out.push(iv(lo, hi));
  }
  return out;
}

/** a \ b as a list of disjoint boxes (possibly [a] when no overlap). */
export function subtractBox(a: Box, b: Box): Box[] {
  const inter = intersectBoxes(a, b);
  if (!inter) return [a];
  const pieces: Box[] = [];
  // Carve axis by axis; `core` shrinks toward the intersection.
  const core = a.map((i) => ({ ...i }));
  for (let ax = 0; ax < a.length; ax++) {
    if (core[ax].lo < inter[ax].lo) {
      const p = core.map((i) => ({ ...i }));
      p[ax] = iv(core[ax].lo, inter[ax].lo);
      pieces.push(p);
    }
    if (inter[ax].hi < core[ax].hi) {
      const p = core.map((i) => ({ ...i }));
      p[ax] = iv(inter[ax].hi, core[ax].hi);
      pieces.push(p);
    }
    core[ax] = { ...inter[ax] };
  }
  return pieces;
}

/**
 * Merge boxes identical on all axes but one and adjacent/overlapping there.
 * Repeats to fixpoint.
 *
 * Union-preserving whether or not the input is disjoint: the two boxes agree on
 * every other axis, so their union really is the single box spanning the
 * differing interval. That is what lets `canonicalize` use it on overlapping
 * input as well as `disjointify` on split input.
 */
function mergePass(boxes: Box[]): Box[] {
  let bs = boxes.slice();
  let changed = true;
  while (changed) {
    changed = false;
    outer: for (let i = 0; i < bs.length; i++) {
      for (let j = i + 1; j < bs.length; j++) {
        const a = bs[i],
          b = bs[j];
        let diffAxis = -1;
        let ok = true;
        for (let ax = 0; ax < a.length; ax++) {
          if (a[ax].lo === b[ax].lo && a[ax].hi === b[ax].hi) continue;
          if (diffAxis !== -1) {
            ok = false;
            break;
          }
          diffAxis = ax;
        }
        if (!ok) continue;
        if (diffAxis === -1) {
          // identical boxes
          bs.splice(j, 1);
          changed = true;
          break outer;
        }
        const ai = a[diffAxis],
          bi = b[diffAxis];
        if (ai.hi >= bi.lo && bi.hi >= ai.lo) {
          const merged = a.map((x) => ({ ...x }));
          merged[diffAxis] = iv(Math.min(ai.lo, bi.lo), Math.max(ai.hi, bi.hi));
          bs.splice(j, 1);
          bs.splice(i, 1);
          bs.push(merged);
          changed = true;
          break outer;
        }
      }
    }
  }
  return bs;
}

export function boundingBox(r: Region): Box | null {
  if (r.boxes.length === 0) return null;
  const rank = r.boxes[0].length;
  const out: Box = [];
  for (let ax = 0; ax < rank; ax++) {
    let lo = Infinity,
      hi = -Infinity;
    for (const b of r.boxes) {
      lo = Math.min(lo, b[ax].lo);
      hi = Math.max(hi, b[ax].hi);
    }
    out.push(iv(lo, hi));
  }
  return out;
}

function mergeReasons(a: string[], b: string[]): string[] {
  const s = new Set([...a, ...b]);
  return [...s];
}

function sameBox(a: Box, b: Box): boolean {
  return a.every((I, i) => I.lo === b[i].lo && I.hi === b[i].hi);
}

function boxContains(outer: Box, inner: Box): boolean {
  return inner.every((I, i) => I.lo >= outer[i].lo && I.hi <= outer[i].hi);
}

/** Drop exact duplicates, then any box wholly inside another. Both are
 * union-preserving: the dropped box contributes no element the survivor lacks.
 * Duplicates go first so two identical boxes cannot each eliminate the other. */
function dropContained(boxes: Box[]): Box[] {
  const uniq: Box[] = [];
  for (const b of boxes) if (!uniq.some((u) => sameBox(u, b))) uniq.push(b);
  return uniq.filter((b, i) => !uniq.some((o, j) => j !== i && boxContains(o, b)));
}

/**
 * Normalize a box list without splitting it.
 *
 * Both simplifications preserve the union exactly, so the represented set is
 * untouched and the exactness contract carries through unchanged. They run to a
 * fixpoint because a merge can expose a newly contained box and vice versa.
 *
 * The cap still exists, but it guards a different failure than it used to.
 * Splitting could multiply boxes combinatorially; accumulating boxes can only
 * add one per contribution, so this form reaches the cap later, not sooner.
 */
export function canonicalize(r: Region, maxBoxes: number = MAX_BOXES): Region {
  let boxes = r.boxes.filter((b) => !isEmptyBox(b));
  let exact = r.exact;
  let reasons = r.reasons.slice();
  let previous = -1;
  while (boxes.length !== previous) {
    previous = boxes.length;
    boxes = mergePass(dropContained(boxes));
  }
  if (boxes.length > maxBoxes) {
    const bb = boundingBox({ boxes, exact, reasons });
    boxes = bb ? [bb] : [];
    exact = false;
    reasons = mergeReasons(reasons, ["box count cap"]);
  }
  return { boxes, exact, reasons };
}

/**
 * Memo for the default-cap disjoint form.
 *
 * Splitting is the expensive direction - each box is subtracted from every box
 * already accepted, and the fragments accumulate - while callers ask for it
 * repeatedly on the *same* region when painting and enumerating. Regions are
 * immutable everywhere here (every operation returns a fresh one), so keying
 * on identity is sound. A caller that mutated `boxes` in place would see a
 * stale result, but the stored/disjoint distinction would already be
 * meaningless if anything did that.
 */
const disjointMemo = new WeakMap<Region, Region>();

/**
 * The disjoint form: split-on-overlap, then merge to a fixpoint.
 *
 * This is a rendering/iteration form, not the storage form. Every element
 * appears in exactly one box, so a consumer that needs explicit pieces can
 * process them without repeated compositing or enumeration. Cardinality does
 * not need those pieces and is computed by `count` directly.
 */
export function disjointify(r: Region, maxBoxes: number = MAX_BOXES): Region {
  if (maxBoxes !== MAX_BOXES) return splitOnOverlap(r, maxBoxes);
  const hit = disjointMemo.get(r);
  if (hit) return hit;
  const out = splitOnOverlap(r, MAX_BOXES);
  disjointMemo.set(r, out);
  // The disjoint form is its own answer, so a consumer that re-asks the form it
  // was just handed does not split it a second time.
  if (!disjointMemo.has(out)) disjointMemo.set(out, out);
  return out;
}

function splitOnOverlap(r: Region, maxBoxes: number): Region {
  // 1. drop empties
  let boxes = r.boxes.filter((b) => !isEmptyBox(b));
  let exact = r.exact;
  let reasons = r.reasons.slice();
  // 2. disjointify (split-on-overlap)
  const disjoint: Box[] = [];
  const softCap = maxBoxes * 8;
  let bailed = false;
  for (const b of boxes) {
    let frags: Box[] = [b];
    for (const d of disjoint) {
      const next: Box[] = [];
      for (const f of frags) next.push(...subtractBox(f, d));
      frags = next;
      if (frags.length === 0) break;
    }
    disjoint.push(...frags);
    if (disjoint.length > softCap) {
      bailed = true;
      break;
    }
  }
  boxes = bailed ? boxes : disjoint;
  // 3. merge to fixpoint; the split above guarantees the result stays disjoint
  if (!bailed) boxes = mergePass(boxes);
  // 4. cap
  if (bailed || boxes.length > maxBoxes) {
    const bb = boundingBox({ boxes, exact, reasons });
    boxes = bb ? [bb] : [];
    exact = false;
    reasons = mergeReasons(reasons, ["box count cap"]);
  }
  return { boxes, exact, reasons };
}

export function union(a: Region, b: Region): Region {
  return canonicalize({
    boxes: [...a.boxes, ...b.boxes],
    exact: a.exact && b.exact,
    reasons: mergeReasons(a.reasons, b.reasons),
  });
}

export function intersect(a: Region, b: Region): Region {
  const boxes: Box[] = [];
  for (const ba of a.boxes)
    for (const bb of b.boxes) {
      const x = intersectBoxes(ba, bb);
      if (x) boxes.push(x);
    }
  return canonicalize({
    boxes,
    exact: a.exact && b.exact,
    reasons: mergeReasons(a.reasons, b.reasons),
  });
}

export function subtract(a: Region, b: Region): Region {
  // `b` is a conservative superset when inexact. Subtracting that represented
  // superset could remove elements that are not in the true set and therefore
  // under-approximate the true difference. The only generally safe result is
  // the minuend itself: it may retain too much, but it cannot lose truth.
  if (!b.exact)
    return canonicalize({
      boxes: a.boxes,
      exact: false,
      reasons: mergeReasons(mergeReasons(a.reasons, b.reasons), ["inexact subtraction"]),
    });
  let frags: Box[] = a.boxes.slice();
  for (const bb of b.boxes) {
    const next: Box[] = [];
    for (const f of frags) next.push(...subtractBox(f, bb));
    frags = next;
  }
  return canonicalize({ boxes: frags, exact: a.exact && b.exact, reasons: a.reasons });
}

/**
 * True when some whole line along `axis` lies inside the region - the honest
 * form of "this cone pulls that axis in full".
 *
 * Deliberately not `boxes.some(box => box spans the axis)`. Boxes are no
 * longer split, so the `matmul(A, A)` case that used to defeat the naive test
 * now passes it - the row band survives as one box. But that is a case going
 * right, not a guarantee. Two boxes can still cover an axis between them
 * without either spanning it: overlap the boxes on the other axes and neither
 * one reaches end to end, while every line through the shared rows does. Only
 * merges of boxes that agree on every axis but one are performed, so nothing
 * fuses those two.
 *
 * Sweep the other axes until their active set is constant, then ask whether
 * the active intervals cover the target axis. This is the same dimensional
 * reduction used by `count`, with an existential result and early exit.
 */
export function coversAxisFully(r: Region, axis: number, extent: number): boolean {
  // An inexact region is a represented superset. Its intervals may bridge a
  // hole that exists in the true dependency set, so it cannot prove coverage.
  if (!r.exact || extent <= 0) return false;
  const boxes = r.boxes.filter((b) => !isEmptyBox(b));
  if (boxes.length === 0 || axis < 0 || axis >= boxes[0].length) return false;
  const otherAxes = Array.from({ length: boxes[0].length }, (_, i) => i)
    .filter((i) => i !== axis);
  return hasFullyCoveredLine(boxes, otherAxes, axis, extent);
}

function intervalsCoverExtent(boxes: Box[], axis: number, extent: number): boolean {
  const intervals = boxes
    .map((b) => b[axis])
    .sort((a, b) => a.lo - b.lo || b.hi - a.hi);
  let reached = 0;
  for (const interval of intervals) {
    if (interval.lo > reached) return false;
    if (interval.hi > reached) reached = interval.hi;
    if (reached >= extent) return true;
  }
  return false;
}

function hasFullyCoveredLine(
  boxes: Box[],
  axes: number[],
  targetAxis: number,
  extent: number
): boolean {
  if (boxes.length === 0) return false;
  if (axes.length === 0) return intervalsCoverExtent(boxes, targetAxis, extent);

  let sweepAxis = axes[0];
  let fewestEndpoints = Infinity;
  for (const candidate of axes) {
    const endpoints = new Set<number>();
    for (const b of boxes) {
      endpoints.add(b[candidate].lo);
      endpoints.add(b[candidate].hi);
    }
    if (endpoints.size < fewestEndpoints) {
      fewestEndpoints = endpoints.size;
      sweepAxis = candidate;
    }
  }

  type Events = { add: Box[]; remove: Box[] };
  const events = new Map<number, Events>();
  const at = (coordinate: number): Events => {
    let event = events.get(coordinate);
    if (!event) {
      event = { add: [], remove: [] };
      events.set(coordinate, event);
    }
    return event;
  };
  for (const b of boxes) {
    at(b[sweepAxis].lo).add.push(b);
    at(b[sweepAxis].hi).remove.push(b);
  }

  const coordinates = [...events.keys()].sort((a, b) => a - b);
  const active = new Set<Box>();
  const remaining = axes.filter((candidate) => candidate !== sweepAxis);
  let previous = coordinates[0];
  for (const coordinate of coordinates) {
    if (coordinate > previous && active.size &&
        hasFullyCoveredLine([...active], remaining, targetAxis, extent)) return true;
    const event = events.get(coordinate)!;
    for (const b of event.remove) active.delete(b);
    for (const b of event.add) active.add(b);
    previous = coordinate;
  }
  return false;
}

/** Exact cardinality of a union of boxes without materializing a partition.
 *
 * Sweep one axis between interval endpoints. Within each slab the active boxes
 * are constant, so its measure is the slab width times the union measure of
 * their projections. Choosing the axis with the fewest endpoints and stopping
 * at one box keeps the common sparse/crossing cases small. Unlike
 * `disjointify`, this never creates geometric fragments and therefore needs no
 * box-count fallback: it measures the represented set exactly even when its
 * cheapest disjoint description would be large.
 *
 * Cost is Klee's measure problem, and this is the naive recursive form: O(n^d)
 * in the worst case for n boxes of rank d. Measured on crossing families, rank
 * 2 stays near a millisecond at 256 boxes while rank 4 reaches ~140ms there.
 * Cones are one to three boxes in practice and `count` is memoized per region,
 * so the cliff is reachable only by a pathological selection on a rank-4
 * tensor. Anything that starts hitting it wants a sweep with a segment tree
 * rather than a wider fast path here. */
function unionVolume(boxes: Box[], axes: number[]): number {
  if (boxes.length === 0) return 0;
  if (axes.length === 0) return 1;
  if (boxes.length === 1) {
    let volume = 1;
    for (const axis of axes)
      volume *= Math.max(0, boxes[0][axis].hi - boxes[0][axis].lo);
    return volume;
  }
  if (axes.length === 1) {
    const axis = axes[0];
    const intervals = boxes
      .map((b) => b[axis])
      .sort((a, b) => a.lo - b.lo || b.hi - a.hi);
    let total = 0;
    let lo = intervals[0].lo;
    let hi = intervals[0].hi;
    for (let i = 1; i < intervals.length; i++) {
      const interval = intervals[i];
      if (interval.lo > hi) {
        total += hi - lo;
        lo = interval.lo;
        hi = interval.hi;
      } else if (interval.hi > hi) hi = interval.hi;
    }
    return total + hi - lo;
  }

  let sweepAxis = axes[0];
  let fewestEndpoints = Infinity;
  for (const axis of axes) {
    const endpoints = new Set<number>();
    for (const b of boxes) {
      endpoints.add(b[axis].lo);
      endpoints.add(b[axis].hi);
    }
    if (endpoints.size < fewestEndpoints) {
      fewestEndpoints = endpoints.size;
      sweepAxis = axis;
    }
  }

  type Events = { add: Box[]; remove: Box[] };
  const events = new Map<number, Events>();
  const at = (coordinate: number): Events => {
    let event = events.get(coordinate);
    if (!event) {
      event = { add: [], remove: [] };
      events.set(coordinate, event);
    }
    return event;
  };
  for (const b of boxes) {
    at(b[sweepAxis].lo).add.push(b);
    at(b[sweepAxis].hi).remove.push(b);
  }

  const coordinates = [...events.keys()].sort((a, b) => a - b);
  const active = new Set<Box>();
  const remaining = axes.filter((axis) => axis !== sweepAxis);
  let total = 0;
  let previous = coordinates[0];
  for (const coordinate of coordinates) {
    if (coordinate > previous && active.size)
      total += (coordinate - previous) * unionVolume([...active], remaining);
    const event = events.get(coordinate)!;
    for (const b of event.remove) active.delete(b);
    for (const b of event.add) active.add(b);
    previous = coordinate;
  }
  return total;
}

const countMemo = new WeakMap<Region, number>();

export function count(r: Region): number {
  const hit = countMemo.get(r);
  if (hit !== undefined) return hit;
  const boxes = r.boxes.filter((b) => !isEmptyBox(b));
  const rank = boxes[0]?.length ?? 0;
  const result = unionVolume(boxes, Array.from({ length: rank }, (_, axis) => axis));
  countMemo.set(r, result);
  return result;
}

/** @internal Exhaustive test oracle; never call on application-sized regions. */
export function* points(r: Region): Generator<number[]> {
  const c = disjointify(r);
  for (const b of c.boxes) {
    const rank = b.length;
    const idx = b.map((i) => i.lo);
    if (b.some((i) => i.hi <= i.lo)) continue;
    while (true) {
      yield idx.slice();
      let ax = rank - 1;
      while (ax >= 0) {
        idx[ax]++;
        if (idx[ax] < b[ax].hi) break;
        idx[ax] = b[ax].lo;
        ax--;
      }
      if (ax < 0) break;
    }
  }
}

export function markInexact(r: Region, reason: string): Region {
  return { boxes: r.boxes, exact: false, reasons: mergeReasons(r.reasons, [reason]) };
}

// ------------------------------------------------------ selections vs regions
//
// Regions and selections now share one shape: an ordered list of boxes that may
// overlap, with set questions answered on demand. They differ in what the boxes
// mean and in how much identity survives.
//
// A user's selection is a list of PARTS, each with a durable identity - its hue,
// its row in the inspector, its own dependency cone, and the ability to be moved
// or deleted on its own. Parts are never normalized at all; two parts may be
// identical and both remain.
//
// A Region is a list of TILES. Their identity is weaker: canonicalize() merges
// boxes that agree on every axis but one and drops boxes contained in another,
// so a box can disappear into a neighbour. What it will not do is split one,
// which is what used to make a box mean nothing at all.
//
// Both stay consistent with measurement the same way: every *set* question goes
// through disjointify() or count() first, so element totals, propagation seeds
// and metrics deduplicate overlap no matter which list they were handed.

/** Append a drawn box as a new part, ignoring an exact duplicate. */
export function addPart(parts: Box[], b: Box): Box[] {
  if (isEmptyBox(b)) return parts;
  const same = (x: Box, y: Box) =>
    x.length === y.length && x.every((I, i) => I.lo === y[i].lo && I.hi === y[i].hi);
  return parts.some((p) => same(p, b)) ? parts : [...parts, b];
}

/**
 * Cut a box out of every part. A part may vanish or split into several; a split
 * necessarily loses that part's identity, since one region becomes many.
 */
export function subtractFromParts(parts: Box[], b: Box): Box[] {
  const out: Box[] = [];
  for (const p of parts) out.push(...subtractBox(p, b));
  return out.filter((p) => !isEmptyBox(p));
}

/**
 * Move a single part along one axis, leaving every other part untouched. The
 * delta is clamped so the moved part stays inside the tensor - pushing into an
 * edge stops there rather than eroding it. Parts may overlap after the move;
 * that is legal and is resolved by canonicalize() wherever a set is needed.
 */
export function translatePart(
  parts: Box[],
  index: number,
  axis: number,
  delta: number,
  shape: number[]
): Box[] {
  const p = parts[index];
  if (!p || delta === 0) return parts;
  const d = Math.max(-p[axis].lo, Math.min(delta, shape[axis] - p[axis].hi));
  if (d === 0) return parts;
  const moved = p.map((I, ax) => (ax === axis ? iv(I.lo + d, I.hi + d) : { ...I }));
  return parts.map((q, i) => (i === index ? moved : q));
}

/** Move every part together, clamped so the whole selection stays in bounds. */
export function translateAllParts(
  parts: Box[],
  axis: number,
  delta: number,
  shape: number[]
): Box[] {
  if (!parts.length || delta === 0) return parts;
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of parts) {
    lo = Math.min(lo, p[axis].lo);
    hi = Math.max(hi, p[axis].hi);
  }
  const d = Math.max(-lo, Math.min(delta, shape[axis] - hi));
  if (d === 0) return parts;
  return parts.map((p) =>
    p.map((I, ax) => (ax === axis ? iv(I.lo + d, I.hi + d) : { ...I }))
  );
}

/** Elements counted once, versus the sum of the boxes' own volumes. The gap is
 * the overlap with multiplicity: for `matmul(A, A)` it is exactly the square
 * both operand bands read, so it is a real quantity rather than an artefact. */
export function partsOverlap(parts: Box[]): { unique: number; summed: number } {
  const summed = parts.reduce((a, p) => a + boxVolume(p), 0);
  return { unique: count({ boxes: parts, exact: true, reasons: [] }), summed };
}

/** `partsOverlap` for a region's boxes. Zero gap means the boxes are disjoint.
 *
 * Counts through `count(r)` on the region itself rather than rebuilding one
 * from its boxes: a caller asking for elements and overlap together then pays
 * for one union sweep, because the cardinality memo is keyed on identity. */
export function regionOverlap(r: Region): { unique: number; summed: number } {
  let summed = 0;
  for (const b of r.boxes) summed += boxVolume(b);
  return { unique: count(r), summed };
}

/** Deterministic ordering, used for byte-identical output & tests. */
export function sortRegion(r: Region): Region {
  const boxes = r.boxes.slice().sort((a, b) => {
    for (let ax = 0; ax < a.length; ax++) {
      if (a[ax].lo !== b[ax].lo) return a[ax].lo - b[ax].lo;
      if (a[ax].hi !== b[ax].hi) return a[ax].hi - b[ax].hi;
    }
    return 0;
  });
  return { boxes, exact: r.exact, reasons: r.reasons.slice() };
}
