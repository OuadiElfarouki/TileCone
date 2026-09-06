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
 * where they are produced. `disjointify` is that form, and every measurement
 * goes through it: `count`, `points`, the FLOP sum, and the canvas fill. The
 * rule is one sentence, and it is the one thing a new consumer must know:
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
 * The disjoint form: split-on-overlap, then merge to a fixpoint.
 *
 * This is the measurement form, not the storage form. Every element appears in
 * exactly one box, so volumes may be summed - which is the only reason it
 * exists. Callers that draw, name, or attribute regions want the boxes instead;
 * callers that count anything want this.
 */
export function disjointify(r: Region, maxBoxes: number = MAX_BOXES): Region {
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

/** Number of distinct elements. Disjointifies internally, so overlapping boxes\n * contribute their shared elements once. */
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
 * Anchoring a unit line at each box's lower corner and asking whether the
 * region contains all of it proves containment against the union, so it can
 * never claim a pull the cone did not make.
 */
export function coversAxisFully(r: Region, axis: number, extent: number): boolean {
  for (const b of r.boxes) {
    const line: Box = b.map((interval, i) =>
      i === axis ? { lo: 0, hi: extent } : { lo: interval.lo, hi: interval.lo + 1 }
    );
    if (isEmpty(subtract(fromBox(line), r))) return true;
  }
  return false;
}

export function count(r: Region): number {
  const c = disjointify(r);
  let n = 0;
  for (const b of c.boxes) n += boxVolume(b);
  return n;
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

/** `partsOverlap` for a region's boxes. Zero gap means the boxes are disjoint. */
export function regionOverlap(r: Region): { unique: number; summed: number } {
  return partsOverlap(r.boxes);
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
