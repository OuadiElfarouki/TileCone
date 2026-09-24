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

export type Interval = Readonly<{ lo: number; hi: number }>; // half-open: lo <= i < hi
export type Box = readonly Interval[]; // length === tensor rank
export type Region = Readonly<{
  boxes: readonly Box[];
  exact: boolean; // false => conservative over-approximation (strict superset allowed)
  reasons: readonly string[]; // why it became inexact; empty when exact
}>;

type MutableInterval = { lo: number; hi: number };
type MutableBox = MutableInterval[];

const immutableRegions = new WeakSet<Region>();

/** Regions are cached by identity in a few hot paths. Make every region
 * produced by this module deeply immutable so that identity remains a sound
 * cache key; foreign mutable region objects are still accepted, but are never
 * memoized. Cloning before freezing also avoids freezing a caller's box. */
function immutableRegion(r: Region): Region {
  const boxes = r.boxes.map((b) => {
    const copy = b.map((interval) => Object.freeze({ ...interval }));
    return Object.freeze(copy) as Box;
  });
  const reasons = Object.freeze(r.reasons.slice());
  const out = Object.freeze({
    boxes: Object.freeze(boxes),
    exact: r.exact,
    reasons,
  }) as Region;
  immutableRegions.add(out);
  return out;
}

const isImmutableRegion = (r: Region): boolean => immutableRegions.has(r);

/** Regions already in canonical form at the default cap, so `canonicalize` can
 * recognise its own output and return it untouched. Membership is a fact about
 * a region this module produced, never a claim about a foreign one. */
const canonicalRegions = new WeakSet<Region>();

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
  return immutableRegion({ boxes: [], exact: true, reasons: [] });
}

export function full(shape: number[]): Region {
  return canonicalize({ boxes: [shape.map((n) => iv(0, n))], exact: true, reasons: [] });
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
  const out: MutableBox = [];
  for (let i = 0; i < a.length; i++) {
    const lo = Math.max(a[i].lo, b[i].lo);
    const hi = Math.min(a[i].hi, b[i].hi);
    if (hi <= lo) return null;
    out.push(iv(lo, hi));
  }
  return out;
}

/** Whether two boxes share any element. Cheaper than intersecting them, for
 * callers that only need to know whether there is anything to do. */
function boxesOverlap(a: Box, b: Box): boolean {
  for (let i = 0; i < a.length; i++)
    if (Math.min(a[i].hi, b[i].hi) <= Math.max(a[i].lo, b[i].lo)) return false;
  return true;
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
 * A box keyed by every axis except `skip`.
 *
 * Two boxes can merge along an axis exactly when they agree on every other one,
 * so this key is what puts the merge candidates together. Grouping on it turns
 * the search for a mergeable pair from a scan over all pairs into a lookup.
 */
function axisGroupKey(b: Box, skip: number): string {
  let key = "";
  for (let ax = 0; ax < b.length; ax++) {
    if (ax === skip) continue;
    key += `${b[ax].lo}:${b[ax].hi};`;
  }
  return key;
}

/**
 * Merge every run of boxes that agree on all axes but `axis` and touch or
 * overlap along it.
 *
 * Union-preserving whether or not the input is disjoint: the boxes agree on
 * every other axis, so their union really is the single box spanning the
 * combined interval. That is what lets `canonicalize` use this on overlapping
 * input as well as `disjointify` on split input.
 *
 * Group, sort, sweep - so one pass costs a sort rather than a scan over all
 * pairs. The previous form compared every pair and restarted from the first box
 * after each merge, which made a region of a few thousand boxes quadratic in
 * the best case and cubic when the boxes did merge. Both are reachable: a
 * reshape decomposes a tile into up to `reshapeRuns` contiguous runs, and a
 * strided op enumerates up to `stridedEnum` positions, all before the box cap
 * is applied.
 *
 * Nothing is mutated; a widened box is a fresh box, so a caller's boxes are
 * safe to pass in.
 */
function mergeAlongAxis(boxes: Box[], axis: number): Box[] {
  if (boxes.length < 2) return boxes;
  const groups = new Map<string, Box[]>();
  for (const b of boxes) {
    const key = axisGroupKey(b, axis);
    const group = groups.get(key);
    if (group) group.push(b);
    else groups.set(key, [b]);
  }
  const out: Box[] = [];
  for (const group of groups.values()) {
    if (group.length === 1) {
      out.push(group[0]);
      continue;
    }
    group.sort((x, y) => x[axis].lo - y[axis].lo || x[axis].hi - y[axis].hi);
    let current = group[0];
    for (let i = 1; i < group.length; i++) {
      const next = group[i];
      // Sorted by `lo`, so this is the whole adjacency test: anything that
      // starts at or before the current end extends it, and anything that
      // starts after it opens a new run. A box wholly inside the current one
      // leaves it unchanged, which is how duplicates disappear.
      if (next[axis].lo <= current[axis].hi) {
        if (next[axis].hi > current[axis].hi) {
          const widened = current.slice();
          widened[axis] = iv(current[axis].lo, next[axis].hi);
          current = widened;
        }
        continue;
      }
      out.push(current);
      current = next;
    }
    out.push(current);
  }
  return out;
}

/** Merge along every axis until no axis has anything left to merge: a merge on
 * one axis can line two boxes up on another. */
function mergeBoxes(boxes: Box[]): Box[] {
  if (boxes.length < 2) return boxes;
  // Rank 0 is the scalar case: every box is the same single point, so one of
  // them represents the union and there is no axis to sweep.
  if (boxes[0].length === 0) return boxes.slice(0, 1);
  let current = boxes;
  for (;;) {
    const before = current.length;
    for (let axis = 0; axis < current[0].length; axis++)
      current = mergeAlongAxis(current, axis);
    if (current.length === before) return current;
  }
}

/** The box covering both, which is what merging them costs you. */
function hull(a: Box, b: Box): Box {
  const out: MutableBox = [];
  for (let ax = 0; ax < a.length; ax++)
    out.push(iv(Math.min(a[ax].lo, b[ax].lo), Math.max(a[ax].hi, b[ax].hi)));
  return out;
}

/**
 * Elements the hull adds beyond the two boxes themselves.
 *
 * An upper bound on the waste, not the exact figure: it charges for the overlap
 * twice when the boxes intersect, which only ever makes a merge look worse than
 * it is. Computing the true union here would cost a measure sweep per candidate
 * pair, and the ranking barely changes - overlapping pairs are cheap either way.
 */
function mergeWaste(a: Box, b: Box): number {
  let hullVolume = 1;
  const h = hull(a, b);
  for (const i of h) hullVolume *= i.hi - i.lo;
  return hullVolume - boxVolume(a) - boxVolume(b);
}

/** Minimal binary heap over (cost, entry), enough for the coarsener. */
type PairEntry = { cost: number; left: number; right: number; stampL: number; stampR: number };

function heapPush(heap: PairEntry[], entry: PairEntry): void {
  heap.push(entry);
  let i = heap.length - 1;
  while (i > 0) {
    const parent = (i - 1) >> 1;
    if (heap[parent].cost <= heap[i].cost) break;
    [heap[parent], heap[i]] = [heap[i], heap[parent]];
    i = parent;
  }
}

function heapPop(heap: PairEntry[]): PairEntry | undefined {
  if (heap.length === 0) return undefined;
  const top = heap[0];
  const last = heap.pop()!;
  if (heap.length) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let small = i;
      if (l < heap.length && heap[l].cost < heap[small].cost) small = l;
      if (r < heap.length && heap[r].cost < heap[small].cost) small = r;
      if (small === i) break;
      [heap[small], heap[i]] = [heap[i], heap[small]];
      i = small;
    }
  }
  return top;
}

/**
 * Reduce a box list to at most `maxBoxes` by merging neighbours into their
 * hulls, cheapest first.
 *
 * This replaces collapsing straight to one bounding box, which was correct but
 * threw away everything: a 300-wide diagonal reported the whole 300x300 matrix,
 * 300 times its own size. Merging pairwise keeps the shape of the set, so the
 * same case comes back within a few hundred elements of the truth.
 *
 * Always a superset: a hull contains both boxes it replaces, so no element is
 * ever lost. The caller marks the result inexact.
 *
 * Candidates are adjacent pairs in lexicographic order by lower corner, not all
 * pairs. All-pairs greedy is cubic and unusable at the sizes this guards, while
 * the orders that actually reach the cap - a strided slice's evenly spaced
 * runs, a diagonal's staircase - are exactly the orders lexicographic sorting
 * puts next to each other.
 */
export function coarsen(boxes: Box[], maxBoxes: number): Box[] {
  if (boxes.length <= maxBoxes || maxBoxes < 1) return boxes;
  const items = boxes.slice().sort((a, b) => {
    for (let ax = 0; ax < a.length; ax++) {
      if (a[ax].lo !== b[ax].lo) return a[ax].lo - b[ax].lo;
      if (a[ax].hi !== b[ax].hi) return a[ax].hi - b[ax].hi;
    }
    return 0;
  });

  const prev = items.map((_, i) => i - 1);
  const next = items.map((_, i) => (i === items.length - 1 ? -1 : i + 1));
  const alive = items.map(() => true);
  // Bumped whenever a box is merged into, so heap entries naming an older
  // version of it can be discarded on pop instead of deleted on merge.
  const stamp = items.map(() => 0);

  const heap: PairEntry[] = [];
  const offer = (left: number, right: number) => {
    if (left < 0 || right < 0) return;
    heapPush(heap, {
      cost: mergeWaste(items[left], items[right]),
      left,
      right,
      stampL: stamp[left],
      stampR: stamp[right],
    });
  };
  for (let i = 0; i < items.length - 1; i++) offer(i, i + 1);

  let count = items.length;
  while (count > maxBoxes) {
    const entry = heapPop(heap);
    if (!entry) break; // no candidates left; the caller's cap still applies
    const { left, right } = entry;
    if (!alive[left] || !alive[right]) continue;
    if (entry.stampL !== stamp[left] || entry.stampR !== stamp[right]) continue;

    items[left] = hull(items[left], items[right]);
    stamp[left]++;
    alive[right] = false;
    next[left] = next[right];
    if (next[right] >= 0) prev[next[right]] = left;
    count--;
    offer(prev[left], left);
    offer(left, next[left]);
  }
  return items.filter((_, i) => alive[i]);
}

export function boundingBox(r: Region): Box | null {
  if (r.boxes.length === 0) return null;
  const rank = r.boxes[0].length;
  const out: MutableBox = [];
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

function mergeReasons(a: readonly string[], b: readonly string[]): string[] {
  const s = new Set([...a, ...b]);
  return [...s];
}

function boxContains(outer: Box, inner: Box): boolean {
  return inner.every((I, i) => I.lo >= outer[i].lo && I.hi <= outer[i].hi);
}

/**
 * Drop every box wholly inside another, exact duplicates included.
 *
 * Union-preserving: a dropped box contributes no element its container lacks.
 *
 * Sorting by lower corner ascending and upper corner descending puts a
 * container strictly before anything it contains - on the first axis where the
 * two differ, a container starts no later, and when they start together it ends
 * no earlier - so testing each box against the boxes already kept is enough to
 * find every containment. It also makes the result independent of the order the
 * boxes arrived in, which is what keeps `canonicalize` idempotent now that the
 * merge step above regroups them.
 *
 * Duplicates need no separate pass for the same reason: identical boxes are
 * adjacent after the sort, and the second is contained in the first, so one
 * survives rather than each eliminating the other.
 */
function dropContained(boxes: Box[]): Box[] {
  if (boxes.length < 2) return boxes;
  const rank = boxes[0].length;
  if (rank === 0) return boxes.slice(0, 1);
  const order = boxes.slice().sort((a, b) => {
    for (let ax = 0; ax < rank; ax++) {
      if (a[ax].lo !== b[ax].lo) return a[ax].lo - b[ax].lo;
      if (a[ax].hi !== b[ax].hi) return b[ax].hi - a[ax].hi;
    }
    return 0;
  });

  const kept: Box[] = [];
  // Boxes that may still contain something later. Sorted by `lo` on axis 0, a
  // candidate ending at or before the current box's start cannot contain it or
  // anything after it, so dropping it here is permanent and the scan stays
  // near-linear on the disjoint families that reach the cap.
  let candidates: Box[] = [];
  for (const b of order) {
    // Every test is made rather than budgeted. Stopping early would leave boxes
    // that could have been dropped, which is harmless on its own - but it can
    // carry a region past the box cap, and that cap is an approximation. A
    // simplification must not be what decides whether the answer is exact.
    candidates = candidates.filter((c) => c[0].hi > b[0].lo);
    if (candidates.some((c) => boxContains(c, b))) continue;
    kept.push(b);
    candidates.push(b);
  }
  return kept;
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
  // Already in this exact form: propagation canonicalizes a seed the executor
  // has canonicalized, unions a region that is already normal, and so on, and
  // at a few hundred boxes each of those passes is real work. A region is only
  // in this set if it came out of here, so the form is known rather than
  // assumed. A caller asking for a different cap is asking a different
  // question and gets the full pass.
  if (maxBoxes === MAX_BOXES && canonicalRegions.has(r)) return r;
  // No defensive copy: nothing below mutates a box, and `immutableRegion`
  // rebuilds every interval it freezes, so a caller's boxes are never touched.
  let boxes: Box[] = r.boxes.filter((b) => !isEmptyBox(b));
  let exact = r.exact;
  let reasons = r.reasons.slice();
  let previous = -1;
  while (boxes.length !== previous) {
    previous = boxes.length;
    boxes = mergeBoxes(dropContained(boxes));
  }
  if (boxes.length > maxBoxes) {
    // Coarsen rather than collapse. Both are supersets, but one bounding box
    // discards the shape of the set entirely, and the shape is what the reader
    // is looking at.
    boxes = coarsen(boxes, maxBoxes);
    exact = false;
    reasons = mergeReasons(reasons, ["box count cap"]);
  }
  const out = immutableRegion({ boxes, exact, reasons });
  if (maxBoxes === MAX_BOXES) canonicalRegions.add(out);
  return out;
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
  if (maxBoxes !== MAX_BOXES) return immutableRegion(splitOnOverlap(r, maxBoxes));
  const hit = disjointMemo.get(r);
  if (hit) return hit;
  const out = immutableRegion(splitOnOverlap(r, MAX_BOXES));
  if (isImmutableRegion(r)) disjointMemo.set(r, out);
  // The disjoint form is its own answer, so a consumer that re-asks the form it
  // was just handed does not split it a second time.
  if (!disjointMemo.has(out)) disjointMemo.set(out, out);
  return out;
}

/**
 * Work a split may do before giving up, counted in `subtractBox` calls.
 *
 * A fragment cap alone does not bound the time. Splitting is quadratic in the
 * boxes and exponential in the rank - `subtractBox` yields up to `2 * rank`
 * pieces per overlap - so a crossing family can spend seconds *approaching* the
 * fragment cap and reach it only at the end. Counting the calls bounds the wall
 * clock directly, which is what an interactive caller actually needs.
 */
const SPLIT_WORK_BUDGET = 200_000;

/** Split-on-overlap, or null when it outgrows the fragment or work budget. */
function trySplit(boxes: Box[], softCap: number): Box[] | null {
  const disjoint: Box[] = [];
  let work = 0;
  for (const b of boxes) {
    let frags: Box[] = [b];
    for (const d of disjoint) {
      const next: Box[] = [];
      for (const f of frags) next.push(...subtractBox(f, d));
      work += frags.length;
      frags = next;
      if (frags.length === 0) break;
    }
    if (work > SPLIT_WORK_BUDGET) return null;
    disjoint.push(...frags);
    if (disjoint.length > softCap) return null;
  }
  return mergeBoxes(disjoint);
}

function splitOnOverlap(r: Region, maxBoxes: number): Region {
  let source = r.boxes.filter((b) => !isEmptyBox(b));
  const softCap = maxBoxes * 8;

  const direct = trySplit(source, softCap);
  if (direct && direct.length <= maxBoxes)
    return { boxes: direct, exact: r.exact, reasons: r.reasons.slice() };

  // Splitting blew up, or produced more pieces than the cap allows. Coarsen the
  // *input* and split that, rather than collapsing the answer to one box: at
  // rank 3 and 4 a family of crossing boxes reliably defeated the split, and a
  // bounding box there reported several times the set it was describing.
  //
  // Coarsening well below the cap is deliberate. The split is what grows the
  // count, so the input has to leave room for it, and a second failure costs
  // another full attempt.
  source = coarsen(source, Math.max(1, maxBoxes >> 3));
  const reasons = mergeReasons(r.reasons, ["box count cap"]);
  const retry = trySplit(source, softCap);
  if (retry && retry.length <= maxBoxes) return { boxes: retry, exact: false, reasons };

  const bb = boundingBox({ boxes: source, exact: false, reasons });
  return { boxes: bb ? [bb] : [], exact: false, reasons };
}

export function union(a: Region, b: Region): Region {
  return canonicalize({
    boxes: [...a.boxes, ...b.boxes],
    exact: a.exact && b.exact,
    reasons: mergeReasons(a.reasons, b.reasons),
  });
}

/**
 * Every pairwise intersection, with no shortcut for a fine result.
 *
 * Two regions of N boxes can meet in N^2 pieces, and families of stripes
 * genuinely do: 256 row bands crossed with 256 column bands is 65536 distinct
 * cells. Building all of them is the only way to say exactly what the two
 * regions share, so that is what happens - the box cap is the one place this
 * module is allowed to lose precision, and it applies afterwards to the true
 * set rather than in place of computing it.
 */
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
  // Each subtrahend box can cut every fragment into up to `2 * rank` pieces, so
  // the fragments multiply. They are carried anyway rather than bounded: a
  // difference is what the contribution verdict is read off, and the only
  // shortcut available here would keep elements the tile genuinely supplies.
  let frags: Box[] = a.boxes.slice();
  for (const bb of b.boxes) {
    const next: Box[] = [];
    for (const f of frags) {
      // A fragment this box does not touch survives as itself. Worth testing
      // first: the fragments narrow as they are cut, so most of these pairs
      // miss, and `subtractBox` would allocate a result array to say so.
      if (!boxesOverlap(f, bb)) next.push(f);
      else next.push(...subtractBox(f, bb));
    }
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
  const cacheable = isImmutableRegion(r);
  const hit = cacheable ? countMemo.get(r) : undefined;
  if (hit !== undefined) return hit;
  const boxes = r.boxes.filter((b) => !isEmptyBox(b));
  const rank = boxes[0]?.length ?? 0;
  const result = unionVolume(boxes, Array.from({ length: rank }, (_, axis) => axis));
  if (cacheable) countMemo.set(r, result);
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

export function markInexact(r: Region, ...reasons: string[]): Region {
  return immutableRegion({ boxes: r.boxes, exact: false, reasons: mergeReasons(r.reasons, reasons) });
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
  return immutableRegion({ boxes, exact: r.exact, reasons: r.reasons.slice() });
}
