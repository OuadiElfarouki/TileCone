/**
 * Region algebra: unions of half-open axis-aligned boxes in tensor index space.
 *
 * Representation choice (see IDEA.md §2.1): `canonicalize` makes the box set
 * DISJOINT via split-on-overlap, then merges adjacent boxes to a fixpoint.
 * `count` therefore sums volumes directly, and `contains` is a linear scan.
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

export function isEmptyBox(b: Box): boolean {
  return b.some((i) => i.hi <= i.lo);
}

export function isEmpty(r: Region): boolean {
  return r.boxes.length === 0;
}

export function boxVolume(b: Box): number {
  let v = 1;
  for (const i of b) v *= Math.max(0, i.hi - i.lo);
  return v;
}

export function intersectBoxes(a: Box, b: Box): Box | null {
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

/** Merge boxes identical on all axes but one and adjacent/overlapping there. Repeats to fixpoint. */
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

export function canonicalize(r: Region, maxBoxes: number = MAX_BOXES): Region {
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
  // 3. merge to fixpoint (only valid on disjoint set)
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
  let frags: Box[] = a.boxes.slice();
  for (const bb of b.boxes) {
    const next: Box[] = [];
    for (const f of frags) next.push(...subtractBox(f, bb));
    frags = next;
  }
  return canonicalize({ boxes: frags, exact: a.exact && b.exact, reasons: a.reasons });
}

/** Number of distinct elements. Canonicalizes internally so boxes are disjoint. */
export function count(r: Region): number {
  const c = canonicalize(r);
  let n = 0;
  for (const b of c.boxes) n += boxVolume(b);
  return n;
}

export function contains(r: Region, point: number[]): boolean {
  for (const b of r.boxes) {
    let inside = true;
    for (let ax = 0; ax < b.length; ax++) {
      if (point[ax] < b[ax].lo || point[ax] >= b[ax].hi) {
        inside = false;
        break;
      }
    }
    if (inside) return true;
  }
  return false;
}

/** Enumerate all points of a region (test-sized regions only). */
export function* points(r: Region): Generator<number[]> {
  const c = canonicalize(r);
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

// -------------------------------------------------------- editing operations
// Used by the UI to manipulate a selection directly. All clamp to `shape` and
// preserve the region's shape where possible rather than clipping it away.

/**
 * Shift the whole region along `axis`. The delta is clamped so the region's
 * bounding box stays inside the tensor — moving into the edge stops there
 * rather than eroding boxes.
 */
export function translateRegion(
  r: Region,
  axis: number,
  delta: number,
  shape: number[]
): Region {
  const bb = boundingBox(r);
  if (!bb || delta === 0) return r;
  const d = Math.max(-bb[axis].lo, Math.min(delta, shape[axis] - bb[axis].hi));
  if (d === 0) return r;
  return canonicalize({
    boxes: r.boxes.map((b) =>
      b.map((I, ax) => (ax === axis ? iv(I.lo + d, I.hi + d) : { ...I }))
    ),
    exact: r.exact,
    reasons: r.reasons,
  });
}

/** Drop one box by index (used by the inspector's per-box delete). */
export function removeBoxAt(r: Region, index: number): Region {
  return canonicalize({
    boxes: r.boxes.filter((_, i) => i !== index),
    exact: r.exact,
    reasons: r.reasons,
  });
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
  return { boxes, exact: r.exact, reasons: r.reasons.slice().sort() };
}
