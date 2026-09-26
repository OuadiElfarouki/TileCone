import { canonicalize, subtractBox, type Box, type Region } from "../core/region";

const differences = new WeakMap<Region, WeakMap<Region, Region>>();
const MAX_PAINT_FRAGMENTS = 256;
const MAX_PAINT_PAIRS = 2048;

/** Display-only subtraction. Keep the full reach when splitting would exceed
 * the paint budget; overlap may darken, but no reached area disappears. Cache
 * by immutable geometry so fade ticks and replay do no further splitting. */
export function playbackDifference(reach: Region, shared: Region): Region {
  const cached = differences.get(reach)?.get(shared);
  if (cached) return cached;
  let result: Region;
  if (!shared.exact) {
    result = { ...reach, exact: false, reasons: [...new Set([
      ...reach.reasons, ...shared.reasons, "inexact playback subtraction",
    ])] };
  } else {
    let fragments = reach.boxes;
    let pairs = 0;
    let exceeded = fragments.length > MAX_PAINT_FRAGMENTS;
    outer: for (const cut of shared.boxes) {
      if (exceeded) break;
      const next: Box[] = [];
      for (const fragment of fragments) {
        if (++pairs > MAX_PAINT_PAIRS) { exceeded = true; break outer; }
        next.push(...subtractBox(fragment, cut));
        if (next.length > MAX_PAINT_FRAGMENTS) { exceeded = true; break outer; }
      }
      fragments = next;
    }
    result = exceeded
      ? { ...reach, exact: false, reasons: [...reach.reasons, "playback subtraction budget"] }
      : canonicalize({ boxes: fragments, exact: reach.exact, reasons: reach.reasons });
  }
  const byShared = differences.get(reach) ?? new WeakMap<Region, Region>();
  byShared.set(shared, result);
  differences.set(reach, byShared);
  return result;
}
