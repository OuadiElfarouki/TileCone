import { type Box, canonicalize, isEmpty, type Region, subtractBox } from "../../core/region";
import { boxColor } from "../palette";
import { Layer } from "./grid";
import { ReplayPlayback } from "../workspace";

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

/**
 * Paint a family replay over the plan. It never substitutes for the plan's own
 * paint, and the plan is unchanged underneath it.
 *
 * On the family's tensor the runs walked so far stay as a faint trail, and the
 * active run is drawn strongly. The runs are contiguous in listing order, so
 * the trail is one growing stretch of the grid with no gaps, and it is cleared
 * all at once when the replay ends. Everywhere else only the active run is
 * drawn: what it reads or computes, in the solid fill the Plan view gives the
 * inspected task's demand, with the part the inspected task demands too at
 * full strength over the rest. Painting only the overlap would leave a run
 * that shares nothing painting nothing.
 *
 * It draws no lattice: the plan's lattice is already on every tiled card.
 */
export function buildReplayPaint({
  tensorId,
  dark,
  playback,
}: {
  tensorId: string;
  dark: boolean;
  playback: ReplayPlayback;
}): { layers: Layer[] } {
  const color = boxColor(0, dark);
  const frames = playback.replay.frames.slice(0, playback.visited);
  const active = frames[frames.length - 1];
  const layers: Layer[] = [];
  if (!active) return { layers };

  if (tensorId === playback.replay.task.tensorId) {
    for (const frame of frames.slice(0, -1))
      layers.push({ region: frame.region, color, alpha: 0.1 * playback.opacity, hatch: false });
    layers.push({ region: active.region, color, alpha: 0.68 * playback.opacity, hatch: false, seed: true });
    return { layers };
  }

  const reached = active.reach.get(tensorId);
  if (!reached) return { layers };
  const { region, shared } = reached;
  /* Cache the display difference across fade ticks. Its splitting budget
     keeps fragmented geometry bounded; on exhaustion the full reach remains
     visible and overlap may darken. */
  const rest = shared ? playbackDifference(region, shared) : region;
  if (!isEmpty(rest))
    layers.push({ region: rest, color, alpha: 0.26 * playback.opacity, hatch: !rest.exact });
  if (shared)
    layers.push({ region: shared, color, alpha: 0.62 * playback.opacity, hatch: !shared.exact, seed: true });
  return { layers };
}
