import { type Box, canonicalize, fromBox, isEmpty, type Region, subtractBox } from "../../core/region";
import type { ReplayFrame } from "../../core/plan/replay";
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
 * A union over the visited frames, computed once per replay rather than once
 * per repaint.
 *
 * `union` canonicalizes to a fixpoint on every call, so folding forty-eight
 * frames pairwise is forty-seven of them for one answer - inside the paint
 * effect, redone on every dependency that repaints the card. One canonicalize
 * over the concatenated boxes represents the same set, with the same exactness
 * and reasons.
 *
 * Keyed on the frames array, which a playback only replaces with a new replay,
 * so ticking `visited`, settling, and fading all reuse the entry. A `WeakMap`
 * keeps it bounded without anyone having to retire it.
 */
const settledUnions = new WeakMap<ReplayFrame[], Map<string, Region | null>>();

function unionAcrossFrames(
  frames: ReplayFrame[],
  all: ReplayFrame[],
  key: string,
  pick: (frame: ReplayFrame) => Region | null | undefined
): Region | null {
  const complete = frames.length === all.length;
  const cached = complete ? settledUnions.get(all)?.get(key) : undefined;
  if (cached !== undefined) return cached;

  const parts = frames.flatMap((frame) => pick(frame) ?? []);
  const united = parts.length === 0 ? null : canonicalize({
    boxes: parts.flatMap((region) => region.boxes),
    exact: parts.every((region) => region.exact),
    reasons: [...new Set(parts.flatMap((region) => region.reasons))],
  });
  if (complete) {
    const byKey = settledUnions.get(all) ?? new Map<string, Region | null>();
    byKey.set(key, united);
    settledUnions.set(all, byKey);
  }
  return united;
}

/**
 * Paint a family replay over the plan. It never substitutes for the plan's own
 * paint, and the plan is unchanged underneath it.
 *
 * On the family's tensor a frame is its task's tile. Everywhere else it is what
 * that task reads or computes, in the solid fill the Plan view gives the
 * inspected task's demand, with the part the inspected task demands too drawn
 * at full strength over the rest. Those two readings are the point: the replay
 * shows where each task's demand lands, and the emphasis shows the duplicate
 * demand the family rows count. Painting only the overlap would leave a task
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
  const all = playback.replay.frames;
  const frames = all.slice(0, playback.visited);
  const active = playback.phase === "playing" ? frames[frames.length - 1] : null;
  const layers: Layer[] = [];

  if (tensorId === playback.replay.task.tensorId) {
    for (const frame of active ? frames.slice(0, -1) : frames)
      layers.push({ region: fromBox(frame.box), color, alpha: 0.1 * playback.opacity, hatch: false });
    if (active)
      layers.push({ region: fromBox(active.box), color, alpha: 0.68 * playback.opacity, hatch: false, seed: true });
    return { layers };
  }

  const reached = active ? active.reach.get(tensorId) ?? null : null;
  const region = active
    ? reached?.region ?? null
    : unionAcrossFrames(frames, all, `${tensorId}|reach`, (frame) => frame.reach.get(tensorId)?.region);
  if (!region) return { layers };
  const shared = active
    ? reached?.shared ?? null
    : unionAcrossFrames(frames, all, `${tensorId}|shared`, (frame) => frame.reach.get(tensorId)?.shared);
  /* Cache the display difference across fade ticks. Its splitting budget
     keeps fragmented geometry bounded; on exhaustion the full reach remains
     visible and overlap may darken. */
  const rest = shared ? playbackDifference(region, shared) : region;
  if (!isEmpty(rest))
    layers.push({
      region: rest,
      color,
      alpha: (active ? 0.26 : 0.09) * playback.opacity,
      hatch: !rest.exact,
    });
  if (shared)
    layers.push({
      region: shared,
      color,
      alpha: (active ? 0.62 : 0.22) * playback.opacity,
      hatch: !shared.exact,
      seed: !!active,
    });
  return { layers };
}
