import { type Box, canonicalize, isEmpty, type Region, subtractBox } from "../../core/region";
import { boxColor } from "../palette";
import { Layer } from "./grid";
import { ReplayPlayback } from "../workspace";
import type { FamilyReplay, ReplayFrame } from "../../core/plan/replay";

/** Boxes per trail layer, under the box budget a layer's fill is split within. */
const TRAIL_LAYER_BOXES = 192;
const trails = new WeakMap<FamilyReplay, Map<string, { visited: number; regions: Region[] }>>();

/** The tiles a frame walks on one tensor: the family's own tiles on the
 * family's tensor, and deeper, the producer tasks its steps move through. */
const walkedOn = (replay: FamilyReplay, frame: ReplayFrame | undefined, tensorId: string): Region | undefined =>
  !frame ? undefined : tensorId === replay.task.tensorId ? frame.region : frame.walked.get(tensorId);

/**
 * A tensor's trail: every tile the frames before the active one walked on it,
 * each once, less the active frame's, as disjoint layers.
 *
 * Several steps of a deeper replay share one family task, so painting each
 * earlier frame as its own layer stacked the same tile once per step, and the
 * trail's brightness came to count steps rather than mark where the walk has
 * been. The tiles are joined by identity first, which is exact: a frame's
 * region is whole tiles, and a tile walked twice is the same box. Cached per
 * replay and position, since fade ticks repaint without moving.
 */
function trailOf(replay: FamilyReplay, visited: number, tensorId: string): Region[] {
  const byTensor = trails.get(replay) ?? new Map<string, { visited: number; regions: Region[] }>();
  trails.set(replay, byTensor);
  const cached = byTensor.get(tensorId);
  if (cached && cached.visited === visited) return cached.regions;
  const active = walkedOn(replay, replay.frames[visited - 1], tensorId);
  const seen = new Map<string, Box>();
  for (const frame of replay.frames.slice(0, Math.max(0, visited - 1)))
    for (const b of walkedOn(replay, frame, tensorId)?.boxes ?? []) seen.set(b.map((i) => `${i.lo}:${i.hi}`).join(","), b);
  const boxes = [...seen.values()].flatMap((b) =>
    (active?.boxes ?? []).reduce<Box[]>((parts, a) => parts.flatMap((p) => subtractBox(p, a)), [b]));
  const regions: Region[] = [];
  for (let i = 0; i < boxes.length; i += TRAIL_LAYER_BOXES)
    regions.push({ boxes: boxes.slice(i, i + TRAIL_LAYER_BOXES), exact: true, reasons: [] });
  byTensor.set(tensorId, { visited, regions });
  return regions;
}

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
 * How strongly a level of the plan is drawn: the run's own reads at full
 * strength, each level further upstream fainter. A replay is a transient
 * overlay, so it may spend alpha on depth as the hover preview does; no
 * committed paint does.
 */
export const levelStrength = (level: number): number => Math.max(0.35, 1 - 0.3 * (level - 1));

/** A producer tile the active run needs, outlined in the plan's neutral ink. */
export type ReplayTile = { box: Box; definite: boolean };

/**
 * Paint a family replay over the plan, which recedes underneath it
 * (`buildPlanPaint`'s `recede`) and is otherwise unchanged.
 *
 * What a step moves is drawn as a walk: on the family's tensor, and deeper
 * on each tensor whose producer tasks the steps pass through, the tiles
 * walked so far stay as a faint trail and the active step's are drawn
 * strongly. At depth 2 of `D2 = f(D)` that is a walk along D's tiles, not
 * only along D2's. Everywhere else only the active run is drawn, the way the
 * Plan view draws the inspected task: what it reads or computes in the solid
 * fill, with the part the inspected task reads too at full strength over the
 * rest, and on a tiled tensor the producer tiles it needs, outlined. Painting
 * only the overlap would leave a run that shares nothing painting nothing.
 * Tensors reached through the plan's own producers at a deeper level are
 * drawn fainter, one step per level, and the part the inspected task does not
 * share is outlined in the hue, so a thin read - one row of a short card -
 * stays visible however faint its fill.
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
}): { layers: Layer[]; tiles: ReplayTile[] } {
  const color = boxColor(0, dark);
  const frames = playback.replay.frames.slice(0, playback.visited);
  const active = frames[frames.length - 1];
  const layers: Layer[] = [];
  if (!active) return { layers, tiles: [] };

  const walked = walkedOn(playback.replay, active, tensorId);
  // Where the walk has been, whichever path the active step is on: with A and
  // B both tiled under `Z = add(A, B)`, A's trail stays through a B step.
  for (const region of trailOf(playback.replay, playback.visited, tensorId))
    layers.push({ region, color, alpha: 0.1 * playback.opacity, hatch: false });
  if (tensorId === playback.replay.task.tensorId) {
    layers.push({ region: active.region, color, alpha: 0.68 * playback.opacity, hatch: false, seed: true });
    return { layers, tiles: [] };
  }

  const reached = active.reach.get(tensorId);
  if (!reached) {
    if (walked) layers.push({ region: walked, color, alpha: 0.68 * playback.opacity, hatch: false, seed: true });
    return { layers, tiles: [] };
  }
  const { region, shared, level, producers } = reached;
  const strength = levelStrength(level) * playback.opacity;
  /* Cache the display difference across fade ticks. Its splitting budget
     keeps fragmented geometry bounded; on exhaustion the full reach remains
     visible and overlap may darken. */
  const rest = shared ? playbackDifference(region, shared) : region;
  if (!isEmpty(rest))
    layers.push({
      region: rest, color, alpha: 0.26 * strength, hatch: !rest.exact,
      outline: true, lineWidth: 1, outlineAlpha: 0.95 * playback.opacity,
    });
  if (shared)
    layers.push({ region: shared, color, alpha: 0.62 * strength, hatch: !shared.exact, seed: true });
  // The step itself, over what it reads of its own tensor.
  if (walked) layers.push({ region: walked, color, alpha: 0.68 * playback.opacity, hatch: false, seed: true });
  const tiles = playback.exiting || !producers
    ? []
    : producers.boxes.map((box) => ({ box, definite: region.exact }));
  return { layers, tiles };
}
