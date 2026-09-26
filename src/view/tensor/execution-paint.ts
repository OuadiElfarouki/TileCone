import { type Box, canonicalize, fromBox, isEmpty, type Region, subtractBox } from "../../core/region";
import { reuseReachAt, ReuseSurface, ReuseSweepFrame } from "../../core/reuse";
import { boxColor } from "../palette";
import { Layer } from "./grid";
import { downstreamPattern } from "./layers";
import { ExecutionPlayback } from "../workspace";

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
 * A union over the visited probes, computed once per sweep rather than once
 * per repaint.
 *
 * `union` canonicalizes to a fixpoint on every call, so folding forty-eight
 * probes pairwise is forty-seven of them for one answer - inside the paint
 * effect, redone on every dependency that repaints the card, for every tensor
 * and relation the sweep touches. One canonicalize over the concatenated boxes
 * represents the same set, with the same exactness and reasons: `union` is
 * that same call over two box lists, and is union-preserving either way. The
 * decomposition may differ, which nothing here reads - the fill is
 * disjointified before painting and a settled summary carries no perimeter.
 *
 * Keyed on the frames array, which a playback only replaces with a new sweep,
 * so ticking `visited`, settling, and fading all reuse the entry. A `WeakMap`
 * keeps it bounded without anyone having to retire it.
 */
const settledUnions = new WeakMap<ReuseSweepFrame[], Map<string, Region | null>>();

function unionAcrossFrames(
  frames: ReuseSweepFrame[],
  all: ReuseSweepFrame[],
  key: string,
  pick: (frame: ReuseSweepFrame) => Region | null | undefined
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

/** Relations in paint order, so a solid never lands on top of a texture that
 *  was meant to read through it. Same order `buildLayers` uses. */
const REUSE_SURFACES: ReuseSurface[] = ["backward", "forward", "entangled"];

/**
 * The mark each relation owns, borrowed from `buildLayers` rather than
 * reinvented: a probe's upstream reach must not be readable as its downstream
 * reach just because it arrived through the playback (§9).
 */
const surfacePattern = (
  surface: ReuseSurface,
  colorIndex: number
): Layer["pattern"] | undefined => {
  if (surface === "forward") return downstreamPattern(colorIndex);
  if (surface === "entangled") return { kind: "stipple", density: 0.5 };
  return undefined;
};

/**
 * Paint the probes the reuse estimator actually performed. This is an overlay:
 * it never substitutes sampled boxes for the user's selection.
 *
 * On the studied tensor a probe is its own rectangle. Everywhere else it is
 * whatever that probe reaches through each relation the sweep was asked for,
 * in that relation's own mark, with the part the anchor also reaches drawn at
 * full strength over the rest. Those two readings are the point: the sweep
 * shows where a tile of this size lands, and the emphasis shows the overlap
 * the estimate is actually counting. Painting only the overlap left a probe
 * that shares nothing painting nothing, which is most of a large sweep.
 *
 * It paints all of that and leaves the lattice alone. The sweep's own cover -
 * tiles of the anchor's extents laid over the tensor - would read naturally as
 * a lattice, but drawing one here breaks the rule that every drawn line is a
 * snapping boundary (§9): the card still takes ordinary selection drags under
 * Execution, and those snap to the square display tile, which the anchor's
 * extents are generally not. The Plan view can draw its own lattice because a
 * gesture on a tiled card inspects rather than draws; here it still draws. The
 * probe rectangles carry the cover anyway, and carry it more honestly - a
 * sampled sweep walks at most `sampleCap` of the tiles a full lattice would
 * have drawn, and only the walked ones stand behind a figure.
 */
export function buildExecutionPaint({
  tensorId,
  dark,
  playback,
}: {
  tensorId: string;
  dark: boolean;
  playback: ExecutionPlayback;
}): { layers: Layer[] } {
  const color = boxColor(playback.colorIndex, dark);
  const frames = playback.frames.slice(0, playback.visited);
  const active = playback.phase === "playing" ? frames[frames.length - 1] : null;
  const layers: Layer[] = [];

  if (tensorId === playback.tensorId) {
    for (const frame of active ? frames.slice(0, -1) : frames)
      layers.push({
        region: fromBox(frame.box),
        color,
        alpha: 0.1 * playback.opacity,
        hatch: false,
      });
    if (active)
      layers.push({
        region: fromBox(active.box),
        color,
        alpha: 0.68 * playback.opacity,
        hatch: false,
        seed: true,
      });
    return { layers };
  }

  for (const surface of REUSE_SURFACES) {
    const reached = active
      ? reuseReachAt(active.surfaces[surface], tensorId) ?? null
      : null;
    const region = active
      ? reached?.region ?? null
      : unionAcrossFrames(frames, playback.frames, `${surface}|${tensorId}|reach`,
          (frame) => reuseReachAt(frame.surfaces[surface], tensorId)?.region);
    if (!region) continue;
    const shared = active
      ? reached?.shared ?? null
      : unionAcrossFrames(frames, playback.frames, `${surface}|${tensorId}|shared`,
          (frame) => reuseReachAt(frame.surfaces[surface], tensorId)?.shared);
    const pattern = surfacePattern(surface, playback.colorIndex);
    /* Cache the display difference across fade ticks. Its splitting budget
       keeps fragmented geometry bounded; on exhaustion the full reach remains
       visible and overlap may darken. This never changes estimator figures. */
    const rest = shared ? playbackDifference(region, shared) : region;
    if (!isEmpty(rest))
      layers.push({
        region: rest,
        color,
        alpha: (active ? 0.26 : 0.09) * playback.opacity,
        hatch: !rest.exact,
        pattern,
      });
    if (shared)
      layers.push({
        region: shared,
        color,
        alpha: (active ? 0.62 : 0.22) * playback.opacity,
        hatch: !shared.exact,
        seed: !!active,
        pattern,
      });
  }

  return { layers };
}
