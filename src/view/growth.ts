/**
 * The tile-growth lens: what growing a tile adds to what it needs.
 *
 * Alt+arrow moves the keyboard tile's right or bottom edge by one grid step.
 * The tile the first step started from is kept as a baseline, and the
 * backward cone of the grown tile is compared with the baseline's: per
 * tensor, the elements the larger tile needs that the starting one did not.
 * For `C = A @ B`, one more column of C needs no more of A and a whole column
 * of B - the shape of the change, which the cone of either tile alone does
 * not show.
 *
 * Only the footprint is compared. No new elements in a tensor means the
 * larger tile reads nothing there it did not already read, not that it does
 * no more arithmetic or moves no more data.
 */

import type { ResolvedGraph } from "../core/graph";
import { executeQuery } from "../core/executor";
import { count, empty, fromBox, subtract, type Box, type Interval, type Region } from "../core/region";
import type { SelPart } from "./workspace";

/**
 * Where a grown tile started and where it is now, with what the change adds.
 * Stored when a step is taken; it describes the keyboard tile only while that
 * tile is still the part the last step installed (`activeGrowth`), so a move,
 * an edit or an undo retires it without anything having to clear it.
 */
export type Growth = {
  index: number;
  tensorId: string;
  baseBox: Box;
  box: Box;
  /** The selection part the last step installed, compared by identity. */
  part: SelPart;
  delta: GrowthDelta;
};

export type GrowthDelta = {
  /** Elements of the grown tile beyond the starting one. */
  outputs: number;
  /** Per tensor the cone reaches, in declaration order: what it needs beyond the baseline's. */
  tensors: { tensorId: string; added: Region; elements: number }[];
  /** False when either cone is a bound, so the differences are approximate. */
  exact: boolean;
};

/**
 * The far edge of `interval` moved one grid step: out to the next boundary of
 * `unit` when growing, back to the previous one when shrinking. Null when it
 * cannot move - at the end of the axis, or at one step wide.
 */
export function stepEdge(interval: Interval, unit: number, extent: number, grow: boolean): Interval | null {
  if (grow) {
    if (interval.hi >= extent) return null;
    return { lo: interval.lo, hi: Math.min(extent, Math.ceil((interval.hi + 1) / unit) * unit) };
  }
  const hi = Math.floor((interval.hi - 1) / unit) * unit;
  return hi > interval.lo ? { lo: interval.lo, hi } : null;
}

/** What the tile `box` on `tensorId` needs beyond what `baseBox` needs. */
export function growthDelta(graph: ResolvedGraph, tensorId: string, baseBox: Box, box: Box): GrowthDelta {
  const cone = (b: Box) => executeQuery(graph, { tensorId, region: fromBox(b), direction: "backward" }).backward!;
  const base = cone(baseBox);
  const now = cone(box);
  let exact = true;
  const tensors: GrowthDelta["tensors"] = [];
  for (const [id, result] of now.tensors) {
    if (id === tensorId) continue;
    const before = base.tensors.get(id)?.region ?? empty(graph.tensors[id].resolved.length);
    if (!result.region.exact || !before.exact) exact = false;
    const added = subtract(result.region, before);
    tensors.push({ tensorId: id, added, elements: count(added) });
  }
  // In the order the source declares the tensors, which is the order a reader
  // looks for them in.
  const order = new Map(Object.keys(graph.tensors).map((id, i) => [id, i]));
  tensors.sort((a, b) => order.get(a.tensorId)! - order.get(b.tensorId)!);
  return { outputs: count(subtract(fromBox(box), fromBox(baseBox))), tensors, exact };
}

/**
 * The stored growth, while it still describes the keyboard tile `index`.
 *
 * Identity, not an equal box: moving the tile away and back, or redrawing the
 * same box, installs another part object, so a growth left behind never
 * returns with a baseline from before the detour. An undo restores the part
 * object itself, and with it the growth that made it.
 */
export function activeGrowth(
  growth: Growth | null,
  parts: readonly SelPart[] | undefined,
  index: number | null
): Growth | null {
  if (!growth || index === null || growth.index !== index) return null;
  return parts?.[index] === growth.part ? growth : null;
}
