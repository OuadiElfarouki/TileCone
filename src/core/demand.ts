/**
 * Demand on one tensor from several readers: what they read summed over
 * readers, and the same demand with each element once.
 *
 * Two features ask this one question of different readers. Under Dependencies
 * the readers are the drawn tiles' cones, and the tensor a graph input; in a
 * plan they are a family's tasks, and the tensor any boundary they read. One
 * computation keeps the figures, their statuses and their wording the same.
 */

import { ResolvedGraph, Tensor } from "./graph";
import { byteFigure, figure, Figure, ratioFigure, sumFigures } from "./metrics";
import { PropResult } from "./propagate";
import { count, isEmpty, Region, unionOf } from "./region";

export type SharedDemand = {
  tensorId: string;
  /** Readers whose demand includes this tensor. */
  readers: number;
  /** Bytes each reader demands, summed over readers. A reader counts an element once. */
  summed: Figure;
  /** Bytes in the union of every reader's demand. */
  distinct: Figure;
  /**
   * `summed / distinct`: the mean number of readers of a demanded element.
   * Demand duplication, not a cache hit rate or a count of transfers.
   */
  duplication: Figure;
  /** `summed - distinct`: bytes demanded by more than one reader. */
  duplicate: Figure;
  /** Every reader's region is exact. */
  exact: boolean;
  /** Why a region is widened, when one is. */
  reasons: string[];
};

/** `perReader` holds one region per reader that demands the tensor. */
export function sharedDemand(tensor: Tensor, perReader: readonly Region[]): SharedDemand {
  const all = unionOf(perReader);
  const summed = sumFigures(perReader.map((region) => byteFigure(tensor, count(region), region)));
  const distinct = byteFigure(tensor, count(all), all);
  return {
    tensorId: tensor.id,
    readers: perReader.length,
    summed,
    distinct,
    duplication: ratioFigure(summed, distinct),
    duplicate: duplicateFigure(summed, distinct),
    exact: all.exact,
    reasons: [...all.reasons],
  };
}

/**
 * The difference of two widened sums is ordinarily no bound at all, but this
 * one is still an upper bound: at each element, widening can only raise the
 * number of readers whose region contains it, and the element's duplicate
 * demand, `max(membership - 1, 0)`, is monotone in that number.
 */
function duplicateFigure(summed: Figure, distinct: Figure): Figure {
  const reasons = [...summed.reasons, ...distinct.reasons];
  if (summed.value === null || distinct.value === null) return figure(0, "unknown", reasons);
  const exact = summed.status === "exact" && distinct.status === "exact";
  return figure(Math.max(0, summed.value - distinct.value), exact ? "exact" : "upper", reasons);
}

/** Graph-input demand across the drawn tiles (`SharedDemand`, with the tiles as readers). */
export type InputSharing = SharedDemand & {
  /** Tiles analysed together; `readers` is how many of them reach this input. */
  selectedTiles: number;
};

/** Element-granular leaf demand only; excludes memory transactions, caches,
 * internal reads and output traffic. This is shareable footprint, not measured traffic.
 * Count a represented union directly, without introducing a box-cap approximation. */
export function inputSharing(graph: ResolvedGraph, cones: PropResult[]): InputSharing[] {
  if (new Set(cones.flatMap((cone) => cone.roots)).size > 1)
    throw new Error("input sharing requires tiles on one tensor");
  return Object.values(graph.tensors).filter((t) => !t.producer).flatMap((input) => {
    const regions = cones.flatMap((cone) => {
      const region = cone.tensors.get(input.id)?.region;
      return region && !isEmpty(region) ? [region] : [];
    });
    if (!regions.length) return [];
    return [{ ...sharedDemand(input, regions), selectedTiles: cones.length }];
  });
}
