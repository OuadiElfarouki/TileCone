/** Reading a reuse sweep: which run is current, and how its figures are qualified. */

import { ResolvedGraph } from "../core/graph";
import { figure, type Figure } from "../core/metrics";
import { Box, sameBox } from "../core/region";
import { ReuseEstimate, ReuseSurface, ReuseSweep } from "../core/reuse";
import { FIGURE_MARK } from "./format";

/**
 * The local half of the reuse answer: what one tile's step along each axis still
 * shares with this tile. Both directions usually agree, and printing "50/50%"
 * for that would be noise, so equal shares collapse to one figure.
 */
export function neighbourShares(
  neighbors: ReuseEstimate["neighbors"],
  label: (axis: number) => string
): { text: string; reasons: string[] } {
  const byAxis = new Map<number, ReuseEstimate["neighbors"]>();
  const reasons = new Set<string>();
  for (const probe of neighbors) {
    const shares = byAxis.get(probe.axis) ?? [];
    shares.push(probe);
    byAxis.set(probe.axis, shares);
    if (!probe.exact) probe.reasons.forEach((reason) => reasons.add(reason));
  }
  const text = [...byAxis]
    .map(([axis, probes]) => {
      const values = probes.map((probe) =>
        `${FIGURE_MARK[probe.exact ? "exact" : "approximate"]}${Math.round(probe.sharedFraction * 100)}%`
      );
      const distinct = [...new Set(values)];
      if (distinct.length === 1) return `${label(axis)} ${distinct[0]}`;
      return `${label(axis)} ${probes.map((probe, index) =>
        `${probe.delta < 0 ? "−" : "+"}${values[index]}`
      ).join(" / ")}`;
    })
    .join(" · ");
  return { text, reasons: [...reasons] };
}

/**
 * The estimate's two headline numbers as figures, so they are marked as every
 * other figure is. Sampling and conservative geometry are independent sources
 * of uncertainty: an exhaustive count over widened regions can only overstate,
 * so it is an upper bound; a sampled count can miss either way. The shared
 * fraction is a ratio, which has no one-sided bound once either applies.
 */
export function reuseFigures(
  estimate: Pick<ReuseEstimate, "exhaustive" | "geometryExact" | "estimatedTiles" | "meanSharedFraction" | "reasons">
): { tiles: Figure; sharedFraction: Figure | null } {
  const { exhaustive, geometryExact, reasons } = estimate;
  return {
    tiles: figure(estimate.estimatedTiles, exhaustive ? (geometryExact ? "exact" : "upper") : "approximate", reasons),
    sharedFraction: estimate.meanSharedFraction === null
      ? null
      : figure(estimate.meanSharedFraction, exhaustive && geometryExact ? "exact" : "approximate", reasons),
  };
}

/** The tile a sweep is anchored on. `colorIndex` is its index in the whole
 *  selection, which is what owns its hue, so the playback paints in the tile's
 *  own colour rather than the first one. */
export type ReuseProbe = { tensorId: string; box: Box; colorIndex: number };

export type ReuseRun = {
  graph: ResolvedGraph;
  probe: ReuseProbe;
  rows: ReuseEstimate[];
  /** The relations this sweep was asked to trace. The rows do not depend on
   *  them - the estimate is backward demand either way - but the playback
   *  does, so a replay after a view toggle has to re-ask rather than repaint
   *  a surface the sweep never walked. */
  surfaces: ReuseSurface[];
  sweep?: ReuseSweep;
};

export const sameSurfaces = (left: readonly ReuseSurface[], right: readonly ReuseSurface[]) =>
  left.length === right.length && left.every((surface, i) => surface === right[i]);


/**
 * A reuse sweep is a cache keyed by the graph and the exact tile that seeded
 * it. Effect-based clearing is too late: React renders once with the new graph
 * before an effect runs, and an old input ID can crash that render.
 */
export function currentReuseRows(
  run: ReuseRun | null,
  graph: ResolvedGraph | null,
  probe: ReuseProbe | null
): ReuseEstimate[] | null {
  return run && graph === run.graph && probe &&
    probe.tensorId === run.probe.tensorId && sameBox(probe.box, run.probe.box)
    ? run.rows
    : null;
}
