import { useMemo } from "react";
import { contributions } from "../core/contribution";
import { AggregateReadout, coneReadout, computeMetrics } from "../core/metrics";
import { coneFindings } from "../core/notes";
import type { PropResult } from "../core/propagate";
import { count, fromBox, Region, union } from "../core/region";
import type { ResolvedGraph } from "../core/graph";
import { analysisTarget, groupPropResult } from "./store";
import type { BoxProp, Direction, SelPart } from "./store";

export type ConeCost = { flops: number; bytes: number };
export type ConeBounds = { fused: ConeCost; unfused: ConeCost | null };

/**
 * Which tensor's tiles the readout below the list describes.
 *
 * Deliberately not a function of tile focus. Focus *narrows* within the group -
 * `groupFocus` below is what applies it - and never re-scopes: a hover is a
 * preview, and a preview that changes what the panel is about makes the pointer
 * an editor. It also made the group header unusable, since the header sits
 * inside the hovered list and the pointer must cross other groups' rows to
 * reach it. A click that means "go there" pins the tile, and pinning names the
 * group in the store.
 *
 * Falls back to the anchor when the group holds no tiles, which is the state a
 * workspace with a single group stays in.
 */
export function analysisTensorId(parts: SelPart[], group: string | null): string | null {
  return analysisTarget(parts, group, null).tensorId;
}

/** The measured tile set; above the attribution cap all group tiles contribute. */
export function measuredParts(parts: SelPart[], tensorId: string | null, hidden: Set<number>, focus: number | null, attributed: boolean) {
  const localFocus = attributed ? groupFocus(parts, focus, tensorId) : null;
  return parts.filter((part, index) => part.tensorId === tensorId &&
    (!attributed || (!hidden.has(index) && (localFocus === null || index === localFocus))));
}

export function measuredElements(parts: SelPart[]): number {
  return count({ boxes: parts.map((part) => part.box), exact: true, reasons: [] });
}

/**
 * The focused tile, but only where it belongs to the group being analysed.
 *
 * One expression of "focus narrows, it never re-scopes", read by everything
 * below the tiles list. The list itself keeps the raw focus, because the row
 * under the pointer should light up whichever group it is in.
 */
export function groupFocus(
  parts: SelPart[],
  focus: number | null,
  tensorId: string | null
): number | null {
  return focus !== null && parts[focus]?.tensorId === tensorId ? focus : null;
}

/** Preserve global indices, which own tile colors, while excluding other groups. */
export function groupAttribution(perBox: BoxProp[] | null, parts: SelPart[], tensorId: string | null): BoxProp[] | null {
  return perBox?.map((prop, index) => parts[index]?.tensorId === tensorId
    ? prop : { backward: null, forward: null }) ?? null;
}

/**
 * The per-tile backward cones the analysis is currently scoped to, in the same
 * order and under the same rules as `enabledPropResult`. `null` past
 * `MAX_PER_BOX_PROPS`, where per-tile attribution has been dropped.
 */
export function enabledBackwardProps(
  perBox: BoxProp[] | null,
  parts: SelPart[],
  hiddenBoxes: Set<number>,
  focusedBox: number | null,
  tensorId: string | null
): PropResult[] | null {
  if (!perBox || !tensorId) return null;
  if (
    focusedBox !== null &&
    !hiddenBoxes.has(focusedBox) &&
    parts[focusedBox]?.tensorId === tensorId
  ) {
    const one = perBox[focusedBox]?.backward;
    return one ? [one] : [];
  }
  return perBox.flatMap((prop, index) =>
    hiddenBoxes.has(index) || parts[index]?.tensorId !== tensorId || !prop.backward
      ? []
      : [prop.backward]
  );
}

/**
 * Two idealized execution scenarios, not guaranteed hardware bounds.
 *
 * Fusion here spans two axes, and the scenarios vary both. `fused`
 * is the merged cone: one kernel over every op and every tile, so no
 * intermediate reaches memory and a shared operand band is fetched once.
 * `unfused` charges every op of every tile as its own job - each intermediate
 * written and read back, each tile fetching its own operands - and adds those
 * up. Actual traffic can lie outside these scenarios (caches, spills, repeated
 * loads, recomputation and materialization choices are not modeled).
 *
 * On a plain GEMM the whole range is the tile axis: two adjacent output tiles
 * read one shared operand band, and `fused` counts it once where `unfused`
 * pays for it twice. On a single tile of a long chain the range is the op axis
 * instead. They collapse to one number when there is neither to be had, which
 * is the honest report for that graph.
 *
 * `unfused` needs per-tile cones, so it is `null` past `MAX_PER_BOX_PROPS`
 * rather than approximated: a merged cone cannot be taken apart again, and
 * splitting it by guess would invent a bound.
 */
export function coneBounds(
  resolved: ResolvedGraph,
  merged: AggregateReadout,
  enabled: PropResult[] | null
): ConeBounds {
  const fused = { flops: merged.flops, bytes: merged.inputBytes + merged.outputBytes };
  if (!enabled) return { fused, unfused: null };
  // Ratio of the sums, never the mean of the ratios: tiles differ in size, and
  // averaging their intensities would weight a one-element tile like a full row.
  const unfused = enabled.reduce<ConeCost>(
    (total, prop) => {
      const m = computeMetrics(resolved, prop);
      return {
        flops: total.flops + m.flops,
        bytes: total.bytes + m.unfusedBytes,
      };
    },
    { flops: 0, bytes: 0 }
  );
  return { fused, unfused };
}

/** Contribution classification is useful only when downstream rows are shown. */
export function contributionAnalysisEnabled(direction: Direction): boolean {
  return direction === "forward" || direction === "both";
}

type InspectorAnalysisArgs = {
  resolved: ResolvedGraph | null;
  selection: { parts: SelPart[] } | null;
  perBox: BoxProp[] | null;
  byTensorRes: Record<string, { backward: PropResult | null; forward: PropResult | null }> | null;
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  /** The tensor whose tiles this readout describes. Analysis never mixes two. */
  activeTensorId: string | null;
  direction: Direction;
};

/**
 * Memoized analysis model consumed by the inspector view.
 *
 * Each derivation keeps its own dependency list: changing only the direction
 * runs the contribution probes precisely when downstream rows become visible,
 * and the per-tile pass behind `bounds` reruns only when the enabled set moves.
 */
export function useInspectorAnalysis({
  resolved,
  selection,
  perBox,
  byTensorRes,
  hiddenBoxes,
  focusedBox,
  activeTensorId,
  direction,
}: InspectorAnalysisArgs) {
  const parts = selection?.parts ?? [];
  const scopedBack = useMemo(
    () => groupPropResult(byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId, "backward"),
    [byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId]
  );
  const scopedFwd = useMemo(
    () => groupPropResult(byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId, "forward"),
    [byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId]
  );

  const metrics = useMemo(() => {
    if (!resolved || !scopedBack) return null;
    return computeMetrics(resolved, scopedBack);
  }, [resolved, scopedBack]);

  /** @see coneBounds - idealized execution scenarios. */
  const bounds = useMemo(() => {
    if (!resolved || !metrics) return null;
    return coneBounds(
      resolved,
      metrics,
      enabledBackwardProps(perBox, parts, hiddenBoxes, focusedBox, activeTensorId)
    );
  }, [resolved, metrics, perBox, parts, hiddenBoxes, focusedBox, activeTensorId]);

  const findings = useMemo(
    () => (resolved && scopedBack ? coneFindings(resolved, scopedBack) : null),
    [resolved, scopedBack]
  );

  /** The tile's own region per tensor, scoped the same way as its cones. */
  const seeds = useMemo(() => {
    const map = new Map<string, Region>();
    if (!selection) return map;
    const onTensor = selection.parts.filter((part) => part.tensorId === activeTensorId);
    const scoped =
      focusedBox !== null &&
      !hiddenBoxes.has(focusedBox) &&
      selection.parts[focusedBox]?.tensorId === activeTensorId
        ? [selection.parts[focusedBox]]
        : onTensor.filter((part) => !hiddenBoxes.has(selection.parts.indexOf(part)));
    for (const part of scoped) {
      const prev = map.get(part.tensorId);
      map.set(part.tensorId, prev ? union(prev, fromBox(part.box)) : fromBox(part.box));
    }
    return map;
  }, [selection, focusedBox, hiddenBoxes, activeTensorId]);

  const contributionEnabled = contributionAnalysisEnabled(direction);
  const contribution = useMemo(
    () =>
      contributionEnabled && resolved && scopedFwd
        ? contributions(resolved, scopedFwd, seeds)
        : null,
    [contributionEnabled, resolved, scopedFwd, seeds]
  );

  const upstream = useMemo(() => {
    if (!metrics || !scopedBack) return [];
    const roots = new Set(scopedBack.roots);
    return metrics.tensors.filter((row) => !roots.has(row.tensorId));
  }, [metrics, scopedBack]);

  const downstream = useMemo(() => {
    if (!resolved || !scopedFwd) return [];
    const roots = new Set(scopedFwd.roots);
    return coneReadout(resolved, scopedFwd).filter((row) => !roots.has(row.tensorId));
  }, [resolved, scopedFwd]);

  return { metrics, bounds, findings, seeds, contribution, upstream, downstream };
}
