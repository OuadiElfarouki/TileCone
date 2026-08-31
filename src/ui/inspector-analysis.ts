import { useMemo } from "react";
import { contributions } from "../core/contribution";
import { coneReadout, computeMetrics } from "../core/metrics";
import { coneFindings } from "../core/notes";
import type { PropResult } from "../core/propagate";
import { fromBox, Region, union } from "../core/region";
import type { ResolvedGraph } from "../core/graph";
import { enabledPropResult } from "./store";
import type { BoxProp, Direction, SelPart } from "./store";

/** Contribution classification is useful only when downstream rows are shown. */
export function contributionAnalysisEnabled(direction: Direction): boolean {
  return direction === "forward" || direction === "both";
}

type InspectorAnalysisArgs = {
  resolved: ResolvedGraph | null;
  selection: { parts: SelPart[] } | null;
  backward: PropResult | null;
  forward: PropResult | null;
  perBox: BoxProp[] | null;
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  countIntermediates: boolean;
  direction: Direction;
};

/**
 * Memoized analysis model consumed by the inspector view.
 *
 * Each derivation keeps its own dependency list: toggling byte accounting does
 * not rerun contribution probes, and changing only the direction runs those
 * probes precisely when downstream rows become visible.
 */
export function useInspectorAnalysis({
  resolved,
  selection,
  backward,
  forward,
  perBox,
  hiddenBoxes,
  focusedBox,
  countIntermediates,
  direction,
}: InspectorAnalysisArgs) {
  const scopedBack = useMemo(
    () => enabledPropResult(backward, perBox, hiddenBoxes, focusedBox, "backward"),
    [backward, perBox, hiddenBoxes, focusedBox]
  );
  const scopedFwd = useMemo(
    () => enabledPropResult(forward, perBox, hiddenBoxes, focusedBox, "forward"),
    [forward, perBox, hiddenBoxes, focusedBox]
  );

  const metrics = useMemo(() => {
    if (!resolved || !scopedBack) return null;
    return computeMetrics(resolved, scopedBack, countIntermediates);
  }, [resolved, scopedBack, countIntermediates]);

  const findings = useMemo(
    () => (resolved && scopedBack ? coneFindings(resolved, scopedBack) : null),
    [resolved, scopedBack]
  );

  /** The tile's own region per tensor, scoped the same way as its cones. */
  const seeds = useMemo(() => {
    const map = new Map<string, Region>();
    if (!selection) return map;
    const parts = focusedBox !== null && !hiddenBoxes.has(focusedBox) && selection.parts[focusedBox]
      ? [selection.parts[focusedBox]]
      : selection.parts.filter((_, index) => !hiddenBoxes.has(index));
    for (const part of parts) {
      const prev = map.get(part.tensorId);
      map.set(part.tensorId, prev ? union(prev, fromBox(part.box)) : fromBox(part.box));
    }
    return map;
  }, [selection, focusedBox, hiddenBoxes]);

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

  return { metrics, findings, seeds, contribution, upstream, downstream };
}
