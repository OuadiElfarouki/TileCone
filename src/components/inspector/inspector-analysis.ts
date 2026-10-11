import { useMemo } from "react";
import { contributions } from "../../core/contribution";
import { coneReadout, computeMetrics } from "../../core/metrics";
import { operandReads } from "../../core/multiplicity";
import { coneFindings } from "../../core/notes";
import { inputSharing } from "../../core/demand";
import type { PropResult } from "../../core/propagate";
import { count, fromBox, Region, union } from "../../core/region";
import type { ResolvedGraph } from "../../core/graph";
import { analysisTarget, groupPropResult, BoxProp, Direction, InspectorTab, SelPart } from "../../view/workspace";

const EMPTY_PARTS: SelPart[] = [];

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
  tab: InspectorTab;
};

/**
 * Memoized analysis model consumed by the inspector view.
 *
 * Each derivation keeps its own dependency list and active-tab gate: Plan uses
 * its own model, and Dependencies measures cost, demand, reads and notes.
 * Direction gates contribution probes.
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
  tab,
}: InspectorAnalysisArgs) {
  const parts = selection?.parts ?? EMPTY_PARTS;
  const coneAnalysis = tab !== "plan";
  const dependencies = tab === "dependencies";
  const scopedBack = useMemo(
    () => coneAnalysis ? groupPropResult(byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId, "backward") : null,
    [coneAnalysis, byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId]
  );
  const scopedFwd = useMemo(
    () => coneAnalysis ? groupPropResult(byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId, "forward") : null,
    [coneAnalysis, byTensorRes, perBox, parts, hiddenBoxes, focusedBox, activeTensorId]
  );

  const metrics = useMemo(() => {
    if (!resolved || !scopedBack) return null;
    return computeMetrics(resolved, scopedBack);
  }, [resolved, scopedBack]);

  /**
   * Duplicate graph-input element demand among enabled tiles. It is unsampled
   * and uses already propagated cones. Conservative regions make duplicate
   * demand an upper bound, and this says nothing about actual cache traffic.
   */
  const sharing = useMemo(() => {
    if (!dependencies) return null;
    const cones = enabledBackwardProps(perBox, parts, hiddenBoxes, focusedBox, activeTensorId);
    if (!resolved || !cones || cones.length < 2) return null;
    return inputSharing(resolved, cones);
  }, [dependencies, resolved, perBox, parts, hiddenBoxes, focusedBox, activeTensorId]);

  const findings = useMemo(
    () => (dependencies && resolved && scopedBack ? coneFindings(resolved, scopedBack) : null),
    [dependencies, resolved, scopedBack]
  );

  /** How many outputs of each operation in the cone read each operand element. */
  const reads = useMemo(
    () => (dependencies && resolved && scopedBack ? operandReads(resolved, scopedBack.tensors) : null),
    [dependencies, resolved, scopedBack]
  );

  /** The tile's own region per tensor, scoped the same way as its cones. */
  const seeds = useMemo(() => {
    const map = new Map<string, Region>();
    if (!coneAnalysis || !selection) return map;
    const onTensor = selection.parts.filter((part, index) => part.tensorId === activeTensorId && !hiddenBoxes.has(index));
    const scoped =
      focusedBox !== null &&
      !hiddenBoxes.has(focusedBox) &&
      selection.parts[focusedBox]?.tensorId === activeTensorId
        ? [selection.parts[focusedBox]]
        : onTensor;
    for (const part of scoped) {
      const prev = map.get(part.tensorId);
      map.set(part.tensorId, prev ? union(prev, fromBox(part.box)) : fromBox(part.box));
    }
    return map;
  }, [coneAnalysis, selection, focusedBox, hiddenBoxes, activeTensorId]);

  const contributionEnabled = dependencies && contributionAnalysisEnabled(direction);
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

  return { metrics, sharing, reads, findings, seeds, contribution, upstream, downstream };
}
