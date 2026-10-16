import { useMemo } from "react";
import { useStore } from "../../state/store";
import { lensPaint, lensRegions, type LensPaint } from "../../view/reuse-lens";

/**
 * The reuse lens's paint for one card, or null when the lens is not on it.
 *
 * Derived here from the same store state the inspector reads, by the same
 * scope rules (`lensRegions`), rather than handed over from the panel: the
 * card then follows a change of tile, focus or task by itself. Every scope
 * selector returns null unless the lens is on this card, so the other cards
 * subscribe to nothing they would repaint for.
 */
export function useLensPaint(tensorId: string): LensPaint | null {
  const lens = useStore((s) =>
    s.reuseLens && s.reuseLens.tensorId === tensorId && s.reuseLens.view === s.inspectorTab ? s.reuseLens : null
  );
  const on = lens !== null;
  const graph = useStore((s) => (on ? s.resolved : null));
  const selection = useStore((s) => (on ? s.selection : null));
  const perBox = useStore((s) => (on ? s.perBox : null));
  const byTensorRes = useStore((s) => (on ? s.byTensorRes : null));
  const hiddenBoxes = useStore((s) => (on ? s.hiddenBoxes : null));
  const focusedBox = useStore((s) => (on ? s.focusedBox : null));
  const analysisGroup = useStore((s) => (on ? s.analysisGroup : null));
  const planSupply = useStore((s) => (on ? s.planSupply : null));
  return useMemo(() => {
    if (!lens || !graph || !hiddenBoxes) return null;
    const regions = lensRegions(lens.view, {
      selection, perBox, byTensorRes, hiddenBoxes, focusedBox, analysisGroup, planSupply,
    });
    return regions ? lensPaint(graph, regions, lens) : null;
  }, [lens, graph, selection, perBox, byTensorRes, hiddenBoxes, focusedBox, analysisGroup, planSupply]);
}
