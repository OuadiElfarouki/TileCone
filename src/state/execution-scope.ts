import { InspectorTab, SelPart, sweepAnchorIndex } from "../view/workspace";
import { State } from "./types";

/**
 * Narrow the attribution to the swept tile on the way into Cost model, and put
 * the reader's own back on the way out.
 *
 * Cost model answers one question about one tile - how many tiles of this size
 * share its demand - so the other tiles are not peers of it here the way they
 * are under Dependencies, and leaving them enabled would put their cones on the
 * cards beside a sweep that is not about them. Disabling is the existing
 * control rather than a second kind of invisibility: each tile keeps its faint
 * rectangle, so the reader can see they are still there.
 *
 * What is restored is what was set aside, not what Cost model ended up with, so
 * the Dependencies view comes back as it was left however the sweep was driven.
 */
export function executionScoping(
  state: State,
  tab: InspectorTab
): Partial<Pick<State, "hiddenBoxes" | "focusedBox" | "pinnedBox" | "executionScope">> {
  const parts = state.selection?.parts ?? [];
  if (tab === "execution") {
    const anchor = sweepAnchorIndex(
      parts,
      state.analysisGroup,
      // Above the attribution cap there is no per-tile propagation, so a focus
      // narrows nothing and must not choose the anchor either.
      state.perBox ? state.focusedBox : null,
      state.hiddenBoxes
    );
    if (anchor === null) return {};
    return {
      executionScope: state.executionScope ?? {
        hidden: [...state.hiddenBoxes].flatMap((index) => parts[index] ?? []),
        focused: state.focusedBox === null ? null : parts[state.focusedBox] ?? null,
        pinned: state.pinnedBox === null ? null : parts[state.pinnedBox] ?? null,
      },
      hiddenBoxes: new Set(parts.map((_part, index) => index).filter((index) => index !== anchor)),
      // Pinned as well as focused: with the tiles list gone there is nothing to
      // hover, but a pointer over the canvas can still move an unpinned focus.
      focusedBox: anchor,
      pinnedBox: anchor,
    };
  }
  const scope = state.executionScope;
  if (!scope) return {};
  const indexOf = new Map(parts.map((part, index) => [part, index]));
  const at = (part: SelPart | null) => (part === null ? null : indexOf.get(part) ?? null);
  return {
    executionScope: null,
    hiddenBoxes: new Set(scope.hidden.flatMap((part) => {
      const index = indexOf.get(part);
      return index === undefined ? [] : [index];
    })),
    focusedBox: at(scope.focused),
    pinnedBox: at(scope.pinned),
  };
}
