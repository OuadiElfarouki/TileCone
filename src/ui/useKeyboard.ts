import { useEffect } from "react";
import { anchorTensorId, useStore, viewAxes } from "./store";
import { nudgeDelta, nudgeUnit } from "./grid";
import { matchesShortcut, SHORTCUTS } from "./shortcuts";

const isTyping = (el: EventTarget | null) => {
  const t = el as HTMLElement | null;
  return !!t && ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName);
};

/** GraphView owns viewport geometry; the one global keyboard listener asks it
 * to fit through this UI-local event rather than installing a second listener. */
export const FIT_GRAPH_EVENT = "tilecone:fit-graph";

/**
 * Global selection keybindings. Arrow keys move the selection in the plane the
 * card is currently showing (its row/col axes), so what moves on screen is what
 * moves in index space.
 */
export function useKeyboard({
  shortcutsOpen,
  showShortcuts,
  closeShortcuts,
}: {
  shortcutsOpen: boolean;
  showShortcuts: () => void;
  closeShortcuts: () => void;
}): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = useStore.getState();

      /**
       * Escape backs out of the innermost active mode. It never destroys the
       * selection — clearing the tiles is the right panel's "clear all", an
       * explicit act, not a side effect of pressing cancel.
       */
      if (matchesShortcut(e, SHORTCUTS.escape)) {
        if (shortcutsOpen) {
          e.preventDefault();
          closeShortcuts();
          return;
        }
        const el = e.target as HTMLElement | null;
        if (isTyping(el)) {
          el?.blur(); // leave text editing, keeping what was typed
          return;
        }
        if (s.dragging) return; // the card cancels its own rubber-band
        if (s.pinnedBox !== null || s.focusedBox !== null) {
          s.clearFocus();
          return;
        }
        return; // nothing active: do nothing
      }

      // Panel shortcuts remain global even while focus is inside the source or
      // an inspector control. They do not modify the field's contents.
      const panel = matchesShortcut(e, SHORTCUTS.leftPanel)
        ? "left"
        : matchesShortcut(e, SHORTCUTS.rightPanel)
          ? "right"
          : null;
      if (panel) {
        e.preventDefault();
        return s.togglePanel(panel);
      }

      if (isTyping(e.target)) return;
      if (matchesShortcut(e, SHORTCUTS.help)) {
        e.preventDefault();
        showShortcuts();
        return;
      }
      if (matchesShortcut(e, SHORTCUTS.fit)) {
        e.preventDefault();
        window.dispatchEvent(new Event(FIT_GRAPH_EVENT));
        return;
      }
      // Arrow keys and slider keys act on one tensor's axes, so they follow the
      // anchor: the focused tile's tensor, else the last one drawn on.
      const anchor = anchorTensorId(s.selection, s.focusedBox);
      const cfg = anchor ? s.viewCfgs[anchor] : null;
      const shape = anchor ? s.resolved?.tensors[anchor].resolved : null;

      // direction / view
      if (matchesShortcut(e, SHORTCUTS.needs)) return s.toggleDirection("backward");
      if (matchesShortcut(e, SHORTCUTS.feeds)) return s.toggleDirection("forward");
      if (matchesShortcut(e, SHORTCUTS.undo)) {
        e.preventDefault();
        return s.undoWorkspace();
      }

      if (!anchor || !cfg || !shape) return;

      const { rowAxis: rowAx, colAxis: colAx } = viewAxes(shape);
      const visible = [rowAx, colAx].filter((a) => a >= 0);
      const unit = nudgeUnit(shape, s.tileScale, s.graphPx, s.snapToGrid);

      const arrows: Record<string, [number, number]> = {
        ArrowLeft: [colAx, -1],
        ArrowRight: [colAx, 1],
        ArrowUp: [rowAx, -1],
        ArrowDown: [rowAx, 1],
      };
      const hit = matchesShortcut(e, SHORTCUTS.move) ? arrows[e.key] : undefined;
      if (hit) {
        const [axis, sign] = hit;
        if (axis < 0) return;
        e.preventDefault();
        const parts = s.selection?.parts ?? [];
        const anchorIndex = s.focusedBox !== null && parts[s.focusedBox]
          ? s.focusedBox
          : parts.length - 1;
        const interval = parts[anchorIndex]?.box[axis];
        if (!interval) return;
        const delta = nudgeDelta(
          interval,
          sign as -1 | 1,
          unit,
          s.snapToGrid,
          matchesShortcut(e, SHORTCUTS.moveFast) ? 8 : 1
        );
        // Auto-repeat records no undo entry, so holding an arrow is one step to
        // undo rather than forty — which would also evict the real history,
        // since it is capped at 40 entries.
        s.moveSelection(axis, delta, !e.repeat);
        return;
      }

      // include/exclude the focused part from merged analysis and paint
      if (matchesShortcut(e, SHORTCUTS.toggleTile) && s.focusedBox !== null && s.perBox) {
        e.preventDefault();
        s.toggleBoxHidden(s.focusedBox);
        return;
      }

      // hidden-axis scrub
      if (matchesShortcut(e, SHORTCUTS.scrub)) {
        const hidden = shape.map((_, ax) => ax).filter((ax) => !visible.includes(ax));
        if (!hidden.length) return;
        const ax = hidden[0];
        const sliders = cfg.sliders.slice();
        const delta = e.key === "]" ? 1 : -1;
        sliders[ax] = Math.max(0, Math.min(shape[ax] - 1, (sliders[ax] ?? 0) + delta));
        s.setViewCfg(anchor, { sliders });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closeShortcuts, shortcutsOpen, showShortcuts]);
}
