import { useEffect } from "react";
import { useStore } from "../../state/store";
import { matchesShortcut, SHORTCUTS } from "../../view/shortcuts";
import { hiddenAxisPositioned, viewAxes } from "../../view/tensor/tensor-view";
import { activeCanvasTool, analysisTarget, graphIsInteractive, InspectorTab, keyboardTile } from "../../view/workspace";
import { activePlanTrail } from "../../state/plan";

const isTyping = (el: EventTarget | null) => {
  const t = el as HTMLElement | null;
  return !!t && ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName);
};

/**
 * True when focus sits in a widget that consumes arrow keys itself. A text
 * field is the obvious case; an open menu is the one that is easy to miss,
 * because it is built from a div and a list rather than from a tag this guard
 * would recognise. Without it the global bindings run *as well as* the widget:
 * arrowing through the example menu also walked the selected tile across its
 * tensor, which reads as the two being connected when they are not.
 */
const ownsWorkspaceKeys = (el: EventTarget | null) => {
  const t = el as HTMLElement | null;
  return (
    isTyping(t) ||
    !!t?.closest?.('[role="listbox"], [role="menu"]')
  );
};

/**
 * The tile a removal acts on, or `null` when the keys should do nothing.
 *
 * The keyboard's tile (`keyboardTile`), as for a move: the one hovered in the
 * list, or the one just drawn or clicked, which the canvas draws with the
 * heavier border. Escape and a press elsewhere release it, so a removal never
 * takes a tile the reader has let go of.
 */
export function removalTarget(keyTile: number | null, tab: InspectorTab): number | null {
  return tab === "dependencies" ? keyTile : null;
}

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
       * selection - clearing the tiles is the right panel's "clear all", an
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
        // An open menu cancels itself. A focused node does not: it takes the
        // arrows, not Escape, so the press still reaches the move tool below.
        if (el?.closest?.('[role="listbox"], [role="menu"]')) return;
        // The reuse lens is an overlay on one card; taking it down comes first.
        if (s.reuseLens !== null) {
          e.preventDefault();
          s.setReuseLens(null);
          return;
        }
        // Resizing ends first; the Move tool it selected stays in effect.
        if (s.resizingTensor !== null) {
          e.preventDefault();
          s.setResizingTensor(null);
          return;
        }
        // A rubber-band or a node drag cancels itself, and that is the whole of
        // the press - it must not also leave the move tool the drag was made in.
        if (s.dragging || e.defaultPrevented) return;
        if (!graphIsInteractive(s)) return;
        if (activeCanvasTool(s) === "move") {
          // The tool is the mode the canvas is in, so it goes before anything
          // the canvas is pointing at: one press puts it back to drawing. In
          // the Plan view no tool is in effect but select, so Escape goes on
          // to the task.
          e.preventDefault();
          s.setCanvasTool("select");
          return;
        }
        if (s.inspectorTab === "plan" && s.planTask) {
          // The innermost thing the Plan view is pointing at. The tilings stay:
          // clearing those is the panel's explicit act, as clearing tiles is.
          e.preventDefault();
          s.selectPlanTask(null);
          return;
        }
        if (s.pinnedBox !== null || s.focusedBox !== null || s.editBox !== null) {
          s.clearFocus();
          return;
        }
        return; // nothing active: do nothing
      }

      // A modal shortcut reference is not a translucent route to the workspace
      // behind it. Tab remains owned by the dialog's focus trap; every app-level
      // binding waits until the sheet closes.
      if (shortcutsOpen) return;

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

      if (ownsWorkspaceKeys(e.target)) return;
      if (s.dragging) return;
      // Movable nodes own arrows, while Undo, Fit and Help remain global.
      const el = e.target as HTMLElement | null;
      if (e.key.startsWith("Arrow") && el?.closest?.("[data-node-move]")) return;
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
      /* Alt+arrow is the browser's Back and Forward on some platforms. The
         workspace claims it whenever it has the keys, so a press with no tile
         to grow - or in the Plan view, or on an old build - cannot navigate
         away from the page. */
      if (matchesShortcut(e, SHORTCUTS.growTile)) {
        e.preventDefault();
        if (!graphIsInteractive(s)) return;
        // In the Plan view Alt+← steps back along the followed producers, as
        // the browser's Back would step back a page.
        if (s.inspectorTab === "plan") {
          const trail = activePlanTrail(s);
          if (e.key === "ArrowLeft" && trail) s.backPlanTask(trail.from.length - 1);
          return;
        }
        const index = keyboardTile(s);
        const tensorId = index === null ? null : s.selection!.parts[index].tensorId;
        const tileShape = tensorId ? s.resolved?.tensors[tensorId].resolved : null;
        if (!tensorId || !tileShape) return;
        const { rowAxis, colAxis } = viewAxes(tileShape, s.viewCfgs[tensorId]);
        const steps: Record<string, [number, boolean]> = {
          ArrowRight: [colAxis, true],
          ArrowLeft: [colAxis, false],
          ArrowDown: [rowAxis, true],
          ArrowUp: [rowAxis, false],
        };
        const [axis, grow] = steps[e.key];
        // Auto-repeat records no undo entry, as a held move does not.
        if (axis >= 0) s.growSelection(axis, grow, !e.repeat);
        return;
      }

      // Navigation and panel controls remain available for an old build;
      // selection, plan edits and workspace undo require the current source.
      if (!graphIsInteractive(s)) return;
      // The same group/focus policy as the inspector and movement action.
      const target = analysisTarget(s.selection?.parts ?? [], s.analysisGroup, s.perBox ? s.focusedBox : null);
      const anchor = target.tensorId;
      const cfg = anchor ? s.viewCfgs[anchor] : null;
      const shape = anchor ? s.resolved?.tensors[anchor].resolved : null;

      // direction / view
      if (matchesShortcut(e, SHORTCUTS.needs)) return s.toggleDirection("backward");
      if (matchesShortcut(e, SHORTCUTS.feeds)) return s.toggleDirection("forward");
      if (matchesShortcut(e, SHORTCUTS.entangled)) return s.toggleEntangled();
      if (matchesShortcut(e, SHORTCUTS.undo)) {
        e.preventDefault();
        return s.undoWorkspace();
      }

      /* In the Plan view the arrows step the inspected task by one tile. The
         same keys move the same kind of thing: the unit the panel below names.
         This runs before the selection anchor is required, because a plan can
         be inspected with nothing drawn. */
      if (s.inspectorTab === "plan" && matchesShortcut(e, SHORTCUTS.movePlanTask)) {
        const task = s.planTask;
        const taskShape = task ? s.resolved?.tensors[task.tensorId].resolved : null;
        if (task && taskShape) {
          const axes = viewAxes(taskShape, s.viewCfgs[task.tensorId]);
          const steps: Record<string, [number, number]> = {
            ArrowLeft: [axes.colAxis, -1],
            ArrowRight: [axes.colAxis, 1],
            ArrowUp: [axes.rowAxis, -1],
            ArrowDown: [axes.rowAxis, 1],
          };
          const step = steps[e.key];
          if (step && step[0] >= 0) {
            e.preventDefault();
            // Auto-repeat records no undo entry, as a held tile move does not.
            s.movePlanTask(step[0], step[1] * (matchesShortcut(e, SHORTCUTS.moveFast) ? 8 : 1), !e.repeat);
            return;
          }
        }
      }

      /* The arrows move the keyboard's tile, on its own tensor's drawn plane;
         with none they move nothing. The rest below still follows the anchor. */
      const keyTile = keyboardTile(s);
      const keyTensor = keyTile === null ? null : s.selection!.parts[keyTile].tensorId;
      const keyShape = keyTensor ? s.resolved?.tensors[keyTensor].resolved : null;
      const keyAxes = keyTensor && keyShape ? viewAxes(keyShape, s.viewCfgs[keyTensor]) : null;
      const arrows: Record<string, [number, number]> = keyAxes
        ? {
            ArrowLeft: [keyAxes.colAxis, -1],
            ArrowRight: [keyAxes.colAxis, 1],
            ArrowUp: [keyAxes.rowAxis, -1],
            ArrowDown: [keyAxes.rowAxis, 1],
          }
        : {};
      /* Moving a tile is an edit to the selection, and the selection is the
         Dependencies view's subject: its list is what shows which tile the
         arrows are about to move. Under Plan that list is not on screen and
         the arrows step the inspected task instead, rather than editing a tile
         the reader cannot see. */
      const editsTiles = s.inspectorTab === "dependencies";
      const hit = editsTiles && matchesShortcut(e, SHORTCUTS.move) ? arrows[e.key] : undefined;
      if (hit) {
        const [axis, sign] = hit;
        if (axis < 0) return;
        e.preventDefault();
        // Auto-repeat records no undo entry, so holding an arrow is one step to
        // undo rather than many individual steps, which would also evict the
        // meaningful history from the store's bounded undo stack.
        s.nudgeSelection(axis, sign as -1 | 1, matchesShortcut(e, SHORTCUTS.moveFast) ? 8 : 1, !e.repeat);
        return;
      }

      /* `deleteBox` records a workspace history entry, so a removal is one
         Ctrl/Cmd+Z away like any other tile edit. */
      const removable = removalTarget(keyTile, s.inspectorTab);
      if (matchesShortcut(e, SHORTCUTS.deleteTile) && removable !== null) {
        e.preventDefault();
        s.deleteBox(removable);
        return;
      }

      // include/exclude the focused part from merged analysis and paint. Also a
      // selection edit, so it follows the same rule as movement above: a pin
      // survives a view switch, and this would otherwise change which tiles are
      // measured while the list saying so is in the other view.
      if (editsTiles && matchesShortcut(e, SHORTCUTS.toggleTile) && keyTile !== null && s.perBox) {
        e.preventDefault();
        s.toggleBoxHidden(keyTile);
        return;
      }

      if (!anchor || !cfg || !shape) return;
      const { rowAxis: rowAx, colAxis: colAx } = viewAxes(shape, cfg);
      const visible = [rowAx, colAx].filter((a) => a >= 0);

      // hidden-axis scrub: the first hidden axis with a position (`hiddenAxisPositioned`)
      if (matchesShortcut(e, SHORTCUTS.scrub)) {
        const ax = shape.findIndex((_, axis) =>
          !visible.includes(axis) && hiddenAxisPositioned(shape, cfg, axis));
        if (ax < 0) return;
        const sliders = cfg.sliders.slice();
        // A tensor with a tile of its own steps a whole tile on this axis.
        const delta = (e.key === "]" ? 1 : -1) * (cfg.tile?.[ax] ?? 1);
        sliders[ax] = Math.max(0, Math.min(shape[ax] - 1, (sliders[ax] ?? 0) + delta));
        s.setViewCfg(anchor, { sliders });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closeShortcuts, shortcutsOpen, showShortcuts]);
}
