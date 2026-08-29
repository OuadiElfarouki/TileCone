import { useEffect } from "react";
import { useStore, viewAxes } from "./store";
import { tileOf } from "./grid";

const isTyping = (el: EventTarget | null) => {
  const t = el as HTMLElement | null;
  return !!t && ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName);
};

/**
 * Global selection keybindings. Arrow keys move the selection in the plane the
 * card is currently showing (its row/col axes), so what moves on screen is what
 * moves in index space.
 */
export function useKeyboard(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return;
      const s = useStore.getState();
      const sel = s.selection;
      const cfg = sel ? s.viewCfgs[sel.tensorId] : null;
      const shape = sel ? s.resolved?.tensors[sel.tensorId].resolved : null;

      // direction / view
      if (e.key === "u") return s.setDirection("backward");
      if (e.key === "d") return s.setDirection("forward");
      if (e.key === "b") return s.setDirection("both");
      if (e.key === "Escape") return s.clearSelection();
      if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        return s.undoSelection();
      }

      if (!sel || !cfg || !shape) return;

      const { rowAxis: rowAx, colAxis: colAx } = viewAxes(shape);
      const visible = [rowAx, colAx].filter((a) => a >= 0);
      // one tile is the visible unit, so that is what an arrow key moves
      const tile = tileOf(shape, s.tileScale);
      const step = e.shiftKey ? tile * 8 : tile;

      const arrows: Record<string, [number, number]> = {
        ArrowLeft: [colAx, -1],
        ArrowRight: [colAx, 1],
        ArrowUp: [rowAx, -1],
        ArrowDown: [rowAx, 1],
      };
      const hit = arrows[e.key];
      if (hit) {
        const [axis, sign] = hit;
        if (axis < 0) return;
        e.preventDefault();
        s.moveSelection(axis, sign * step);
        return;
      }

      // hidden-axis scrub
      if (e.key === "[" || e.key === "]") {
        const hidden = shape.map((_, ax) => ax).filter((ax) => !visible.includes(ax));
        if (!hidden.length) return;
        const ax = hidden[0];
        const sliders = cfg.sliders.slice();
        const delta = e.key === "]" ? 1 : -1;
        sliders[ax] = Math.max(0, Math.min(shape[ax] - 1, (sliders[ax] ?? 0) + delta));
        s.setViewCfg(sel.tensorId, { sliders });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
