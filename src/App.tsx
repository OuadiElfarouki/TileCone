import React, { useEffect } from "react";
import { SidePanel } from "./ui/SidePanel";
import { GraphView } from "./ui/GraphView";
import { Inspector } from "./ui/Inspector";
import { Toolbar } from "./ui/Toolbar";
import { useStore } from "./ui/store";
import { useKeyboard } from "./ui/useKeyboard";
import { fromBox } from "./core/region";

export default function App(): React.ReactElement {
  const loadExample = useStore((s) => s.loadExample);
  const applyDSL = useStore((s) => s.applyDSL);
  const setSelection = useStore((s) => s.setSelection);
  const setDirection = useStore((s) => s.setDirection);
  const resolved = useStore((s) => s.resolved);

  useKeyboard();

  useEffect(() => {
    const m = /^#s=(.+)$/.exec(location.hash);
    if (m) {
      try {
        const state = JSON.parse(decodeURIComponent(escape(atob(m[1]))));
        applyDSL(state.dsl);
        if (state.dir) setDirection(state.dir);
        if (typeof state.tile === "number") useStore.getState().setTileScale(state.tile);
        if (state.sel) {
          const boxes = state.sel.boxes as [number, number][][];
          setSelection(
            state.sel.t,
            boxes
              .map((b) => fromBox(b.map(([lo, hi]) => ({ lo, hi }))))
              .reduce((a, r) => ({ boxes: [...a.boxes, ...r.boxes], exact: true, reasons: [] })),
            "replace"
          );
        }
        return;
      } catch {
        /* fall through to default example */
      }
    }
    loadExample(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app">
      <Toolbar />
      <div className="main">
        <SidePanel />
        {resolved ? <GraphView /> : <div className="canvas-empty">loading…</div>}
        <Inspector />
      </div>
    </div>
  );
}
