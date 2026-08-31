import React, { useEffect } from "react";
import { PanelFrame } from "./ui/PanelFrame";
import { SidePanel } from "./ui/SidePanel";
import { GraphView } from "./ui/GraphView";
import { Inspector } from "./ui/Inspector";
import { WorkspaceHeader } from "./ui/WorkspaceHeader";
import { useStore } from "./ui/store";
import { useDragGuard } from "./ui/useDragGuard";
import { useKeyboard } from "./ui/useKeyboard";
import { decodeWorkspace } from "./ui/share";

export default function App(): React.ReactElement {
  const loadExample = useStore((s) => s.loadExample);
  const applyDSL = useStore((s) => s.applyDSL);
  const restoreSelection = useStore((s) => s.restoreSelection);
  const setDirection = useStore((s) => s.setDirection);
  const resolved = useStore((s) => s.resolved);
  const theme = useStore((s) => s.theme);

  useKeyboard();
  useDragGuard();

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("tilecone.theme", theme);
    } catch {
      // Theme persistence is optional (for example in privacy-restricted tabs).
    }
  }, [theme]);

  useEffect(() => {
    const link = decodeWorkspace(location.hash);
    if (link) {
      try {
        applyDSL(link.dsl);
        setDirection(link.dir);
        useStore.getState().setTileScale(link.tile);
        useStore.getState().setSnapToGrid(link.snap !== false);
        if (link.sel)
          restoreSelection(
            link.sel.map((p) => ({
              tensorId: p.t,
              box: p.box.map(([lo, hi]) => ({ lo, hi })),
            }))
          );
        return;
      } catch {
        /* a link naming a tensor this graph lacks: fall back to the example */
      }
    }
    loadExample(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app">
      <WorkspaceHeader />
      <div className="main">
        <PanelFrame side="left" label="source">
          <SidePanel />
        </PanelFrame>
        {resolved ? <GraphView /> : <div className="canvas-empty">loading…</div>}
        <PanelFrame side="right" label="tiles">
          <Inspector />
        </PanelFrame>
      </div>
    </div>
  );
}
