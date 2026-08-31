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
  const restoreWorkspace = useStore((s) => s.restoreWorkspace);
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
      const restored = restoreWorkspace({
        dsl: link.dsl,
        direction: link.dir,
        tileScale: link.tile,
        snapToGrid: link.snap !== false,
        parts:
          link.sel?.map((p) => ({
            tensorId: p.t,
            box: p.box.map(([lo, hi]) => ({ lo, hi })),
          })) ?? null,
      });
      if (restored)
        return;
    }
    loadExample(0);
  }, [loadExample, restoreWorkspace]);

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
