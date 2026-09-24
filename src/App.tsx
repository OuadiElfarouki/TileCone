import React, { useCallback, useEffect, useState } from "react";
import { PanelFrame } from "./ui/PanelFrame";
import { SidePanel } from "./ui/SidePanel";
import { GraphView } from "./ui/GraphView";
import { Inspector } from "./ui/Inspector";
import { WorkspaceHeader } from "./ui/WorkspaceHeader";
import { useStore } from "./ui/store";
import { useDragGuard } from "./ui/useDragGuard";
import { useKeyboard } from "./ui/useKeyboard";
import { decodeWorkspace } from "./ui/share";
import { ShortcutsDialog } from "./ui/ShortcutsDialog";

export default function App(): React.ReactElement {
  const loadExampleAsync = useStore((s) => s.loadExampleAsync);
  const restoreWorkspaceAsync = useStore((s) => s.restoreWorkspaceAsync);
  const resolved = useStore((s) => s.resolved);
  const theme = useStore((s) => s.theme);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const showShortcuts = useCallback(() => setShortcutsOpen(true), []);
  const closeShortcuts = useCallback(() => setShortcutsOpen(false), []);

  useKeyboard({ shortcutsOpen, showShortcuts, closeShortcuts });
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
    let live = true;
    void (async () => {
      const link = decodeWorkspace(location.hash);
      if (link) {
        const restored = await restoreWorkspaceAsync({
          dsl: link.dsl,
          direction: link.dir,
          showEntangled: link.ent === true,
          tileScale: link.tile,
          snapToGrid: link.snap !== false,
          axisMode: link.axes ?? "symbolic",
          viewCfgs: link.views,
          tensorOffsets: Object.fromEntries(
            Object.entries(link.pos ?? {}).map(([id, [dx, dy]]) => [id, { dx, dy }])
          ),
          parts:
            link.sel?.map((p) => ({
              tensorId: p.t,
              box: p.box.map(([lo, hi]) => ({ lo, hi })),
            })) ?? null,
        });
        if (!live || restored) return;
      }
      await loadExampleAsync(0);
    })();
    return () => { live = false; };
  }, [loadExampleAsync, restoreWorkspaceAsync]);

  return (
    <div className="app">
      <WorkspaceHeader onShowShortcuts={showShortcuts} />
      <div className="main">
        <PanelFrame side="left" label="source">
          <SidePanel />
        </PanelFrame>
        {resolved ? <GraphView /> : <div className="canvas-empty">loading…</div>}
        <PanelFrame side="right" label="tiles">
          <Inspector />
        </PanelFrame>
      </div>
      <ShortcutsDialog open={shortcutsOpen} onClose={closeShortcuts} />
    </div>
  );
}
