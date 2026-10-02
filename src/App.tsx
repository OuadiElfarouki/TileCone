import React, { useCallback, useEffect, useState } from "react";
import { PanelFrame } from "./components/chrome/PanelFrame";
import { SidePanel } from "./components/chrome/SidePanel";
import { GraphView } from "./components/graph/GraphView";
import { Inspector } from "./components/inspector/Inspector";
import { WorkspaceHeader } from "./components/chrome/WorkspaceHeader";
import { useStore } from "./state/store";
import { useDragGuard } from "./components/hooks/useDragGuard";
import { useKeyboard } from "./components/hooks/useKeyboard";
import { useFocusPolicy } from "./components/hooks/useFocusPolicy";
import { decodeWorkspace } from "./state/share";
import { ShortcutsDialog } from "./components/chrome/ShortcutsDialog";

export default function App(): React.ReactElement {
  const loadExampleAsync = useStore((s) => s.loadExampleAsync);
  const restoreWorkspaceAsync = useStore((s) => s.restoreWorkspaceAsync);
  const resolved = useStore((s) => s.resolved);
  const theme = useStore((s) => s.theme);
  const moveOps = useStore((s) => s.moveOps);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const showShortcuts = useCallback(() => setShortcutsOpen(true), []);
  const closeShortcuts = useCallback(() => setShortcutsOpen(false), []);

  useKeyboard({ shortcutsOpen, showShortcuts, closeShortcuts });
  useFocusPolicy();
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
    try {
      localStorage.setItem("tilecone.moveOps", moveOps ? "on" : "off");
    } catch {
      // As with the theme, persistence is optional: the setting still applies
      // to this session.
    }
  }, [moveOps]);

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
          nodeOffsets: Object.fromEntries(
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
