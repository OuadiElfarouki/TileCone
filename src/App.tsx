import React, { useCallback, useEffect, useState } from "react";
import { PanelFrame } from "./components/chrome/PanelFrame";
import { SidePanel } from "./components/chrome/SidePanel";
import { GraphView } from "./components/graph/GraphView";
import { Inspector } from "./components/inspector/Inspector";
import { WorkspaceHeader } from "./components/chrome/WorkspaceHeader";
import { useStore } from "./state/store";
import { useDragGuard } from "./components/hooks/useDragGuard";
import { useKeyboard } from "./components/hooks/useKeyboard";
import { useOperationSelection } from "./components/hooks/useOperationSelection";
import { useEditTileRelease } from "./components/hooks/useEditTileRelease";
import { useFocusPolicy } from "./components/hooks/useFocusPolicy";
import { ShortcutsDialog } from "./components/chrome/ShortcutsDialog";

export default function App(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const theme = useStore((s) => s.theme);
  const canvasTool = useStore((s) => s.canvasTool);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const showShortcuts = useCallback(() => setShortcutsOpen(true), []);
  const closeShortcuts = useCallback(() => setShortcutsOpen(false), []);

  useKeyboard({ shortcutsOpen, showShortcuts, closeShortcuts });
  useFocusPolicy();
  useOperationSelection();
  useEditTileRelease();
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
      localStorage.setItem("tilecone.canvasTool", canvasTool);
    } catch {
      // As with the theme, persistence is optional: the setting still applies
      // to this session.
    }
  }, [canvasTool]);

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
