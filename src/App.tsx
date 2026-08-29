import React, { useEffect, useState } from "react";
import { GraphOutline } from "./ui/GraphOutline";
import { GraphView } from "./ui/GraphView";
import { Inspector } from "./ui/Inspector";
import { Toolbar } from "./ui/Toolbar";
import { useStore } from "./ui/store";
import { useKeyboard } from "./ui/useKeyboard";
import { fromBox } from "./core/region";

function EditorDrawer(): React.ReactElement | null {
  const editorOpen = useStore((s) => s.editorOpen);
  const dslText = useStore((s) => s.dslText);
  const applyDSL = useStore((s) => s.applyDSL);
  const applyJSON = useStore((s) => s.applyJSON);
  const loadError = useStore((s) => s.loadError);
  const [text, setText] = useState(dslText);

  useEffect(() => setText(dslText), [dslText]);
  if (!editorOpen) return null;
  return (
    <div className="editor-drawer">
      <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
      <div className="editor-actions">
        <button onClick={() => applyDSL(text)}>apply DSL</button>
        <button onClick={() => applyJSON(text)}>apply JSON</button>
        <label className="file-btn">
          load file
          <input
            type="file"
            accept=".json,.txt,.dsl"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              f.text().then((s) => {
                setText(s);
                if (f.name.endsWith(".json")) applyJSON(s);
                else applyDSL(s);
              });
            }}
          />
        </label>
        {loadError && <span className="error">{loadError}</span>}
      </div>
    </div>
  );
}

export default function App(): React.ReactElement {
  const loadExample = useStore((s) => s.loadExample);
  const applyDSL = useStore((s) => s.applyDSL);
  const setSelection = useStore((s) => s.setSelection);
  const setDirection = useStore((s) => s.setDirection);
  const loadError = useStore((s) => s.loadError);
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
      <EditorDrawer />
      {loadError && !useStore.getState().editorOpen && <div className="error banner">{loadError}</div>}
      <div className="main">
        <GraphOutline />
        {resolved ? <GraphView /> : <div className="canvas-empty">loading…</div>}
        <Inspector />
      </div>
    </div>
  );
}
