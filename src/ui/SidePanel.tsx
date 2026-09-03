import React, { useEffect, useMemo, useRef, useState } from "react";
import { fromBox } from "../core/region";
import { EXAMPLES } from "../examples";
import { tileOf } from "./grid";
import { selectionToLink, shareTarget } from "./share";
import { copyText } from "./clipboard";
import { enabledPropResult, useStore, viewAxes } from "./store";

/**
 * Copies a link that restores this workspace — source, selection, cone direction
 * and tile detail. It sits with the source actions rather than in the header
 * because the source is most of what it encodes.
 */
function ShareButton(): React.ReactElement {
  const dslText = useStore((s) => s.dslText);
  const selection = useStore((s) => s.selection);
  const direction = useStore((s) => s.direction);
  const tileScale = useStore((s) => s.tileScale);
  const snapToGrid = useStore((s) => s.snapToGrid);
  const axisMode = useStore((s) => s.axisMode);
  const tensorOffsets = useStore((s) => s.tensorOffsets);
  const [copyState, setCopyState] = useState<"copied" | "failed" | null>(null);

  const copy = async () => {
    const target = shareTarget(location.origin, location.pathname, {
      dsl: dslText,
      dir: direction,
      tile: tileScale,
      snap: snapToGrid,
      axes: axisMode,
      pos: Object.fromEntries(
        Object.entries(tensorOffsets).map(([id, { dx, dy }]) => [id, [dx, dy]])
      ),
      sel: selectionToLink(selection),
    });
    setCopyState((await copyText(target)) ? "copied" : "failed");
    setTimeout(() => setCopyState(null), 1600);
  };

  return (
    <button
      className={`mini share-btn${copyState === "failed" ? " copy-failed" : ""}`}
      onClick={copy}
      title="copy a link that restores this source, selection, graph layout, needs view and feeds view"
      aria-live="polite"
    >
      {copyState === "copied" ? "copied ✓" : copyState === "failed" ? "copy failed" : "share"}
    </button>
  );
}

/** The graph source, editable in place. Ctrl/Cmd+Enter runs it. */
function SourceEditor(): React.ReactElement {
  const dslText = useStore((s) => s.dslText);
  const applyDSL = useStore((s) => s.applyDSL);
  const loadError = useStore((s) => s.loadError);
  const [text, setText] = useState(dslText);
  const [ranAt, setRanAt] = useState(0);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Follow the store when a different example is loaded, but never clobber
  // edits in progress.
  useEffect(() => setText(dslText), [dslText]);

  const dirty = text !== dslText;
  const run = () => {
    applyDSL(text);
    setRanAt(Date.now());
  };

  const errorLine = useMemo(() => {
    const m = loadError && /line (\d+)/.exec(loadError);
    return m ? Number(m[1]) : null;
  }, [loadError]);

  return (
    <>
      <textarea
        ref={taRef}
        className={loadError ? "source error-state" : "source"}
        value={text}
        aria-label="graph source"
        spellCheck={false}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            run();
          }
        }}
      />
      <div className="source-actions">
        <button className="run-btn" onClick={run} disabled={!dirty && !loadError}>
          ▶ run
        </button>
        <ShareButton />
        <span className="source-status">
          {loadError ? (
            <span className="error">
              {errorLine !== null && <b>line {errorLine}: </b>}
              {loadError.replace(/^line \d+: /, "")}
            </span>
          ) : dirty ? (
            <span className="muted">unrun changes · ⌘/ctrl+↵</span>
          ) : ranAt ? (
            <span className="ok">✓ graph built</span>
          ) : (
            <span className="muted">⌘/ctrl+↵ to run</span>
          )}
        </span>
      </div>
    </>
  );
}

/** Prototype-style operation list. Clicking a row probes its first output. */
function Operations(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const setFocusTensor = useStore((s) => s.setFocusTensor);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const setSelection = useStore((s) => s.setSelection);
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);

  if (!resolved) return <p className="hint">no graph</p>;

  const involved = new Set<string>();
  const enabledBackward = enabledPropResult(backwardRes, perBox, hiddenBoxes, null, "backward");
  const enabledForward = enabledPropResult(forwardRes, perBox, hiddenBoxes, null, "forward");
  for (const res of [enabledBackward, enabledForward])
    if (res) for (const id of res.tensors.keys()) involved.add(id);

  const probe = (tensorId: string) => {
    const shape = resolved.tensors[tensorId].resolved!;
    setFocusTensor(tensorId);
    if (shape.some((extent) => extent <= 0)) return;
    const { rowAxis, colAxis } = viewAxes(shape);
    const tile = tileOf(shape, tileScale, graphPx);
    setSelection(
      tensorId,
      fromBox(
        shape.map((extent, axis) => ({
          lo: 0,
          hi: axis === rowAxis || axis === colAxis ? Math.min(tile, extent) : Math.min(1, extent),
        }))
      ),
      "replace"
    );
  };

  return (
    <div className="operation-list">
      {resolved.topo.map((node) => {
        const outputs = node.outputs.map((id) => resolved.tensors[id]);
        const hot = [...node.inputs, ...node.outputs].some((id) => involved.has(id));
        const signature = `${outputs.map((t) => t.name).join(", ")} = ${node.op}(${node.inputs
          .map((id) => resolved.tensors[id].name)
          .join(", ")})`;
        const meta = outputs
          .map((t) => `[${t.resolved!.join(" × ")}] ${t.dtype}`)
          .join(" · ");
        return (
          <button
            key={node.id}
            className={`operation-row${involved.size ? (hot ? " hot" : " dim") : ""}`}
            title={`select a starter tile on ${outputs[0].name}\n${JSON.stringify(node.attrs)}`}
            onClick={() => probe(outputs[0].id)}
          >
            <i />
            <span>
              <code>{signature}</code>
              <small>{meta}</small>
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function SidePanel(): React.ReactElement {
  const exampleIndex = useStore((s) => s.exampleIndex);
  const loadExample = useStore((s) => s.loadExample);

  return (
    <nav className="side-panel">
      <div className="side-panel-scroll">
        <header className="source-heading">
          <h2 className="panel-title">Graph source</h2>
          <p>
            Declare dimensions in <code>params</code>, tensors as <code>input</code>, <code>weight</code>,
            or <code>const</code>. Shapes are inferred when the graph is rendered.
          </p>
        </header>
        <div className="source-workspace">
          <SourceEditor />
        </div>

        <div className="side-kicker">try an example</div>
        <div className="example-list">
          {EXAMPLES.map((example, index) => (
            <button
              key={example.name}
              className={exampleIndex === index ? "on" : ""}
              title={example.name}
              aria-pressed={exampleIndex === index}
              onClick={() => loadExample(index)}
            >
              {example.name}
            </button>
          ))}
        </div>

        <div className="side-divider" />
        <h3 className="panel-title operations-heading">Operations</h3>
        <Operations />
      </div>
    </nav>
  );
}
