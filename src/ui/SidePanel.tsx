import React, { useEffect, useMemo, useRef, useState } from "react";
import { fromBox } from "../core/region";
import { EXAMPLES } from "../examples";
import { tileOf } from "./grid";
import { selectionToLink, shareTarget } from "./share";
import { copyText } from "./clipboard";
import { enabledPropResult, useStore } from "./store";
import { matchesShortcut, SHORTCUTS } from "./shortcuts";
import { viewAxes } from "./tensor-view";
import { highlightDSL } from "./dsl-highlight";

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
  const text = useStore((s) => s.draftText);
  const setText = useStore((s) => s.setDraftText);
  const applyDSL = useStore((s) => s.applyDSL);
  const loadError = useStore((s) => s.loadError);
  const diagnostics = useStore((s) => s.diagnostics);
  const built = useStore((s) => s.resolved !== null);
  const [ranAt, setRanAt] = useState(0);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLPreElement>(null);
  const highlighted = useMemo(() => highlightDSL(text), [text]);

  // Two ways to owe a run: the text has moved away from what was built, or the
  // app has not installed its initial graph yet.
  const dirty = text !== dslText;
  const unbuilt = dirty || !built;
  const run = () => {
    applyDSL(text);
    setRanAt(Date.now());
  };

  return (
    <>
      <div className="source-editor">
        <pre className="source-highlight" ref={highlightRef} aria-hidden="true">
          {highlighted.map((token, index) => (
            <span key={index} className={`syntax-${token.kind}`}>{token.text}</span>
          ))}
        </pre>
        <textarea
          ref={taRef}
          className={loadError ? "source error-state" : "source"}
          value={text}
          aria-label="graph source"
          spellCheck={false}
          onChange={(e) => setText(e.target.value)}
          onScroll={(e) => {
            if (!highlightRef.current) return;
            highlightRef.current.scrollTop = e.currentTarget.scrollTop;
            highlightRef.current.scrollLeft = e.currentTarget.scrollLeft;
          }}
          onKeyDown={(e) => {
            if (matchesShortcut(e.nativeEvent, SHORTCUTS.run)) {
              e.preventDefault();
              run();
            }
          }}
        />
      </div>
      <div className="source-actions">
        <button className="run-btn" onClick={run} disabled={!unbuilt && !loadError}>
          ▶ run
        </button>
        <ShareButton />
        <span className="source-status">
          {diagnostics.length ? (
            /* Every independent error, not just the first. The compiler finds
               them in one pass, and showing one at a time would put the author
               back on the fix-and-recompile loop that collecting exists to
               end. */
            <span className="error">
              {diagnostics.length > 1 && (
                <b>{diagnostics.length} errors · </b>
              )}
              {diagnostics.map((d, i) => (
                <span key={i} className="diag">
                  <b>line {d.span.start.line}: </b>
                  {d.message}
                </span>
              ))}
            </span>
          ) : loadError ? (
            <span className="error">{loadError}</span>
          ) : dirty ? (
            <span className="muted">unrun changes · ⌘/ctrl+↵</span>
          ) : !built ? (
            <span className="muted">not built yet · ⌘/ctrl+↵</span>
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
  const setFocusNode = useStore((s) => s.setFocusNode);
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

  // The row names an operation, so the viewport goes to the operator. The
  // starter tile still lands on its first output, which is where a cone has to
  // begin - the two were the same request while only tensors could be focused.
  const probe = (nodeId: string, tensorId: string) => {
    const shape = resolved.tensors[tensorId].resolved!;
    setFocusNode({ kind: "op", id: nodeId });
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
            onClick={() => probe(node.id, outputs[0].id)}
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

/**
 * The example picker is a menu rather than a row of chips. Eight names wrapped
 * to three lines above the operation list and read as a control surface with
 * more weight than it has: the examples are a way in, not part of the answer.
 * It is drawn from the same parts as everything else here - a bordered trigger
 * that goes accent when open, a floating surface at the popover radius - so it
 * is a smaller version of the panel, not a native widget dropped into it.
 */
function ExamplePicker(): React.ReactElement {
  const exampleIndex = useStore((s) => s.exampleIndex);
  const draftText = useStore((s) => s.draftText);
  const stageExample = useStore((s) => s.stageExample);
  const [open, setOpen] = useState(false);
  // Which row the keyboard is on. It follows the current example when the menu
  // opens, so ↓ from a loaded example moves to the next one rather than to the
  // top of the list.
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Two different questions, and the picker answers both at once. What is *in
  // the editor* names the trigger; what is actually *built* takes the accent.
  // Between picking and running they disagree, and that gap is the point: the
  // menu proposes a source, the run button commits it.
  const staged = EXAMPLES.findIndex((ex) => ex.dsl === draftText);
  const current = staged >= 0 ? EXAMPLES[staged] : null;

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  useEffect(() => {
    if (open) listRef.current?.focus();
  }, [open]);

  // The menu shows three and a half rows, so the keyboard can walk the
  // highlight out of view. Follow it. `nearest` keeps a pointer-driven change
  // from jumping the list under the cursor.
  useEffect(() => {
    if (!open) return;
    listRef.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const show = () => {
    setActive(staged >= 0 ? staged : 0);
    setOpen(true);
  };

  const choose = (index: number) => {
    if (index !== staged) stageExample(index);
    setOpen(false);
    triggerRef.current?.focus();
  };

  const onListKeyDown = (e: React.KeyboardEvent) => {
    const step = (delta: number) => {
      e.preventDefault();
      setActive((i) => (i + delta + EXAMPLES.length) % EXAMPLES.length);
    };
    if (e.key === "ArrowDown") step(1);
    else if (e.key === "ArrowUp") step(-1);
    else if (e.key === "Home") { e.preventDefault(); setActive(0); }
    else if (e.key === "End") { e.preventDefault(); setActive(EXAMPLES.length - 1); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(active); }
    else if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
    } else if (e.key === "Tab") setOpen(false);
  };

  return (
    <div className="example-picker" ref={rootRef}>
      <span className="side-kicker" id="example-picker-label">try an example</span>
      <div className="example-menu">
        <button
          ref={triggerRef}
          className={`example-trigger${open ? " on" : ""}`}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={open ? "example-options" : undefined}
          aria-labelledby="example-picker-label example-picker-value"
          title={
            current
              ? exampleIndex === staged
                ? `${current.name} · running`
                : `${current.name} · loaded, not run yet`
              : "load one of the built-in graphs"
          }
          onClick={() => (open ? setOpen(false) : show())}
          onKeyDown={(e) => {
            if (!open && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
              e.preventDefault();
              show();
            }
          }}
        >
          <span
            id="example-picker-value"
            className={`example-current${current ? "" : " none"}${
              current && exampleIndex === staged ? " built" : ""
            }`}
          >
            {current ? current.name : "custom source"}
          </span>
          <i aria-hidden="true">▾</i>
        </button>
        {open && (
          <ul
            id="example-options"
            className="example-options"
            role="listbox"
            tabIndex={-1}
            ref={listRef}
            aria-labelledby="example-picker-label"
            aria-activedescendant={`example-option-${active}`}
            onKeyDown={onListKeyDown}
          >
            {EXAMPLES.map((example, index) => (
              <li
                key={example.name}
                id={`example-option-${index}`}
                role="option"
                // The widget's value is what it would run, not what is running.
                aria-selected={staged === index}
                className={`example-option${index === active ? " active" : ""}${
                  exampleIndex === index ? " on" : ""
                }${staged === index && exampleIndex !== index ? " staged" : ""}`}
                onPointerEnter={() => setActive(index)}
                onClick={() => choose(index)}
              >
                <span>{example.name}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export function SidePanel(): React.ReactElement {
  const sourcePending = useStore((s) => s.draftText !== s.dslText);
  return (
    <nav className="side-panel">
      <div className="side-panel-scroll">
        <header className="source-heading">
          <h2 className="panel-title">Graph source</h2>
          <p>
            Assign dimensions as numbers, graph inputs with <code>Tensor</code>, and learned weights
            with <code>Parameter</code>. Shapes are inferred when the graph is rendered.
          </p>
        </header>
        <div className="source-workspace">
          <SourceEditor />
        </div>

        <ExamplePicker />

        <div className="side-divider" />
        <section
          className={`operations-section${sourcePending ? " pending" : ""}`}
          title={sourcePending ? "Operations from the currently built graph" : undefined}
        >
          <div className="operations-heading">
            <h3 className="panel-title">Operations</h3>
            {sourcePending && <span className="operations-pending">built graph</span>}
          </div>
          <Operations />
        </section>
      </div>
    </nav>
  );
}
