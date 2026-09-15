import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { fromBox } from "../core/region";
import { EXAMPLES } from "../examples";
import { tileOf } from "./grid";
import { selectionToLink, shareTarget } from "./share";
import { CopyButton } from "./CopyButton";
import { opLabel } from "../core/ops/index";
import { dslTextOf, exampleIndexOf, involvedTensorIds, useStore } from "./store";
import { formatReport, reportHeadline, reportLines } from "../import/report";
import { matchesShortcut, SHORTCUTS } from "./shortcuts";
import { viewAxes } from "./tensor-view";
import { overlayTokens } from "./dsl-highlight";

/**
 * Copies a link that restores this workspace - source, selection, analysis views
 * and tile detail. It sits with the source actions rather than in the header
 * because the source is most of what it encodes.
 */
function ShareButton(): React.ReactElement {
  // A link carries the source as DSL text, and an imported model has none. The
  // honest answer is to say so rather than to share a link that would restore
  // an empty workspace, or to silently share the text of whatever was open
  // before the import. This is the declared limitation, not an oversight.
  const imported = useStore((s) => s.source.kind === "import");
  if (imported)
    return (
      <button
        className="mini share-btn"
        disabled
        title="an imported model is not shareable as a link: a link carries DSL source, and this workspace has none"
      >
        share
      </button>
    );

  // Read at click time rather than subscribing: the link is built from nine
  // pieces of state and none of them change how this button looks.
  const link = () => {
    const s = useStore.getState();
    return shareTarget(location.origin, location.pathname, {
      dsl: dslTextOf(s.source) ?? "",
      dir: s.direction,
      ent: s.showEntangled,
      tile: s.tileScale,
      snap: s.snapToGrid,
      axes: s.axisMode,
      views: Object.fromEntries(
        Object.entries(s.viewCfgs).filter(
          ([, cfg]) => !cfg.projection || cfg.sliders.some((v) => v !== 0)
        )
      ),
      pos: Object.fromEntries(
        Object.entries(s.tensorOffsets).map(([id, { dx, dy }]) => [id, [dx, dy]])
      ),
      sel: selectionToLink(s.selection),
    });
  };

  return (
    <CopyButton
      className="mini share-btn"
      title="copy a link that restores this source, selection, graph layout, and analysis views"
      label="share"
      text={link}
    />
  );
}

/**
 * Open a converted model.
 *
 * JSON only, and that is the boundary rather than a placeholder for one: a
 * converter that runs where the shape inference actually exists hands its
 * result over as a document, and this is where the document lands. A decoder
 * that reads bytes in the browser will call the same store action.
 */
function OpenModelButton(): React.ReactElement {
  const importJSON = useStore((s) => s.importJSON);
  const reportImportError = useStore((s) => s.reportImportError);
  const input = useRef<HTMLInputElement>(null);

  return (
    <>
      <button
        className="mini open-model"
        title="open a converted model (JSON graph or import document)"
        onClick={() => input.current?.click()}
      >
        open model
      </button>
      <input
        ref={input}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={async (event) => {
          const file = event.target.files?.[0];
          // Cleared before the await: the same file picked twice in a row fires
          // no change event otherwise, so a failed import could not be retried.
          event.target.value = "";
          if (!file) return;
          try {
            importJSON(await file.text(), { fileName: file.name });
          } catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            reportImportError(`could not read "${file.name}": ${detail}`);
          }
        }}
      />
    </>
  );
}

/**
 * What an imported workspace shows where the editor would be.
 *
 * Not an editor, because there is no text behind an imported graph and
 * generating some would rename the very nodes the report addresses. What the
 * reader needs instead is the conversion's own account of itself: what was
 * mapped, what was assumed, and which nodes are barriers whose regions are
 * bounds rather than counts. Two barriers named here is a fact about precision
 * you can act on; two dropped nodes would be a wrong answer you could not see.
 */
function ImportSummary(): React.ReactElement {
  const report = useStore((s) => (s.source.kind === "import" ? s.source.report : null));
  const setFocusNode = useStore((s) => s.setFocusNode);
  if (!report) return <p className="hint">no import</p>;
  const lines = reportLines(report);

  return (
    <>
      <div className="import-report">
        <p className="import-headline">{reportHeadline(report)}</p>
        {lines.length ? (
          <dl className="import-lines">
            {lines.map((line) => (
              <div key={line.label} className="import-line">
                <dt>{line.label}</dt>
                <dd>
                  {line.nodes.length ? (
                    /* The import path's substitute for a source span: a
                       diagnostic with no text to underline points at a node,
                       and the canvas is where the reader sees it. */
                    <button
                      className="linky"
                      title={`show ${line.nodes.length === 1 ? "this operation" : "the first of these operations"} on the canvas`}
                      onClick={() => setFocusNode({ kind: "op", id: line.nodes[0] })}
                    >
                      {line.text}
                    </button>
                  ) : (
                    line.text
                  )}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="hint">
            the document made no claims about its conversion, so nothing here is
            marked as approximate
          </p>
        )}
      </div>
      <div className="source-actions">
        <CopyButton
          title="copy the import report"
          label="copy report"
          text={() => formatReport(report)}
        />
        <ShareButton />
        <span className="source-status" id="source-status" role="status" aria-live="polite">
          <span className="muted">imported · no source text</span>
        </span>
      </div>
    </>
  );
}

/** Import failures are attempts, not properties of whichever source remains installed. */
function ImportFailures(): React.ReactElement | null {
  const sourceIsImport = useStore((s) => s.source.kind === "import");
  const loadError = useStore((s) => s.loadError);
  const importDiagnostics = useStore((s) => s.importDiagnostics);
  // A failed replacement preserves the installed source. Show its diagnostics
  // in either source mode; store-only refusals such as expanding an imported
  // graph have no ImportDiagnostic, so they use the same surface only while an
  // import remains installed.
  const failures = importDiagnostics.length
    ? importDiagnostics
    : sourceIsImport && loadError
      ? [{ severity: "error" as const, message: loadError }]
      : [];
  if (!failures.length) return null;

  return (
    <div className="import-errors error" role="alert" aria-live="assertive">
      {failures.length > 1 && <b>{failures.length} import errors</b>}
      {failures.map((diagnostic, index) => (
        <span className="diag" key={index}>
          {diagnostic.subject && (
            <b>
              {diagnostic.subject.kind} {diagnostic.subject.id}
              {diagnostic.subject.attribute ? ` · ${diagnostic.subject.attribute}` : ""}
              {": "}
            </b>
          )}
          {diagnostic.message}
        </span>
      ))}
    </div>
  );
}

/** The graph source, editable in place. Ctrl/Cmd+Enter runs it. */
function DSLEditor(): React.ReactElement {
  const dslText = useStore((s) => dslTextOf(s.source) ?? "");
  const text = useStore((s) => s.draftText);
  const setText = useStore((s) => s.setDraftText);
  const applyDSL = useStore((s) => s.applyDSL);
  // An import attempt must not mark valid installed DSL as invalid. It has its
  // own source-independent alert above this editor.
  const loadError = useStore((s) => (s.importDiagnostics.length ? null : s.loadError));
  const diagnostics = useStore((s) => s.diagnostics);
  const built = useStore((s) => s.resolved !== null);
  const [ranAt, setRanAt] = useState(0);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLPreElement>(null);
  const highlighted = useMemo(() => overlayTokens(text), [text]);

  // Two ways to owe a run: the text has moved away from what was built, or the
  // app has not installed its initial graph yet.
  const dirty = text !== dslText;
  const unbuilt = dirty || !built;
  const run = () => {
    applyDSL(text);
    setRanAt(Date.now());
  };

  /* The overlay carries the ink for a caret it cannot see, so it has to hold
     the textarea's scroll offset exactly. The scroll event covers the usual
     case; this covers the one it does not, where new text shortens the box and
     the browser clamps the offset, which it reports only after the commit. */
  const syncScroll = () => {
    const ta = taRef.current;
    const highlight = highlightRef.current;
    if (!ta || !highlight) return;
    highlight.scrollTop = ta.scrollTop;
    highlight.scrollLeft = ta.scrollLeft;
  };
  useLayoutEffect(syncScroll, [text]);

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
          aria-invalid={!!loadError}
          aria-describedby="source-status"
          aria-errormessage={loadError ? "source-status" : undefined}
          spellCheck={false}
          onChange={(e) => setText(e.target.value)}
          onScroll={syncScroll}
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
        <span
          className="source-status"
          id="source-status"
          role={loadError ? "alert" : "status"}
          aria-live={loadError ? "assertive" : "polite"}
        >
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
  const setSelectedOp = useStore((s) => s.setSelectedOp);
  const selectedOp = useStore((s) => s.selectedOp);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const direction = useStore((s) => s.direction);
  const selection = useStore((s) => s.selection);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const setSelection = useStore((s) => s.setSelection);
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);

  if (!resolved) return <p className="hint">no graph</p>;

  /* The same set the canvas lights, so hiding a cone dims its operations here
     too rather than leaving the list on the union of both directions. */
  const involved = involvedTensorIds(selection, backwardRes, forwardRes, perBox, hiddenBoxes, direction);

  // The row names an operation, so the viewport goes to the operator. The
  // starter tile still lands on its first output, which is where a cone has to
  // begin - the two were the same request while only tensors could be focused.
  const probe = (nodeId: string, tensorId: string) => {
    const shape = resolved.tensors[tensorId].resolved!;
    setFocusNode({ kind: "op", id: nodeId });
    // `setSelection` below lights the row for whatever produced the tensor it
    // lands on, which for an operation's own output is this row - but only when
    // the tile is actually placed. A degenerate output takes the early return,
    // and the click should still show which row was pressed.
    setSelectedOp(nodeId);
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
        const signature = `${outputs.map((t) => t.name).join(", ")} = ${opLabel(node)}(${node.inputs
          .map((id) => resolved.tensors[id].name)
          .join(", ")})`;
        const meta = outputs
          .map((t) => `[${t.resolved!.join(" × ")}] ${t.dtype}`)
          .join(" · ");
        return (
          <button
            key={node.id}
            className={`operation-row${involved.size ? (hot ? " hot" : " dim") : ""}${
              node.id === selectedOp ? " selected" : ""
            }`}
            aria-current={node.id === selectedOp || undefined}
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
  const imported = useStore((s) => s.source.kind === "import");
  const exampleIndex = useStore((s) => exampleIndexOf(s.source));
  const draftText = useStore((s) => s.draftText);
  const chooseExample = useStore((s) => s.chooseExample);
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
    // There is no editor in an imported workspace, so staging text would be an
    // invisible no-op. The picker labels this as replacement there and commits
    // the selected example immediately; DSL workspaces keep their preview/run
    // transaction.
    if (imported || index !== staged) chooseExample(index);
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
      <span className="side-kicker" id="example-picker-label">
        {imported ? "replace with example" : "try an example"}
      </span>
      <div className="example-menu">
        <button
          ref={triggerRef}
          className={`example-trigger${open ? " on" : ""}`}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={open ? "example-options" : undefined}
          aria-labelledby="example-picker-label example-picker-value"
          title={
            imported
              ? "replace the imported model with one of the built-in DSL graphs"
              : current
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
            {imported ? "choose replacement" : current ? current.name : "custom source"}
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
  const imported = useStore((s) => s.source.kind === "import");
  // An imported workspace has no built source to be pending against: the draft
  // is empty and the graph did not come from it, so the operations list is
  // never showing an older graph than the editor.
  const sourcePending = useStore((s) => s.source.kind === "dsl" && s.draftText !== s.source.text);
  return (
    <aside className="side-panel" aria-label="Graph source and operations">
      <div className="side-panel-scroll">
        <header className="source-heading">
          <h2 className="panel-title">{imported ? "Imported model" : "Graph source"}</h2>
          <OpenModelButton />
        </header>
        <div className="source-workspace">
          <ImportFailures />
          {imported ? <ImportSummary /> : <DSLEditor />}
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
    </aside>
  );
}
