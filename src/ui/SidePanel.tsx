import React, { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "./store";

/** Collapsible titled section. Both panes share the panel's vertical space. */
function Section({
  title,
  open,
  onToggle,
  grow,
  right,
  children,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  grow?: boolean;
  right?: React.ReactNode;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <section className={`panel-section${open ? " open" : ""}${open && grow ? " grow" : ""}`}>
      <header onClick={onToggle}>
        <span className="chevron">{open ? "▾" : "▸"}</span>
        <span className="panel-title">{title}</span>
        <span className="panel-head-right" onClick={(e) => e.stopPropagation()}>
          {right}
        </span>
      </header>
      {open && <div className="panel-body">{children}</div>}
    </section>
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

/** Searchable list of the graph's nodes and tensors. */
function Outline(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const setFocusTensor = useStore((s) => s.setFocusTensor);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const [q, setQ] = useState("");

  if (!resolved) return <p className="hint">no graph</p>;

  const involved = new Set<string>();
  for (const res of [backwardRes, forwardRes])
    if (res) for (const id of res.tensors.keys()) involved.add(id);

  const match = (s: string) => !q || s.toLowerCase().includes(q.toLowerCase());
  const cls = (id: string) =>
    selection?.tensorId === id ? "sel" : involved.has(id) ? "hot" : "";

  const inputs = Object.values(resolved.tensors).filter((t) => !t.producer && match(t.name));

  return (
    <>
      <input
        className="outline-search"
        placeholder="filter tensors…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <ul className="outline-list">
        {inputs.length > 0 && (
          <li>
            <span className="node-label">inputs</span>
            <ul>
              {inputs.map((t) => (
                <li key={t.id}>
                  <button className={`tensor-link ${cls(t.id)}`} onClick={() => setFocusTensor(t.id)}>
                    {t.name} <span className="muted">{t.resolved!.join("×")}</span>
                  </button>
                </li>
              ))}
            </ul>
          </li>
        )}
        {resolved.topo.map((n) => {
          const outs = n.outputs.filter(match);
          if (!outs.length) return null;
          return (
            <li key={n.id}>
              <span className="node-label" title={JSON.stringify(n.attrs)}>
                {n.op}
              </span>
              <ul>
                {outs.map((tid) => {
                  const t = resolved.tensors[tid];
                  return (
                    <li key={tid}>
                      <button className={`tensor-link ${cls(tid)}`} onClick={() => setFocusTensor(tid)}>
                        {t.name} <span className="muted">{t.resolved!.join("×")}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </li>
          );
        })}
      </ul>
    </>
  );
}

export function SidePanel(): React.ReactElement {
  const [sourceOpen, setSourceOpen] = useState(true);
  const [outlineOpen, setOutlineOpen] = useState(true);
  const resolved = useStore((s) => s.resolved);
  const nTensors = resolved ? Object.keys(resolved.tensors).length : 0;

  return (
    <nav className="side-panel">
      <Section
        title="source"
        open={sourceOpen}
        grow
        onToggle={() => setSourceOpen(!sourceOpen)}
      >
        <SourceEditor />
      </Section>
      <Section
        title="graph"
        open={outlineOpen}
        grow
        onToggle={() => setOutlineOpen(!outlineOpen)}
        right={<span className="muted count">{nTensors} tensors</span>}
      >
        <Outline />
      </Section>
    </nav>
  );
}
