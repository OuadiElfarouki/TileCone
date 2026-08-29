import React, { useState } from "react";
import { useStore } from "./store";

export function GraphOutline(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const setFocusTensor = useStore((s) => s.setFocusTensor);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(true);

  if (!resolved) return <nav className="outline" />;

  const involved = new Set<string>();
  for (const res of [backwardRes, forwardRes])
    if (res) for (const id of res.tensors.keys()) involved.add(id);

  const match = (s: string) => !q || s.toLowerCase().includes(q.toLowerCase());

  return (
    <nav className="outline">
      <div className="outline-head">
        <button className="mini" onClick={() => setOpen(!open)}>{open ? "▾" : "▸"}</button>
        <input placeholder="search…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      {open && (
        <ul>
          {resolved.topo.map((n) => (
            <li key={n.id}>
              <span className="node-label" title={JSON.stringify(n.attrs)}>
                {n.op}
              </span>
              <ul>
                {n.outputs.filter(match).map((tid) => {
                  const t = resolved.tensors[tid];
                  const cls =
                    selection?.tensorId === tid ? "sel" : involved.has(tid) ? "hot" : "";
                  return (
                    <li key={tid}>
                      <button className={`tensor-link ${cls}`} onClick={() => setFocusTensor(tid)}>
                        {t.name} <span className="muted">[{t.resolved!.join("×")}]</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </li>
          ))}
          <li>
            <span className="node-label">inputs</span>
            <ul>
              {Object.values(resolved.tensors)
                .filter((t) => !t.producer && match(t.name))
                .map((t) => {
                  const cls =
                    selection?.tensorId === t.id ? "sel" : involved.has(t.id) ? "hot" : "";
                  return (
                    <li key={t.id}>
                      <button className={`tensor-link ${cls}`} onClick={() => setFocusTensor(t.id)}>
                        {t.name} <span className="muted">[{t.resolved!.join("×")}]</span>
                      </button>
                    </li>
                  );
                })}
            </ul>
          </li>
        </ul>
      )}
    </nav>
  );
}
