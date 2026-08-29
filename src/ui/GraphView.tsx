import dagre from "dagre";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { isExpandable } from "../core/expand";
import { cardSize, TensorCard } from "./TensorCard";
import { useStore } from "./store";

type Placed = {
  id: string;
  kind: "tensor" | "op";
  x: number;
  y: number;
  w: number;
  h: number;
};

type EdgeLine = { from: string; to: string; points: { x: number; y: number }[]; hot: boolean };

export function GraphView(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const viewCfgs = useStore((s) => s.viewCfgs);
  const hideInert = useStore((s) => s.hideInert);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const selection = useStore((s) => s.selection);
  const expandNodeInPlace = useStore((s) => s.expandNodeInPlace);
  const focusTensor = useStore((s) => s.focusTensor);

  const [tf, setTf] = useState({ x: 20, y: 20, k: 1 });
  const containerRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ x0: number; y0: number; tx: number; ty: number } | null>(null);

  const contributing = useMemo(() => {
    const s = new Set<string>();
    if (selection) s.add(selection.tensorId);
    for (const res of [backwardRes, forwardRes])
      if (res) for (const id of res.tensors.keys()) s.add(id);
    return s;
  }, [backwardRes, forwardRes, selection]);

  const hasResult = contributing.size > 0;

  // Canvas backing-store multiplier, bucketed to powers of two so that zooming
  // re-renders the cards sharply without redrawing on every wheel tick.
  const renderScale = useMemo(
    () => Math.min(4, Math.max(1, 2 ** Math.ceil(Math.log2(Math.max(1, tf.k))))),
    [tf.k]
  );

  const layout = useMemo(() => {
    if (!resolved) return { placed: [] as Placed[], edges: [] as EdgeLine[], w: 800, h: 600 };
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "LR", nodesep: 26, ranksep: 46, marginx: 20, marginy: 20 });
    g.setDefaultEdgeLabel(() => ({}));

    const nodeHot = (nid: string) => {
      const n = resolved.nodes.find((x) => x.id === nid)!;
      return [...n.inputs, ...n.outputs].some((t) => contributing.has(t));
    };
    const skipTensor = (tid: string) => hideInert && hasResult && !contributing.has(tid);
    const skipOp = (nid: string) => hideInert && hasResult && !nodeHot(nid);

    for (const t of Object.values(resolved.tensors)) {
      if (skipTensor(t.id)) continue;
      const cfg = viewCfgs[t.id];
      const { w, h } = cfg ? cardSize(t.resolved!, cfg) : { w: 230, h: 120 };
      g.setNode(`t:${t.id}`, { width: w, height: h });
    }
    for (const n of resolved.nodes) {
      if (skipOp(n.id)) continue;
      g.setNode(`n:${n.id}`, { width: Math.max(64, n.op.length * 8 + 22), height: 30 });
      for (const t of n.inputs) if (!skipTensor(t)) g.setEdge(`t:${t}`, `n:${n.id}`);
      for (const t of n.outputs) if (!skipTensor(t)) g.setEdge(`n:${n.id}`, `t:${t}`);
    }
    dagre.layout(g);
    const placed: Placed[] = [];
    for (const id of g.nodes()) {
      const nd = g.node(id);
      if (!nd) continue;
      placed.push({
        id: id.slice(2),
        kind: id.startsWith("t:") ? "tensor" : "op",
        x: nd.x - nd.width / 2,
        y: nd.y - nd.height / 2,
        w: nd.width,
        h: nd.height,
      });
    }
    const edges: EdgeLine[] = [];
    for (const e of g.edges()) {
      const pts = g.edge(e).points ?? [];
      const fromT = e.v.startsWith("t:") ? e.v.slice(2) : e.w.slice(2);
      const opId = e.v.startsWith("n:") ? e.v.slice(2) : e.w.slice(2);
      const hot = hasResult && contributing.has(fromT) && nodeHot(opId);
      edges.push({ from: e.v, to: e.w, points: pts, hot });
    }
    const gr = g.graph();
    return { placed, edges, w: gr.width ?? 800, h: gr.height ?? 600 };
  }, [resolved, viewCfgs, hideInert, contributing, hasResult]);

  const fit = () => {
    const el = containerRef.current;
    if (!el) return;
    const k = Math.min(el.clientWidth / (layout.w + 40), el.clientHeight / (layout.h + 40), 1.25);
    setTf({ x: 20 * k, y: 20 * k, k });
  };

  useEffect(() => {
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolved]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
      if (e.key === "f") fit(); // the rest of the bindings live in useKeyboard
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);

  useEffect(() => {
    if (!focusTensor) return;
    const p = layout.placed.find((x) => x.kind === "tensor" && x.id === focusTensor);
    const el = containerRef.current;
    if (p && el)
      setTf((t) => ({
        ...t,
        x: el.clientWidth / 2 - (p.x + p.w / 2) * t.k,
        y: el.clientHeight / 2 - (p.y + p.h / 2) * t.k,
      }));
  }, [focusTensor, layout]);

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const el = containerRef.current!;
    const rect = el.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setTf((t) => {
      const k = Math.min(12, Math.max(0.08, t.k * Math.exp(-e.deltaY * 0.0012)));
      const scale = k / t.k;
      return { k, x: mx - (mx - t.x) * scale, y: my - (my - t.y) * scale };
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.target !== e.currentTarget) return; // only background pans
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    panRef.current = { x0: e.clientX, y0: e.clientY, tx: tf.x, ty: tf.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const p = panRef.current;
    if (!p) return;
    setTf((t) => ({ ...t, x: p.tx + e.clientX - p.x0, y: p.ty + e.clientY - p.y0 }));
  };
  const onPointerUp = () => (panRef.current = null);

  if (!resolved) return <div className="canvas-empty">no graph loaded</div>;

  return (
    <div
      ref={containerRef}
      className="graph-canvas"
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <button className="fit-btn" onClick={fit} title="fit to view (f)">
        ⤢ fit
      </button>
      <div
        className="graph-inner"
        style={{ transform: `translate(${tf.x}px, ${tf.y}px) scale(${tf.k})`, width: layout.w, height: layout.h }}
      >
        <svg className="edges" width={layout.w} height={layout.h}>
          {layout.edges.map((e, i) => (
            <path
              key={i}
              d={e.points.map((p, j) => `${j === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ")}
              className={hasResult ? (e.hot ? "edge hot" : "edge dim") : "edge"}
            />
          ))}
        </svg>
        {layout.placed.map((p) => {
          if (p.kind === "op") {
            const node = resolved.nodes.find((n) => n.id === p.id)!;
            const hot = !hasResult || [...node.inputs, ...node.outputs].some((t) => contributing.has(t));
            return (
              <div
                key={`n:${p.id}`}
                className={`op-node${hot ? "" : " dim"}`}
                style={{ left: p.x, top: p.y, width: p.w, height: p.h }}
                title={`${node.op}\n${JSON.stringify(node.attrs)}`}
              >
                <span>{node.label ?? node.op}</span>
                {isExpandable(node.op) && (
                  <button
                    className="expand-btn"
                    title="expand into primitive ops"
                    onClick={(ev) => {
                      ev.stopPropagation();
                      expandNodeInPlace(p.id);
                    }}
                  >
                    ⊞
                  </button>
                )}
              </div>
            );
          }
          const t = resolved.tensors[p.id];
          const hot = !hasResult || contributing.has(p.id);
          return (
            <div
              key={`t:${p.id}`}
              className={hot ? "card-slot" : "card-slot dim"}
              style={{ left: p.x, top: p.y, width: p.w }}
            >
              <TensorCard tensor={t} renderScale={renderScale} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
