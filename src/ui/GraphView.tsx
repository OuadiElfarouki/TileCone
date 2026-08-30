import dagre from "dagre";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { isExpandable } from "../core/expand";
import { constrainRectMotion, curvedEdgePath, Rect, WORLD_MARGIN } from "./graph-geometry";
import { cardSize, TensorCard } from "./TensorCard";
import { TensorOffset, useStore } from "./store";

type Placed = {
  id: string;
  kind: "tensor" | "op";
  x: number;
  y: number;
  w: number;
  h: number;
};

type EdgeLine = { from: string; to: string; path: string; hot: boolean };

type CardDrag = {
  id: string;
  pointerId: number;
  lastClient: { x: number; y: number };
  rect: Rect;
  offset: TensorOffset;
  before: TensorOffset;
  blockers: Rect[];
  moved: boolean;
};

export function GraphView(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const viewCfgs = useStore((s) => s.viewCfgs);
  const graphPx = useStore((s) => s.graphPx);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const selection = useStore((s) => s.selection);
  const expandNodeInPlace = useStore((s) => s.expandNodeInPlace);
  const focusTensor = useStore((s) => s.focusTensor);
  const setDragging = useStore((s) => s.setDragging);
  const tensorOffsets = useStore((s) => s.tensorOffsets);
  const setTensorOffset = useStore((s) => s.setTensorOffset);
  const commitTensorMove = useStore((s) => s.commitTensorMove);

  const [tf, setTf] = useState({ x: 20, y: 20, k: 1 });
  const [movingTensor, setMovingTensor] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ x0: number; y0: number; tx: number; ty: number } | null>(null);
  const cardDragRef = useRef<CardDrag | null>(null);
  /** True once the user has panned or zoomed away from a fitted view. Resizing
   * the viewport re-fits only while this is false, so collapsing a panel keeps a
   * fitted graph fitted without throwing away a view someone deliberately set. */
  const movedRef = useRef(false);

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
    if (!resolved)
      return { placed: [] as Placed[], solids: [] as Placed[], edges: [] as EdgeLine[], w: 800, h: 600 };
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "LR", nodesep: 26, ranksep: 46, marginx: 20, marginy: 20 });
    g.setDefaultEdgeLabel(() => ({}));

    const nodeHot = (nid: string) => {
      const n = resolved.nodes.find((x) => x.id === nid)!;
      return [...n.inputs, ...n.outputs].some((t) => contributing.has(t));
    };
    for (const t of Object.values(resolved.tensors)) {
      const cfg = viewCfgs[t.id];
      const { w, h } = cfg ? cardSize(t.resolved!, cfg, graphPx, t.name) : { w: 230, h: 120 };
      g.setNode(`t:${t.id}`, { width: w, height: h });
    }
    for (const n of resolved.nodes) {
      const label = n.label ?? n.op;
      g.setNode(`n:${n.id}`, { width: Math.max(64, label.length * 8 + 22), height: 30 });
      for (const t of n.inputs) g.setEdge(`t:${t}`, `n:${n.id}`);
      for (const t of n.outputs) g.setEdge(`n:${n.id}`, `t:${t}`);
    }
    dagre.layout(g);
    const solids: Placed[] = [];
    for (const id of g.nodes()) {
      const nd = g.node(id);
      if (!nd) continue;
      const tensorId = id.startsWith("t:") ? id.slice(2) : null;
      const offset = tensorId ? tensorOffsets[tensorId] ?? { dx: 0, dy: 0 } : { dx: 0, dy: 0 };
      solids.push({
        id: id.slice(2),
        kind: id.startsWith("t:") ? "tensor" : "op",
        x: nd.x - nd.width / 2 + offset.dx,
        y: nd.y - nd.height / 2 + offset.dy,
        w: nd.width,
        h: nd.height,
      });
    }
    const placed = solids;
    const byGraphId = new Map(
      placed.map((p) => [`${p.kind === "tensor" ? "t" : "n"}:${p.id}`, p])
    );
    const edges: EdgeLine[] = [];
    for (const e of g.edges()) {
      const fromT = e.v.startsWith("t:") ? e.v.slice(2) : e.w.slice(2);
      const opId = e.v.startsWith("n:") ? e.v.slice(2) : e.w.slice(2);
      const hot = hasResult && contributing.has(fromT) && nodeHot(opId);
      const from = byGraphId.get(e.v);
      const to = byGraphId.get(e.w);
      if (from && to) edges.push({ from: e.v, to: e.w, path: curvedEdgePath(from, to), hot });
    }
    const gr = g.graph();
    const w = Math.max(gr.width ?? 800, ...solids.map((p) => p.x + p.w + WORLD_MARGIN));
    const h = Math.max(gr.height ?? 600, ...solids.map((p) => p.y + p.h + WORLD_MARGIN));
    return { placed, solids, edges, w, h };
  }, [resolved, viewCfgs, contributing, hasResult, graphPx, tensorOffsets]);

  const fit = () => {
    const el = containerRef.current;
    if (!el) return;
    const k = Math.min(el.clientWidth / (layout.w + 40), el.clientHeight / (layout.h + 40), 1.25);
    setTf({ x: 20 * k, y: 20 * k, k });
    movedRef.current = false;
  };

  useEffect(() => {
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolved]);

  // Re-fit when the viewport changes size — panel collapse/restore, panel drag,
  // window resize. An observer is used rather than a timeout after each of those
  // actions because it fires when layout has actually settled, and it covers
  // window resize (which never re-fitted at all) for free.
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (!movedRef.current) fit();
    });
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);

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
    movedRef.current = true;
    setTf((t) => {
      const k = Math.min(12, Math.max(0.08, t.k * Math.exp(-e.deltaY * 0.0012)));
      const scale = k / t.k;
      return { k, x: mx - (mx - t.x) * scale, y: my - (my - t.y) * scale };
    });
  };

  const zoomBy = (factor: number) => {
    const el = containerRef.current;
    if (!el) return;
    movedRef.current = true;
    const mx = el.clientWidth / 2;
    const my = el.clientHeight / 2;
    setTf((t) => {
      const k = Math.min(12, Math.max(0.08, t.k * factor));
      const scale = k / t.k;
      return { k, x: mx - (mx - t.x) * scale, y: my - (my - t.y) * scale };
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.target !== e.currentTarget) return; // only background pans
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    panRef.current = { x0: e.clientX, y0: e.clientY, tx: tf.x, ty: tf.y };
    setDragging(true); // so the drag guard suppresses text selection
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const p = panRef.current;
    if (!p) return;
    movedRef.current = true;
    setTf((t) => ({ ...t, x: p.tx + e.clientX - p.x0, y: p.ty + e.clientY - p.y0 }));
  };
  const endPan = () => {
    if (!panRef.current) return;
    panRef.current = null;
    setDragging(false);
  };

  const startCardDrag = (e: React.PointerEvent<HTMLButtonElement>, placed: Placed) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    const before = tensorOffsets[placed.id] ?? { dx: 0, dy: 0 };
    cardDragRef.current = {
      id: placed.id,
      pointerId: e.pointerId,
      lastClient: { x: e.clientX, y: e.clientY },
      rect: { x: placed.x, y: placed.y, w: placed.w, h: placed.h },
      offset: before,
      before,
      blockers: layout.solids
        .filter((other) => !(other.kind === "tensor" && other.id === placed.id))
        .map(({ x, y, w, h }) => ({ x, y, w, h })),
      moved: false,
    };
    setMovingTensor(placed.id);
    setDragging(true);
  };

  const moveCard = (e: React.PointerEvent<HTMLButtonElement>) => {
    const drag = cardDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    const delta = {
      x: (e.clientX - drag.lastClient.x) / tf.k,
      y: (e.clientY - drag.lastClient.y) / tf.k,
    };
    const rect = constrainRectMotion(drag.rect, delta, drag.blockers);
    const accepted = { x: rect.x - drag.rect.x, y: rect.y - drag.rect.y };
    drag.lastClient = { x: e.clientX, y: e.clientY };
    drag.rect = rect;
    if (accepted.x === 0 && accepted.y === 0) return;
    drag.moved = true;
    drag.offset = { dx: drag.offset.dx + accepted.x, dy: drag.offset.dy + accepted.y };
    setTensorOffset(drag.id, drag.offset);
  };

  const finishCardDrag = (commit: boolean) => {
    const drag = cardDragRef.current;
    if (!drag) return;
    cardDragRef.current = null;
    if (commit && drag.moved) commitTensorMove(drag.id, drag.before);
    else if (!commit && drag.moved) setTensorOffset(drag.id, drag.before);
    setMovingTensor(null);
    setDragging(false);
  };

  useEffect(() => {
    const cancel = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || !cardDragRef.current) return;
      e.preventDefault();
      finishCardDrag(false);
    };
    window.addEventListener("keydown", cancel);
    return () => window.removeEventListener("keydown", cancel);
  });

  if (!resolved) return <div className="canvas-empty">no graph loaded</div>;

  return (
    <div
      ref={containerRef}
      className="graph-canvas"
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endPan}
      onPointerCancel={endPan}
      onLostPointerCapture={endPan}
    >
      <div
        className="graph-inner"
        style={{ transform: `translate(${tf.x}px, ${tf.y}px) scale(${tf.k})`, width: layout.w, height: layout.h }}
      >
        <svg className="edges" width={layout.w} height={layout.h}>
          {layout.edges.map((e, i) => (
            <path
              key={i}
              d={e.path}
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
              className={`${hot ? "card-slot" : "card-slot dim"}${movingTensor === p.id ? " moving" : ""}`}
              style={{ left: p.x, top: p.y, width: p.w, height: p.h }}
            >
              <button
                className="tensor-grab"
                aria-label={`move tensor ${t.name}`}
                title={`drag to reposition ${t.name}`}
                onPointerDown={(e) => startCardDrag(e, p)}
                onPointerMove={moveCard}
                onPointerUp={() => finishCardDrag(true)}
                onPointerCancel={() => finishCardDrag(false)}
                onLostPointerCapture={() => finishCardDrag(false)}
              />
              <TensorCard tensor={t} renderScale={renderScale} />
            </div>
          );
        })}
      </div>
      <div className="graph-hud">
        <div className="zoom-controls">
          <button onClick={() => zoomBy(1 / 1.25)} title="zoom out">−</button>
          <button onClick={() => zoomBy(1.25)} title="zoom in">+</button>
          <button onClick={fit} title="fit to view (f)">fit</button>
          <span>{Math.round(tf.k * 100)}%</span>
        </div>
        <div className="graph-help" title="canvas interaction shortcuts">
          dotted handle moves tensors · drag the grid to select · esc cancels · ctrl/cmd+z undoes · scroll zooms
        </div>
      </div>
    </div>
  );
}
