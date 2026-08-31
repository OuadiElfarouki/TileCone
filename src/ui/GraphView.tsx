import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isExpandable } from "../core/expand";
import { constrainRectMotion, Rect } from "./graph-geometry";
import {
  buildBaseGraphLayout,
  buildGraphScene,
  PlacedGraphNode,
} from "./graph-scene";
import { cardSize, TensorCard } from "./TensorCard";
import { selectedTensorIds, TensorOffset, useStore } from "./store";

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
  const graphPx = useStore((s) => s.graphPx);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const selection = useStore((s) => s.selection);
  const expandNodeInPlace = useStore((s) => s.expandNodeInPlace);
  const direction = useStore((s) => s.direction);
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

  // Highlighting follows the direction filter, not the analysis: both cones are
  // always computed, and a hot edge should mean "the cone you asked for".
  const contributing = useMemo(() => {
    const s = new Set<string>();
    for (const id of selectedTensorIds(selection)) s.add(id);
    const shown = [
      direction === "backward" || direction === "both" ? backwardRes : null,
      direction === "forward" || direction === "both" ? forwardRes : null,
    ];
    for (const res of shown)
      if (res) for (const id of res.tensors.keys()) s.add(id);
    return s;
  }, [backwardRes, direction, forwardRes, selection]);

  const hasResult = contributing.size > 0;

  // Canvas backing-store multiplier, bucketed to powers of two so that zooming
  // re-renders the cards sharply without redrawing on every wheel tick.
  const renderScale = useMemo(
    () => Math.min(4, Math.max(1, 2 ** Math.ceil(Math.log2(Math.max(1, tf.k))))),
    [tf.k]
  );

  /** Dagre placement depends only on graph structure and tensor footprints. */
  const baseLayout = useMemo(
    () =>
      resolved
        ? buildBaseGraphLayout(resolved, (tensor) =>
            cardSize(tensor.resolved!, graphPx, tensor.name)
          )
        : null,
    [resolved, graphPx]
  );

  /** Offsets and connector routes are cheap live scene projection, not layout. */
  const scene = useMemo(
    () =>
      baseLayout
        ? buildGraphScene(baseLayout, tensorOffsets)
        : null,
    [baseLayout, tensorOffsets]
  );

  const nodeById = useMemo(
    () => new Map(resolved?.nodes.map((node) => [node.id, node]) ?? []),
    [resolved]
  );
  const hotNodes = useMemo(() => {
    const hot = new Set<string>();
    if (!hasResult || !resolved) return hot;
    for (const node of resolved.nodes)
      if ([...node.inputs, ...node.outputs].some((tensorId) => contributing.has(tensorId)))
        hot.add(node.id);
    return hot;
  }, [contributing, hasResult, resolved]);

  const sceneRef = useRef(scene);
  sceneRef.current = scene;
  const fit = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const current = sceneRef.current;
    if (!current) return;
    const k = Math.min(
      el.clientWidth / (current.width + 40),
      el.clientHeight / (current.height + 40),
      1.25
    );
    setTf({ x: 20 * k, y: 20 * k, k });
    movedRef.current = false;
  }, []);

  useEffect(() => {
    fit();
  }, [resolved, fit]);

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
  }, [fit]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
      if (e.key === "f") fit(); // the rest of the bindings live in useKeyboard
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fit]);

  useEffect(() => {
    if (!focusTensor) return;
    const p = sceneRef.current?.nodes.find(
      (x) => x.kind === "tensor" && x.id === focusTensor
    );
    const el = containerRef.current;
    if (p && el)
      setTf((t) => ({
        ...t,
        x: el.clientWidth / 2 - (p.x + p.w / 2) * t.k,
        y: el.clientHeight / 2 - (p.y + p.h / 2) * t.k,
      }));
  }, [focusTensor]);

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

  const startCardDrag = (e: React.PointerEvent<HTMLButtonElement>, placed: PlacedGraphNode) => {
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
      blockers: (scene?.nodes ?? [])
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

  const finishCardDrag = useCallback((commit: boolean) => {
    const drag = cardDragRef.current;
    if (!drag) return;
    cardDragRef.current = null;
    if (commit && drag.moved) commitTensorMove(drag.id, drag.before);
    else if (!commit && drag.moved) setTensorOffset(drag.id, drag.before);
    setMovingTensor(null);
    setDragging(false);
  }, [commitTensorMove, setDragging, setTensorOffset]);

  useEffect(() => {
    const cancel = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || !cardDragRef.current) return;
      e.preventDefault();
      finishCardDrag(false);
    };
    window.addEventListener("keydown", cancel);
    return () => window.removeEventListener("keydown", cancel);
  }, [finishCardDrag]);

  if (!resolved || !scene) return <div className="canvas-empty">no graph loaded</div>;

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
        style={{ transform: `translate(${tf.x}px, ${tf.y}px) scale(${tf.k})`, width: scene.width, height: scene.height }}
      >
        <svg className="edges" width={scene.width} height={scene.height}>
          {scene.edges.map((e) => {
            const hot = contributing.has(e.tensorId) && hotNodes.has(e.opId);
            const cls = hasResult ? (hot ? "edge hot" : "edge dim") : "edge";
            return (
              <g key={e.key}>
                <path d={e.path} className={cls} />
                {/* Flow direction. Structural, so it is drawn whether or not a
                    query is live. */}
                {e.mark && <path d={e.mark} className={`${cls} edge-arrow`} />}
              </g>
            );
          })}
        </svg>
        {scene.nodes.map((p) => {
          if (p.kind === "op") {
            const node = nodeById.get(p.id)!;
            const hot = !hasResult || hotNodes.has(p.id);
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
