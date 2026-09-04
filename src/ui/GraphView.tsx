import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isExpandable } from "../core/expand";
import { constrainRectMotion, Rect } from "./graph-geometry";
import {
  buildBaseGraphLayout,
  buildGraphScene,
  GraphScene,
  PlacedGraphNode,
} from "./graph-scene";
import { cardSize, TensorCard } from "./TensorCard";
import { shapeLabel, symbolicExtentLabel } from "./shape-label";
import { enabledPropResult, selectedTensorIds, TensorOffset, useStore } from "./store";
import { MIN_SIDE_PX } from "./tiling";
import { FIT_GRAPH_EVENT } from "./useKeyboard";

type CardDrag = {
  id: string;
  pointerId: number;
  lastClient: { x: number; y: number };
  rect: Rect;
  offset: TensorOffset;
  before: TensorOffset;
  blockers: Rect[];
  moved: boolean;
  viewportMovedBefore: boolean;
};

type EdgePresentation = {
  className: "edge" | "edge hot" | "edge dim";
  layer: "behind" | "front";
};

/** Dim connectors are context, so cards should occlude them. Connectors in the
 * active answer stay above cards and read as crossings instead of broken links. */
/** @internal Pure stacking seam for graph-view tests. */
export function edgePresentation(hasResult: boolean, hot: boolean): EdgePresentation {
  if (hasResult && !hot) return { className: "edge dim", layer: "behind" };
  return { className: hot ? "edge hot" : "edge", layer: "front" };
}

/** @internal Pure fit seam for viewport regression tests. */
export function fittedTransform(
  scene: Pick<GraphScene, "left" | "top" | "width" | "height">,
  viewport: { width: number; height: number },
  bounds: { min: number; max: number } = { min: 0, max: 1.25 }
): { x: number; y: number; k: number } {
  const natural = Math.min(
    viewport.width / (scene.width + 40),
    viewport.height / (scene.height + 40),
    1.25
  );
  const k = Math.min(bounds.max, Math.max(bounds.min, natural));
  return {
    x: (20 - scene.left) * k,
    y: (20 - scene.top) * k,
    k,
  };
}

/** Useful zoom is bounded by legibility at the low end and the canvas backing
 * scale at the high end. Operators do not vote: tensor cards are what the user
 * needs to inspect, and their smallest side determines the floor. */
export function graphZoomBounds(
  nodes: Pick<PlacedGraphNode, "kind" | "w" | "h">[]
): { min: number; max: number } {
  const tensorSides = nodes
    .filter((node) => node.kind === "tensor")
    .map((node) => Math.min(node.w, node.h));
  const smallest = tensorSides.length ? Math.min(...tensorSides) : MIN_SIDE_PX;
  return { min: Math.min(1, MIN_SIDE_PX / smallest), max: 4 };
}

/** Elements that own their pointer gesture instead of panning the viewport. */
const GRAPH_PAN_BLOCKERS = ".card-slot, .op-node, .zoom-controls";

/** @internal DOM-light hit-test seam for the graph interaction tests. */
export function canStartGraphPan(target: unknown): boolean {
  if (!target || typeof (target as { closest?: unknown }).closest !== "function") return true;
  return !(target as { closest: (selector: string) => unknown }).closest(GRAPH_PAN_BLOCKERS);
}

/** The header is the card's drag surface, so the one thing in it that owns a
 * click has to be carved back out: the name is the focus target for the shape
 * popover, and starting a drag there would swallow the gesture that opens it. */
const CARD_DRAG_BLOCKERS = ".tc-name-wrap";

/** @internal DOM-light hit-test seam for the card gesture tests. */
export function canStartCardDrag(target: unknown): boolean {
  if (!target || typeof (target as { closest?: unknown }).closest !== "function") return true;
  return !(target as { closest: (selector: string) => unknown }).closest(CARD_DRAG_BLOCKERS);
}

export function GraphView({ onShowShortcuts }: { onShowShortcuts: () => void }): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const graphPx = useStore((s) => s.graphPx);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const selection = useStore((s) => s.selection);
  const expandNodeInPlace = useStore((s) => s.expandNodeInPlace);
  const direction = useStore((s) => s.direction);
  const focusTensor = useStore((s) => s.focusTensor);
  const setDragging = useStore((s) => s.setDragging);
  const tensorOffsets = useStore((s) => s.tensorOffsets);
  const setTensorOffset = useStore((s) => s.setTensorOffset);
  const commitTensorMove = useStore((s) => s.commitTensorMove);
  const resetTensorLayout = useStore((s) => s.resetTensorLayout);

  const [tf, setTf] = useState({ x: 20, y: 20, k: 1 });
  const [movingTensor, setMovingTensor] = useState<string | null>(null);
  const [blockedTensor, setBlockedTensor] = useState<string | null>(null);
  const [panning, setPanning] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ x0: number; y0: number; tx: number; ty: number } | null>(null);
  const cardDragRef = useRef<CardDrag | null>(null);
  const fitAfterResetRef = useRef(false);
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
      direction === "backward" || direction === "both"
        ? enabledPropResult(backwardRes, perBox, hiddenBoxes, null, "backward")
        : null,
      direction === "forward" || direction === "both"
        ? enabledPropResult(forwardRes, perBox, hiddenBoxes, null, "forward")
        : null,
    ];
    for (const res of shown)
      if (res) for (const id of res.tensors.keys()) s.add(id);
    return s;
  }, [backwardRes, direction, forwardRes, hiddenBoxes, perBox, selection]);

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
            cardSize(tensor.resolved!, graphPx, tensor.name, [
              shapeLabel(tensor, "symbolic"),
              symbolicExtentLabel(tensor),
            ])
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
  const zoomBounds = useMemo(
    () => graphZoomBounds(baseLayout?.nodes ?? []),
    [baseLayout]
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
    setTf(fittedTransform(
      current,
      { width: el.clientWidth, height: el.clientHeight },
      zoomBounds
    ));
    movedRef.current = false;
  }, [zoomBounds]);

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
    const onFit = () => fit();
    window.addEventListener(FIT_GRAPH_EVENT, onFit);
    return () => window.removeEventListener(FIT_GRAPH_EVENT, onFit);
  }, [fit]);

  useEffect(() => {
    if (!fitAfterResetRef.current) return;
    fitAfterResetRef.current = false;
    fit();
  }, [scene, fit]);

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
      const k = Math.min(zoomBounds.max, Math.max(zoomBounds.min, t.k * Math.exp(-e.deltaY * 0.0012)));
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
      const k = Math.min(zoomBounds.max, Math.max(zoomBounds.min, t.k * factor));
      const scale = k / t.k;
      return { k, x: mx - (mx - t.x) * scale, y: my - (my - t.y) * scale };
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !canStartGraphPan(e.target)) return;
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    panRef.current = { x0: e.clientX, y0: e.clientY, tx: tf.x, ty: tf.y };
    setPanning(true);
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
    setPanning(false);
    setDragging(false);
  };

  const startCardDrag = (e: React.PointerEvent<HTMLElement>, placed: PlacedGraphNode) => {
    if (!canStartCardDrag(e.target)) return;
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
      viewportMovedBefore: movedRef.current,
    };
    setMovingTensor(placed.id);
    setBlockedTensor(null);
    setDragging(true);
  };

  const moveCard = (e: React.PointerEvent<HTMLElement>) => {
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
    if (accepted.x === 0 && accepted.y === 0) {
      if (Math.abs(delta.x) > 0.1 || Math.abs(delta.y) > 0.1) setBlockedTensor(drag.id);
      return;
    }
    setBlockedTensor(null);
    drag.moved = true;
    movedRef.current = true;
    drag.offset = { dx: drag.offset.dx + accepted.x, dy: drag.offset.dy + accepted.y };
    setTensorOffset(drag.id, drag.offset);
  };

  const finishCardDrag = useCallback((commit: boolean) => {
    const drag = cardDragRef.current;
    if (!drag) return;
    cardDragRef.current = null;
    if (commit && drag.moved) commitTensorMove(drag.id, drag.before);
    else if (!commit && drag.moved) {
      setTensorOffset(drag.id, drag.before);
      movedRef.current = drag.viewportMovedBefore;
    }
    setMovingTensor(null);
    setBlockedTensor(null);
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

  const resetLayout = () => {
    if (!Object.keys(tensorOffsets).length) return;
    fitAfterResetRef.current = true;
    movedRef.current = false;
    resetTensorLayout();
  };

  if (!resolved || !scene) return <div className="canvas-empty">no graph loaded</div>;

  const renderEdges = (layer: EdgePresentation["layer"]) => (
    <svg className={`edges ${layer}`} width={scene.width} height={scene.height} aria-hidden>
      {scene.edges.map((edge) => {
        const hot = contributing.has(edge.tensorId) && hotNodes.has(edge.opId);
        const presentation = edgePresentation(hasResult, hot);
        if (presentation.layer !== layer) return null;
        return (
          <g key={edge.key}>
            <path d={edge.path} className={presentation.className} />
            {/* Flow direction. Structural, so it is drawn whether or not a
                query is live. */}
            {edge.mark && <path d={edge.mark} className={`${presentation.className} edge-arrow`} />}
          </g>
        );
      })}
    </svg>
  );

  return (
    <div
      ref={containerRef}
      className={`graph-canvas${panning ? " panning" : ""}`}
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
        {renderEdges("behind")}
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
          const moveHandlers = {
            onPointerDown: (e: React.PointerEvent<HTMLElement>) => startCardDrag(e, p),
            onPointerMove: moveCard,
            onPointerUp: () => finishCardDrag(true),
            onPointerCancel: () => finishCardDrag(false),
            onLostPointerCapture: () => finishCardDrag(false),
          };
          return (
            <div
              key={`t:${p.id}`}
              className={`${hot ? "card-slot" : "card-slot dim"}${movingTensor === p.id ? " moving" : ""}${blockedTensor === p.id ? " blocked" : ""}`}
              style={{ left: p.x, top: p.y, width: p.w, height: p.h }}
            >
              <button
                className="tensor-grab"
                aria-label={`move tensor ${t.name}`}
                title={blockedTensor === p.id ? `${t.name} is blocked by a neighbouring node` : `drag to reposition ${t.name}`}
                {...moveHandlers}
              />
              <TensorCard
                tensor={t}
                renderScale={renderScale}
                viewScale={tf.k}
                moveHandlers={moveHandlers}
              />
            </div>
          );
        })}
        {renderEdges("front")}
      </div>
      <div className="graph-hud">
        <div className="zoom-controls">
          <button onClick={() => zoomBy(1 / 1.25)} title="zoom out">−</button>
          <button onClick={() => zoomBy(1.25)} title="zoom in">+</button>
          <button onClick={fit} title="fit to view (f)">fit</button>
          <button onClick={resetLayout} disabled={!Object.keys(tensorOffsets).length} title="restore generated tensor layout (undoable)">reset</button>
          <button onClick={onShowShortcuts} title="keyboard shortcuts (?)" aria-label="show keyboard shortcuts">?</button>
          <span>{Math.round(tf.k * 100)}%</span>
        </div>
      </div>
    </div>
  );
}
