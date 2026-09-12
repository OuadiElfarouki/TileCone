import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
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
import { enabledPropResult, PANEL_RAIL, planesOf, selectedTensorIds, useStore } from "./store";
import type { TensorOffset } from "./tensor-layout";
import { MIN_SIDE_PX, settledTiles } from "./tiling";
import { overviewLabels } from "./overview-labels";
import { paintScale } from "./grid";
import { FIT_GRAPH_EVENT } from "./useKeyboard";
import { GridControls } from "./GridControls";

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
    Math.max(1, viewport.width - 40) / Math.max(1, scene.width),
    Math.max(1, viewport.height - 40) / Math.max(1, scene.height),
    1.25
  );
  // Fit is an overview: the manual legibility floor must not crop the scene.
  const k = Math.min(bounds.max, natural);
  /* Centre the scaled scene on both axes rather than pinning it to the top
     left. One axis is what `natural` was limited by and keeps the 20px margin
     exactly; the slack on the other used to collect entirely below and to the
     right of the graph. Margins only grow here, never shrink, so the reason the
     20px exists - keeping overview labels inside the viewport - still holds. */
  return {
    x: (viewport.width - scene.width * k) / 2 - scene.left * k,
    y: (viewport.height - scene.height * k) / 2 - scene.top * k,
    k,
  };
}

/**
 * @internal Pure low-zoom clamp seam, for the same reason `fittedTransform` is
 * one. Because `fit` may sit below the legibility floor, the floor for a manual
 * gesture is the lower of the two - and it is derived from the scene, never
 * from the current scale. Keying it to the current `k` let zooming in past the
 * floor raise the floor, so the fitted overview became unreachable by wheel or
 * button.
 */
export function lowZoomBound(
  scene: Pick<GraphScene, "left" | "top" | "width" | "height">,
  viewport: { width: number; height: number },
  bounds: { min: number; max: number }
): number {
  return Math.min(bounds.min, fittedTransform(scene, viewport, bounds).k);
}

/** How long the viewport takes to slide to a focused node, in ms. Long enough
 * to show the direction travelled, short enough not to be waited on. */
export const GLIDE_MS = 320;

/** Scales within this of each other are the same scale for display purposes.
 * The fitted scale is recomputed from the viewport, so an exact equality would
 * flicker on a sub-pixel resize. */
export const ZOOM_EPSILON = 1e-3;

/** @internal Pure "bring this node to the middle" seam. */
export function centredOn(
  node: Pick<PlacedGraphNode, "x" | "y" | "w" | "h">,
  viewport: { width: number; height: number },
  k: number
): { x: number; y: number } {
  return {
    x: viewport.width / 2 - (node.x + node.w / 2) * k,
    y: viewport.height / 2 - (node.y + node.h / 2) * k,
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
const GRAPH_PAN_BLOCKERS = ".card-slot, .op-node, .zoom-controls, .grid-controls";

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

/** Tensors carrying visible combined-with regions. Kept separate from the
 * directional cone set because restoring a card's opacity must not make its
 * unrelated producer or consumer edges look like dependency paths.
 * @internal Pure visibility seam exported for graph-view tests. */
export function visibleEntangledTensorIds(
  entangled: { tensorId: string }[][] | null,
  hiddenBoxes: ReadonlySet<number>,
  enabled: boolean
): Set<string> {
  const visible = new Set<string>();
  if (!enabled || !entangled) return visible;
  entangled.forEach((entries, index) => {
    if (hiddenBoxes.has(index)) return;
    for (const entry of entries) visible.add(entry.tensorId);
  });
  return visible;
}

export function GraphView(): React.ReactElement {
  const resolved = useStore((s) => s.resolved);
  const graphPx = useStore((s) => s.graphPx);
  const tileScale = useStore((s) => s.tileScale);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const perBox = useStore((s) => s.perBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const selection = useStore((s) => s.selection);
  const expandNodeInPlace = useStore((s) => s.expandNodeInPlace);
  const direction = useStore((s) => s.direction);
  const showEntangled = useStore((s) => s.showEntangled);
  const entangled = useStore((s) => s.entangled);
  const focusNode = useStore((s) => s.focusNode);
  const setSelectedOp = useStore((s) => s.setSelectedOp);
  const setFocusNode = useStore((s) => s.setFocusNode);
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
  const visibleEntangled = useMemo(
    () => visibleEntangledTensorIds(entangled, hiddenBoxes, showEntangled),
    [entangled, hiddenBoxes, showEntangled]
  );

  // Canvas backing-store multiplier, bucketed to powers of two so that zooming
  // reallocates only at bucket crossings. Fine paint buckets separately bound redraws.
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

  /**
   * The tile every tensor settled on, or `null` when the fit rule coarsened
   * some of them and the setup strip can only print a range. Computed once
   * here rather than per card: `planesOf` walks every tensor in the graph.
   */
  const uniformTile = useMemo(() => {
    if (!resolved) return null;
    const { min, max } = settledTiles(planesOf(resolved), tileScale, graphPx);
    return min === max ? min : null;
  }, [resolved, tileScale, graphPx]);

  /** Offsets and connector routes are cheap live scene projection, not layout. */
  const scene = useMemo(
    () =>
      baseLayout
        ? buildGraphScene(baseLayout, tensorOffsets)
        : null,
    [baseLayout, tensorOffsets]
  );
  /* Bucketed, not raw: the label solver is an all-pairs collision pass over
     every node, and keyed to `tf.k` it re-ran on every wheel event at the zoom
     range where node counts are highest. `paintScale` is the same 1/32-octave
     bucketing the canvas already trusts for its own redraws, and a 2% change in
     scale cannot flip a collision that was close enough to matter. */
  const labelScale = paintScale(tf.k);
  const overview = useMemo(() => overviewLabels(
    scene?.nodes ?? [], labelScale,
    {
      ...Object.fromEntries(
        Object.values(resolved?.tensors ?? {}).map((tensor) => [tensor.id, tensor.name])
      ),
      ...Object.fromEntries(
        (resolved?.nodes ?? []).map((node) => [node.id, node.label ?? node.op])
      ),
    }
  ), [scene, labelScale, resolved]);
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
  const tfRef = useRef(tf);
  tfRef.current = tf;

  /* A focused node is slid into view rather than jumped to, because a viewport
     that changes without showing the movement leaves the reader to work out
     what moved and which way. Any deliberate viewport gesture cancels it: the
     user's own pan or zoom always wins over an animation still in flight. */
  const glideRef = useRef<number | null>(null);
  const cancelGlide = useCallback(() => {
    if (glideRef.current !== null) cancelAnimationFrame(glideRef.current);
    glideRef.current = null;
  }, []);

  const glideTo = useCallback((to: { x: number; y: number }) => {
    cancelGlide();
    const from = { x: tfRef.current.x, y: tfRef.current.y };
    const still = Math.abs(to.x - from.x) < 0.5 && Math.abs(to.y - from.y) < 0.5;
    // Honour a reduced-motion preference, and skip the machinery when the view
    // is already where it is going.
    if (still || window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches) {
      setTf((t) => ({ ...t, ...to }));
      return;
    }
    const started = performance.now();
    const step = (now: number) => {
      const p = Math.min(1, (now - started) / GLIDE_MS);
      // Ease out: leaves quickly, arrives gently, so the end reads as settling
      // rather than stopping.
      const e = 1 - (1 - p) ** 3;
      setTf((t) => ({ ...t, x: from.x + (to.x - from.x) * e, y: from.y + (to.y - from.y) * e }));
      glideRef.current = p < 1 ? requestAnimationFrame(step) : null;
    };
    glideRef.current = requestAnimationFrame(step);
  }, [cancelGlide]);

  useEffect(() => cancelGlide, [cancelGlide]);
  /* The low zoom clamp. `fittedTransform` deliberately ignores `zoomBounds.min`
     so an overview is never cropped, which means the fitted scale can sit below
     the manual legibility floor. Clamping against the current `k` let the user
     zoom in past the floor and then be unable to get back out, so the floor is
     derived from the scene instead: the fitted scale is recomputed here rather
     than remembered, because a viewport resize changes it while `movedRef` is
     true and a cached value would go stale. */
  const lowZoom = useCallback(() => {
    const el = containerRef.current;
    const current = sceneRef.current;
    if (!el || !current || el.clientWidth <= 0 || el.clientHeight <= 0)
      return zoomBounds.min;
    return lowZoomBound(current, { width: el.clientWidth, height: el.clientHeight }, zoomBounds);
  }, [zoomBounds]);

  const fit = useCallback(() => {
    cancelGlide();
    if (useStore.getState().selectedOp !== null) useStore.getState().setSelectedOp(null);
    const el = containerRef.current;
    if (!el || el.clientWidth <= 0 || el.clientHeight <= 0) return;
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

  /**
   * Hold the world still when the left panel opens, closes, or is dragged.
   *
   * The scene is translated from this container's top-left corner, and the left
   * panel *is* that corner: its width is the container's origin. Opening or
   * closing it therefore slides the whole world by that width, for a gesture
   * that was about the panel. Only the left panel can do this - the right one
   * and the window move the far edge, which a top-left origin does not follow.
   *
   * The correction has to happen in the same commit as the width change, which
   * is why it reads the panel's width from the store rather than measuring the
   * container afterwards. A `ResizeObserver` fires after layout *and paint*, so
   * compensating there drew the displaced scene for a frame and then pulled it
   * back - the shift was still happening, just briefly. Here the width and the
   * transform that cancels it are one render: React flushes a layout effect's
   * state update before the browser paints, so the displaced position is never
   * on screen at all.
   *
   * `movedRef` is untouched. Cancelling a shift is not a gesture, and a view
   * that was fitted is still showing everything afterwards.
   */
  const leftInset = useStore((s) => (s.panelCollapsed.left ? PANEL_RAIL : s.panelW.left));
  const leftInsetRef = useRef(leftInset);
  useLayoutEffect(() => {
    const dx = leftInset - leftInsetRef.current;
    leftInsetRef.current = leftInset;
    if (dx !== 0) setTf((t) => ({ ...t, x: t.x - dx }));
  }, [leftInset]);

  /**
   * Keep an untouched overview an overview when the room available changes.
   *
   * A window resize genuinely changes what "show me all of it" means, so a view
   * nobody has moved re-fits. A panel does not: how much room the reader wants
   * *beside* the graph is not a question answered by moving the graph, and the
   * left panel's own displacement is already cancelled above. The observer is
   * used rather than a window listener because it fires when layout has settled
   * and covers the container reaching a usable size on first paint.
   */
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const panelSignature = () => {
      const { panelW, panelCollapsed } = useStore.getState();
      return `${panelCollapsed.left}:${panelW.left}:${panelCollapsed.right}:${panelW.right}`;
    };
    let lastPanels = panelSignature();
    const ro = new ResizeObserver(() => {
      const panels = panelSignature();
      const panelDriven = panels !== lastPanels;
      lastPanels = panels;
      if (!panelDriven && !movedRef.current) fit();
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
    if (!focusNode) return;
    const target = sceneRef.current?.nodes.find(
      (node) =>
        node.id === focusNode.id &&
        (node.kind === "tensor") === (focusNode.kind === "tensor")
    );
    const el = containerRef.current;
    if (target && el) {
      glideTo(centredOn(
        target,
        { width: el.clientWidth, height: el.clientHeight },
        tfRef.current.k
      ));
      // Centring is a deliberate viewport position like a pan or a zoom. Left
      // unmarked, the ResizeObserver still considered the view fitted and the
      // next panel collapse or window resize re-fitted away from the node the
      // user asked to see.
      movedRef.current = true;
      // Consumed, so asking for the same node again centres it again after a
      // pan has moved it off screen. Cleared only on success: a request that
      // arrives before the scene has the node stays pending.
      setFocusNode(null);
    }
  }, [focusNode, setFocusNode, glideTo]);

  /* A deliberate viewport gesture means the reader is looking somewhere else,
     so the operations list stops claiming they are working at a row. Hooked to
     the gestures themselves rather than to `movedRef`, which the focus glide
     also sets - and that glide is the *consequence* of clicking a row, so it
     must not clear the row it was asked for. */
  const leaveOperation = () => {
    if (useStore.getState().selectedOp !== null) setSelectedOp(null);
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    cancelGlide();
    leaveOperation();
    const el = containerRef.current!;
    const rect = el.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    movedRef.current = true;
    setTf((t) => {
      const k = Math.min(zoomBounds.max, Math.max(lowZoom(), t.k * Math.exp(-e.deltaY * 0.0012)));
      const scale = k / t.k;
      return { k, x: mx - (mx - t.x) * scale, y: my - (my - t.y) * scale };
    });
  };

  const zoomBy = (factor: number) => {
    cancelGlide();
    leaveOperation();
    const el = containerRef.current;
    if (!el) return;
    movedRef.current = true;
    const mx = el.clientWidth / 2;
    const my = el.clientHeight / 2;
    setTf((t) => {
      const k = Math.min(zoomBounds.max, Math.max(lowZoom(), t.k * factor));
      const scale = k / t.k;
      return { k, x: mx - (mx - t.x) * scale, y: my - (my - t.y) * scale };
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !canStartGraphPan(e.target)) return;
    cancelGlide();
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    panRef.current = { x0: e.clientX, y0: e.clientY, tx: tf.x, ty: tf.y };
    leaveOperation();
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
            {edge.operandLabel && (
              <text
                x={edge.operandLabel.x}
                y={edge.operandLabel.y}
                className={`edge-operand${hot ? " hot" : hasResult ? " dim" : ""}`}
                style={{
                  fontSize: 9 / Math.min(1, tf.k),
                  strokeWidth: 3 / Math.min(1, tf.k),
                }}
              >
                {edge.operandLabel.text}
              </text>
            )}
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
            const label = overview.ops.get(p.id);
            return (
              <div
                key={`n:${p.id}`}
                className={`op-node${hot ? "" : " dim"}${label ? " overview" : ""}`}
                style={{
                  left: p.x, top: p.y, width: p.w, height: p.h,
                  "--view-scale": tf.k,
                } as React.CSSProperties}
                title={`${node.op}\n${JSON.stringify(node.attrs)}`}
              >
                <span style={label ? { width: label.w, top: label.dy } : undefined}>
                  {node.label ?? node.op}
                </span>
                {isExpandable(node.op) && (
                  <button
                    className="expand-btn"
                    /* The title states the cost, because the action is larger
                       than its affordance: it replaces the written source with
                       generated DSL. Undo restores it, and saying so is what
                       makes the click safe to try. */
                    title={`substitute ${node.op} with its primitive subgraph · rewrites the source (undoable)`}
                    aria-label={`substitute ${node.op} with its primitive subgraph`}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      expandNodeInPlace(p.id);
                    }}
                  >
                    {/* Not the tile-span glyph: a card header already reads
                        `⊞ 4×128` a few pixels away, and one mark cannot mean
                        both "tile span" and "substitute the definition". */}
                    ⌄
                  </button>
                )}
              </div>
            );
          }
          const t = resolved.tensors[p.id];
          // Entanglement restores the target card so its stipple is legible,
          // but deliberately does not enter `contributing`: doing that would
          // heat every unrelated edge that happens to carry the same tensor.
          const hot = !hasResult || contributing.has(p.id) || visibleEntangled.has(p.id);
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
                uniformTile={uniformTile}
                tensor={t}
                renderScale={renderScale}
                viewScale={tf.k}
                overviewWidth={overview.tensors.get(t.id)}
                moveHandlers={moveHandlers}
              />
            </div>
          );
        })}
        {renderEdges("front")}
      </div>
      {/* Everything that changes the canvas, on the canvas: the viewport on the
          left, the lattice a tile is cut against on the right, where the strip
          that used to carry it sat. */}
      <div className="graph-hud">
        <div className="zoom-controls">
          <button onClick={() => zoomBy(1 / 1.25)} title="zoom out">−</button>
          <button onClick={() => zoomBy(1.25)} title="zoom in">+</button>
          <button onClick={fit} title="fit to view (f)">fit</button>
          <button onClick={resetLayout} disabled={!Object.keys(tensorOffsets).length} title="restore generated tensor layout (undoable)">reset</button>
          {/* At the floor the percentage is a number with no reference — 12%
              of what, and why will it not go lower. `fit` names the scale the
              zoom-out is actually resting against, which D72 made a derived
              and knowable quantity. */}
          <span title={`${Math.round(tf.k * 100)}% of actual size`}>
            {Math.abs(tf.k - lowZoom()) < ZOOM_EPSILON ? "fit" : `${Math.round(tf.k * 100)}%`}
          </span>
        </div>
        <GridControls />
      </div>
    </div>
  );
}
