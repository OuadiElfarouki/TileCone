import { useGraphGestures } from "./useGraphGestures";
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { isExpandable } from "../../core/expand";
import { Rect, rectsOverlap } from "../../view/graph/graph-geometry";
import {
  buildGraphScene,
  GraphScene,
  nodeKey,
  PlacedGraphNode,
} from "../../view/graph/graph-scene";
import { TensorCard } from "../card/TensorCard";
import { carriedOffsets } from "../../view/graph/carry-placement";
import { cardShapeView, type CardShapeView } from "../../view/tensor/card-size";
import { displayedCards } from "../../view/tensor/resize";
import { opLabel } from "../../core/ops/index";
import { useStore } from "../../state/store";
import { MIN_SIDE_PX } from "../../view/tensor/tiling";
import {
  OPERAND_FONT_PX,
  OVERVIEW_SCALE,
  operandLabelsLegible,
  overviewLabels,
  overviewTextScale,
} from "../../view/graph/overview-labels";
import { paintScale } from "../../view/tensor/grid";
import { FIT_GRAPH_EVENT } from "../hooks/useKeyboard";
import { GridControls } from "../chrome/GridControls";
import { LayoutControls } from "../chrome/LayoutControls";
import { useFrameThrottle } from "../hooks/useFrameThrottle";
import { activeCanvasTool, graphIsInteractive, involvedTensorIds, layoutMovable, PANEL_RAIL } from "../../view/workspace";
import { isPrimaryPress } from "../pointer";
import { useMoveModifier } from "../hooks/useMoveModifier";
import { useGraphLayout } from "../hooks/useGraphLayout";
import { graphRasterAllocation } from "../../view/tensor/raster";

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
const GRAPH_PAN_BLOCKERS =
  ".card-slot, .op-node, .zoom-controls, .layout-controls, .grid-controls";

/** @internal DOM-light hit-test seam for the graph interaction tests. */
export function canStartGraphPan(target: unknown): boolean {
  if (!target || typeof (target as { closest?: unknown }).closest !== "function") return true;
  return !(target as { closest: (selector: string) => unknown }).closest(GRAPH_PAN_BLOCKERS);
}

/** How far a background press may travel, in screen px, and still be a click
 * rather than a pan: enough to absorb a hand's tremor on press. */
const CLICK_SLOP_PX = 4;

/** @internal Pure seam: whether a background press released at `release` was a
 * click, which deselects, rather than a pan, which does not. */
export function isBackgroundClick(
  press: { x0: number; y0: number },
  release: { clientX: number; clientY: number }
): boolean {
  return Math.hypot(release.clientX - press.x0, release.clientY - press.y0) <= CLICK_SLOP_PX;
}

/** Under the select tool a press anywhere on an operation selects it, except on
 * the substitute button, which owns its click. Under the move tool the node's
 * contents stop taking the pointer, so the whole box moves it. */
const OPERATION_SELECT_BLOCKERS = ".expand-btn";

/** @internal DOM-light hit-test seam for the node gesture tests. */
export function canSelectOperation(target: unknown): boolean {
  if (!target || typeof (target as { closest?: unknown }).closest !== "function") return true;
  return !(target as { closest: (selector: string) => unknown }).closest(OPERATION_SELECT_BLOCKERS);
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
  const interactive = useStore(graphIsInteractive);
  const resolved = useStore((s) => s.resolved);
  const workerBaseLayout = useStore((s) => s.baseLayout);
  const workerGraphId = useStore((s) => s.workerGraphId);
  const graphPx = useStore((s) => s.graphPx);
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
  const selectedOp = useStore((s) => s.selectedOp);
  const setFocusNode = useStore((s) => s.setFocusNode);
  const setDragging = useStore((s) => s.setDragging);
  const nodeOffsets = useStore((s) => s.nodeOffsets);
  const moveTool = useStore((s) => activeCanvasTool(s) === "move");
  const movable = useStore((s) => layoutMovable(s.inspectorTab));
  // Holding Ctrl/Cmd shows the move cursor only where a drag could follow.
  const moveHeld = useMoveModifier() && movable;
  const resetNodeLayout = useStore((s) => s.resetNodeLayout);

  const [tf, setTf] = useState({ x: 20, y: 20, k: 1 });
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [panning, setPanning] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ x0: number; y0: number; tx: number; ty: number } | null>(null);
  const fitAfterResetRef = useRef(false);
  /** True once the user has panned or zoomed away from a fitted view. Resizing
   * the viewport re-fits only while this is false, so collapsing a panel keeps a
   * fitted graph fitted without throwing away a view someone deliberately set. */
  const movedRef = useRef(false);

  // Highlighting follows the direction filter, not the analysis: both cones are
  // always computed, and a hot edge should mean "the cone you asked for".
  const planView = useStore((s) => s.inspectorTab === "plan");
  const planSupply = useStore((s) => s.planSupply);
  // The replay's frames, not its playback: they change once per replay rather
  // than on every step, so the graph does not re-render with each run.
  const replay = useStore((s) => (s.inspectorTab === "plan" ? s.replayPlayback?.replay ?? null : null));
  const contributing = useMemo(
    () =>
      // The Plan view lights the inspected task's tensor, the ones it reads and
      // the untiled ones it computes, so the graph shows what the cards show.
      // A replay adds what its runs reach, which at depth reaches past them.
      planView
        ? new Set([
            ...(planSupply
              ? [
                  planSupply.task.tensorId,
                  ...planSupply.demand.map((d) => d.tensorId),
                  ...planSupply.computes.map((c) => c.tensorId),
                ]
              : []),
            ...(replay ? replay.frames.flatMap((frame) => [...frame.reach.keys()]) : []),
          ])
        : involvedTensorIds(selection, backwardRes, forwardRes, perBox, hiddenBoxes, direction),
    [backwardRes, direction, forwardRes, hiddenBoxes, perBox, selection, planView, planSupply, replay]
  );

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
  /**
   * The cards whose footprint is not the default, as a stable key: a chosen
   * pair of axes, or hidden axes sliced or unfolded. Only these change a
   * footprint, as does adding or removing a projected tile-position control.
   * Position and tile extent edits that keep the same controls do not re-run placement.
   */
  const footprintKey = useStore((s) => {
    if (!s.resolved) return "";
    return Object.entries(s.viewCfgs)
      .map(([id, cfg]) => [id, s.resolved!.tensors[id] && cardShapeView(s.resolved!.tensors[id].resolved!, cfg)] as const)
      .filter(([, view]) => !!view)
      .map(([id, view]) => `${id}:${JSON.stringify(view)}`)
      .sort()
      .join("|");
  });
  const layoutViews = useMemo(() => {
    const views: Record<string, CardShapeView> = {};
    for (const [id, cfg] of Object.entries(useStore.getState().viewCfgs)) {
      const view = resolved?.tensors[id] && cardShapeView(resolved.tensors[id].resolved!, cfg);
      if (view) views[id] = view;
    }
    return views;
  }, [resolved, footprintKey]);
  const { layout: baseLayout, pending: layoutPending, error: layoutError } = useGraphLayout(
    resolved, workerBaseLayout, workerGraphId, graphPx, layoutViews,
  );

  /** Resized cards' stored scales, as a stable key. */
  const cardScaleKey = useStore((s) => JSON.stringify(Object.entries(s.viewCfgs).flatMap(([id, cfg]) =>
    cfg.scale === undefined ? [] : [[id, cfg.scale]])));
  /** Resized cards' scale and size, by node key. */
  const cardDisplay = useMemo(
    () => resolved
      ? displayedCards(useStore.getState().viewCfgs, resolved, graphPx)
      : { scales: {}, sizes: {} },
    [resolved, graphPx, footprintKey, cardScaleKey]
  );
  /** Offsets, resized cards and connector routes are cheap live scene projection, not layout. */
  const scene = useMemo(
    () =>
      baseLayout
        ? buildGraphScene(baseLayout, nodeOffsets, cardDisplay.sizes)
        : null,
    [baseLayout, nodeOffsets, cardDisplay]
  );
  /** Cards whose footprint overlaps another node: a later layout can leave a
   * resized card less room than it had, and making room is then offered. */
  const crowded = useMemo(() => {
    const nodes = scene?.nodes ?? [];
    return new Set(nodes.filter((node) => node.kind === "tensor" &&
      nodes.some((other) => other !== node && rectsOverlap(node, other, 0))).map((node) => node.id));
  }, [scene]);
  const raster = useMemo(
    () => graphRasterAllocation(scene?.nodes ?? [], tf, viewport),
    [scene, tf, viewport],
  );
  /* Bucketed, not raw: the label solver is an all-pairs collision pass over
     every node, and keyed to `tf.k` it re-ran on every wheel event at the zoom
     range where node counts are highest. `paintScale` is the same 1/32-octave
     bucketing the canvas already trusts for its own redraws, and a 2% change in
     scale cannot flip a collision that was close enough to matter. */
  const labelScale = paintScale(tf.k);
  const zoomedOut = labelScale < OVERVIEW_SCALE;
  const overview = useMemo(() => overviewLabels(
    scene?.nodes ?? [], labelScale,
    {
      ...Object.fromEntries(
        Object.values(resolved?.tensors ?? {}).map((tensor) => [tensor.id, tensor.name])
      ),
      ...Object.fromEntries(
        (resolved?.nodes ?? []).map((node) => [node.id, opLabel(node)])
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
  const baseLayoutRef = useRef(baseLayout);
  baseLayoutRef.current = baseLayout;

  /* A run that changed only sizes keeps every node where it was on screen. The
     store flags it; the canvas, which owns placement, measures where each node
     was last drawn against the new generated layout and installs the offsets
     that hold it there, before the browser paints the generated one. The view
     is kept too, so the fit that follows a new graph is skipped once. */
  const placementCarry = useStore((s) => s.placementCarry);
  const carryPlacement = useStore((s) => s.carryPlacement);
  const drawnRef = useRef<Map<string, Rect>>(new Map());
  const keepViewRef = useRef(false);
  useLayoutEffect(() => {
    if (!placementCarry || !baseLayout || layoutPending) return;
    keepViewRef.current = true;
    // A new layout recreates the ResizeObserver, whose initial notification
    // would otherwise re-fit an untouched viewport after the skipped fit.
    movedRef.current = true;
    carryPlacement(carriedOffsets(drawnRef.current, baseLayout.nodes));
  }, [placementCarry, baseLayout, layoutPending, carryPlacement]);
  /* A new layout of the same graph is a card's footprint changing - a swapped
     plane, a slice or unfold, a tile-position control. Placement reruns, and
     offsets measured against the old layout would put the cards somewhere the
     reader never left them, a grown card on top of its neighbour. Carry the
     drawn placement across instead, as a size-only run does: every card stays
     where it was drawn, and one that grew is nudged clear. */
  const layoutSourceRef = useRef<{ resolved: typeof resolved; layout: typeof baseLayout } | null>(null);
  useLayoutEffect(() => {
    if (!baseLayout || layoutPending || !resolved) return;
    const previous = layoutSourceRef.current;
    layoutSourceRef.current = { resolved, layout: baseLayout };
    if (placementCarry || !previous || previous.resolved !== resolved || previous.layout === baseLayout) return;
    carryPlacement(carriedOffsets(drawnRef.current, baseLayout.nodes));
  }, [resolved, baseLayout, layoutPending, placementCarry, carryPlacement]);
  useLayoutEffect(() => {
    // Not while a carry is pending: this commit's scene is the generated layout
    // the carry is about to replace, and the previous one is what it reads.
    if (!scene || placementCarry) return;
    drawnRef.current = new Map(
      scene.nodes.map((node) => [nodeKey(node.kind, node.id), { x: node.x, y: node.y, w: node.w, h: node.h }])
    );
  }, [scene, placementCarry]);
  const tfRef = useRef(tf);
  tfRef.current = tf;
  const { movingNode, blockedNode, startNodeDrag, startResize, resizeByKey,
    nudgeNode, clearBlocked, dragFollow, isDraggingNode, isResizing } = useGraphGestures({
      sceneRef, baseLayoutRef, tfRef, movedRef, interactive,
    });

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
    // A size-only rebuild must carry against the final remapped layout first.
    if (useStore.getState().placementCarry) return;
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
    if (layoutPending) return;
    if (keepViewRef.current) {
      keepViewRef.current = false;
      return;
    }
    fit();
  }, [resolved, fit, layoutPending]);

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
    if (!el) return;
    const measure = () => setViewport((current) =>
      current.width === el.clientWidth && current.height === el.clientHeight
        ? current : { width: el.clientWidth, height: el.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const panelSignature = () => {
      const { panelW, panelCollapsed } = useStore.getState();
      return `${panelCollapsed.left}:${panelW.left}:${panelCollapsed.right}:${panelW.right}`;
    };
    let lastPanels = panelSignature();
    const ro = new ResizeObserver(() => {
      measure();
      const panels = panelSignature();
      const panelDriven = panels !== lastPanels;
      lastPanels = panels;
      if (!panelDriven && !movedRef.current) fit();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fit, resolved, !!scene]);

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

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (isResizing()) return;
    cancelGlide();
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
    if (!isPrimaryPress(e) || !canStartGraphPan(e.target)) return;
    cancelGlide();
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    panRef.current = { x0: e.clientX, y0: e.clientY, tx: tf.x, ty: tf.y };
    setPanning(true);
    setDragging(true); // so the drag guard suppresses text selection
  };
  /* Pan, card drag and the hover preview all arrive at pointer rate and are
     all read at frame rate. The position is kept in a ref and applied once per
     frame: an intermediate transform is overwritten before it reaches a pixel,
     so computing it only spends the frame the next one needs. The applied
     position is absolute rather than incremental, which is what makes a
     coalesced move identical to the moves it replaced. */
  const panPointRef = useRef<{ x: number; y: number } | null>(null);
  const applyPan = useCallback(() => {
    const p = panRef.current;
    const point = panPointRef.current;
    if (!p || !point) return;
    setTf((t) => ({ ...t, x: p.tx + point.x - p.x0, y: p.ty + point.y - p.y0 }));
  }, []);
  const requestPan = useFrameThrottle(applyPan);

  const onPointerMove = (e: React.PointerEvent) => {
    if (!panRef.current) return;
    movedRef.current = true;
    panPointRef.current = { x: e.clientX, y: e.clientY };
    requestPan();
  };
  /* `release` is the pointer-up that ended the gesture, absent when it was
     cancelled. A release that travelled no further than a click was one, and a
     click on empty canvas lets the selected operation go; a pan or a zoom keeps
     it, since moving the view is not choosing anything else. */
  const endPan = (release?: { clientX: number; clientY: number }) => {
    const p = panRef.current;
    if (!p) return;
    // Apply the last coalesced position before the gesture's state is cleared,
    // or a pending frame would find nothing to apply and drop the final move.
    applyPan();
    if (graphIsInteractive(useStore.getState()) && release && isBackgroundClick(p, release)) {
      if (useStore.getState().selectedOp !== null) setSelectedOp(null);
      useStore.getState().releaseEditBox();
    }
    panRef.current = null;
    panPointRef.current = null;
    setPanning(false);
    setDragging(false);
  };

  const makeRoom = useCallback((id: string) => {
    if (!baseLayout || layoutPending) return;
    const before = useStore.getState().nodeOffsets;
    useStore.getState().makeRoomForCard(id, baseLayout);
    if (useStore.getState().nodeOffsets !== before) {
      fitAfterResetRef.current = true;
      movedRef.current = false;
    }
  }, [baseLayout, layoutPending]);

  const resetLayout = () => {
    if (!movable || !Object.keys(nodeOffsets).length) return;
    fitAfterResetRef.current = true;
    movedRef.current = false;
    resetNodeLayout();
  };

  if (!resolved) return <div className="canvas-empty">no graph loaded</div>;
  if (!scene) return <div className="canvas-empty" role="status">{layoutError ? "Unable to arrange graph." : "Arranging graph…"}</div>;

  const operandsShown = operandLabelsLegible(tf.k);
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
            {/* Sized with the connector it labels, so it never outgrows the
                nodes around it, and left out once it is too small to read. */}
            {edge.operandLabel && operandsShown && (
              <text
                x={edge.operandLabel.x}
                y={edge.operandLabel.y}
                className={`edge-operand${hot ? " hot" : hasResult ? " dim" : ""}`}
                style={{ fontSize: OPERAND_FONT_PX }}
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
      className={`graph-canvas${interactive ? "" : " inactive"}${panning ? " panning" : ""}${interactive && (moveTool || moveHeld) ? " move-tool" : ""}`}
      onWheel={onWheel}
      /* On macOS a Ctrl press is also a right-click, so a borrowed move would
         open the card's menu over the node being dragged. */
      onContextMenuCapture={(e) => {
        if (!isDraggingNode()) return;
        e.preventDefault();
        e.stopPropagation();
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endPan}
      onPointerCancel={() => endPan()}
      onLostPointerCapture={() => endPan()}
    >
      <div
        className="graph-inner"
        {...(!interactive ? { inert: "" } : {})}
        aria-disabled={!interactive || undefined}
        style={{ transform: `translate(${tf.x}px, ${tf.y}px) scale(${tf.k})`, width: scene.width, height: scene.height }}
      >
        {renderEdges("behind")}
        {scene.nodes.map((p) => {
          if (p.kind === "op") {
            const node = nodeById.get(p.id)!;
            const hot = !hasResult || hotNodes.has(p.id);
            const label = overview.ops.get(p.id);
            const key = nodeKey("op", p.id);
            return (
              <div
                key={`n:${p.id}`}
                className={`op-node${hot ? "" : " dim"}${label ? " overview" : ""}${
                  selectedOp === p.id ? " selected" : ""
                }${movingNode === key ? " moving" : ""}${blockedNode === key ? " blocked" : ""}`}
                style={{
                  left: p.x, top: p.y, width: p.w, height: p.h,
                  "--view-scale": tf.k,
                  "--label-scale": overviewTextScale(tf.k),
                } as React.CSSProperties}
                /* The registry name stays in the tooltip beside the attributes:
                   the card says what was written, the title says what runs. The
                   move tool adds a line rather than replacing them: the
                   operation is still the subject, dragging is just now available. */
                title={`${node.op}\n${JSON.stringify(node.attrs)}${
                  moveTool
                    ? blockedNode === key
                      ? "\ndrag: blocked by a neighbouring node"
                      : "\ndrag to reposition"
                    : ""
                }`}
                /* Selection is always available; the move tool adds movement. */
                tabIndex={0}
                role="group"
                data-node-move={moveTool ? "" : undefined}
                aria-label={`${opLabel(node)} operation${selectedOp === p.id ? ", selected" : ""}`}
                aria-keyshortcuts={moveTool ? "Enter Space ArrowUp ArrowDown ArrowLeft ArrowRight" : "Enter Space"}
                onKeyDown={(e) => {
                  if (!graphIsInteractive(useStore.getState())) return;
                  if (e.target !== e.currentTarget) return;
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    e.stopPropagation();
                    setSelectedOp(p.id);
                  } else nudgeNode(e, "op", p.id);
                }}
                onBlur={() => clearBlocked(key)}
                onClick={(e) => {
                  if (!graphIsInteractive(useStore.getState())) return;
                  if (!canSelectOperation(e.target)) return;
                  e.stopPropagation();
                  setSelectedOp(p.id);
                }}
                /* Moving an operation selects it, so the list shows which one
                   is being placed. */
                onPointerDownCapture={(e) => {
                  if (startNodeDrag(e, "op", p.id)) setSelectedOp(p.id);
                }}
                onPointerDown={(e) => {
                  if (!graphIsInteractive(useStore.getState())) return;
                  if (!isPrimaryPress(e) || !canSelectOperation(e.target)) return;
                  e.preventDefault();
                  e.stopPropagation();
                  setSelectedOp(p.id);
                }}
                {...dragFollow}
              >
                {/* Below the overview scale a label the solver could not place
                    is left out rather than drawn at its 100% size inside the
                    box: beside the enlarged labels it read as a different,
                    tiny kind of text. The name stays in the tooltip and the
                    aria label. */}
                {(label || !zoomedOut) && (
                  <span style={label ? { width: label.w, top: label.dy } : undefined}>
                    {opLabel(node)}
                  </span>
                )}
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
                      if (!graphIsInteractive(useStore.getState())) return;
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
          const key = nodeKey("tensor", p.id);
          return (
            <div
              key={`t:${p.id}`}
              className={`${hot ? "card-slot" : "card-slot dim"}${movingNode === key ? " moving" : ""}${blockedNode === key ? " blocked" : ""}`}
              data-tensor-id={p.id}
              style={{ left: p.x, top: p.y, width: p.w, height: p.h }}
              /* Under the move tool the card itself is the keyboard target, as
                 an operation is: Tab reaches it and the arrows move it. Under
                 the select tool it takes no focus, so Tab goes to its name. */
              {...(moveTool
                ? {
                    tabIndex: 0,
                    role: "group",
                    "data-node-move": "",
                    "aria-label": `${t.name} tensor`,
                    "aria-keyshortcuts": "ArrowUp ArrowDown ArrowLeft ArrowRight",
                  }
                : {})}
              title={blockedNode === key ? `${t.name} is blocked by a neighbouring node` : undefined}
              onKeyDown={(e) => {
                if (e.target === e.currentTarget) nudgeNode(e, "tensor", p.id);
              }}
              onBlur={() => clearBlocked(key)}
              onPointerDownCapture={(e) => startNodeDrag(e, "tensor", p.id)}
              {...dragFollow}
            >
              <TensorCard
                tensor={t}
                renderScale={renderScale}
                viewScale={tf.k}
                overviewWidth={overview.tensors.get(t.id)}
                rasterize={raster.visible.has(t.id)}
                rasterPixels={raster.pixelsPerCard}
                crowded={crowded.has(t.id)}
                onResizeStart={startResize}
                onResizeKeyDown={resizeByKey}
                onMakeRoom={layoutPending ? undefined : makeRoom}
              />
            </div>
          );
        })}
        {renderEdges("front")}
      </div>
      {!interactive && (
        <p className="graph-inactive-status" role="status">
          Run the source to render the graph · Current is outdated.
        </p>
      )}
      {/* Everything that changes the canvas, on the canvas: the viewport on the
          left, the lattice a tile is cut against on the right, where the strip
          that used to carry it sat. */}
      <div className="graph-hud">
        <div className="hud-left">
        {(layoutPending || layoutError) && (
          <span className="hint" role="status" title={layoutError ?? undefined}>
            {layoutError ? "Unable to arrange graph." : "Arranging graph…"}
          </span>
        )}
        <div className="zoom-controls">
          <button onClick={() => zoomBy(1 / 1.25)} title="zoom out">−</button>
          <button onClick={() => zoomBy(1.25)} title="zoom in">+</button>
          <button onClick={fit} title="fit to view (f)">fit</button>
          {/* At the floor the percentage is a number with no reference — 12%
              of what, and why will it not go lower. `fit` names the scale the
              zoom-out is actually resting against, which D72 made a derived
              and knowable quantity. */}
          <span title={`${Math.round(tf.k * 100)}% of actual size`}>
            {Math.abs(tf.k - lowZoom()) < ZOOM_EPSILON ? "fit" : `${Math.round(tf.k * 100)}%`}
          </span>
        </div>
          {interactive && <LayoutControls onResetLayout={resetLayout} />}
        </div>
        {interactive && <GridControls />}
      </div>
    </div>
  );
}
