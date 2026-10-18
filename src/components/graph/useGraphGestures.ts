import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ResolvedGraph } from "../../core/graph";
import { constrainRectMotion, type Rect } from "../../view/graph/graph-geometry";
import { type BaseGraphLayout, type GraphScene, type GraphNodeKind, nodeKey } from "../../view/graph/graph-scene";
import { canvasSize, cardCanResize, tensorCardSize } from "../../view/tensor/card-size";
import { resizeScale, type ResizeCorner } from "../../view/tensor/resize";
import type { NodeOffset, NodeOffsets } from "../../view/graph/node-layout";
import { useStore } from "../../state/store";
import { useFrameThrottle } from "../hooks/useFrameThrottle";
import { activeCanvasTool, graphIsInteractive, layoutMovable } from "../../view/workspace";
import { matchesShortcut, SHORTCUTS } from "../../view/shortcuts";
import { isPrimaryPress } from "../pointer";
import { borrowsMoveTool } from "../hooks/useMoveModifier";

type NodeDrag = {
  graph: ResolvedGraph;
  host: HTMLElement;
  /** Scene key, so one gesture serves cards and operation nodes alike. */
  key: string;
  pointerId: number;
  lastClient: { x: number; y: number };
  rect: Rect;
  offset: NodeOffset;
  before: NodeOffset;
  blockers: Rect[];
  moved: boolean;
  viewportMovedBefore: boolean;
};

/** A card about to be resized from one corner. It starts from its stored
 * scale, and an undo or cancel returns it and every node it moved aside to
 * where they were. */
type ResizeTarget = {
  placed: Rect; fromScale: number; beforeScale: number; beforeOffset: NodeOffset;
  beforeOffsets: NodeOffsets; layout: BaseGraphLayout;
  canvas: { w: number; h: number }; measure: (scale: number) => Rect;
};

type CardResize = ResizeTarget & {
  graph: ResolvedGraph;
  id: string; pointerId: number; host: HTMLElement; corner: ResizeCorner;
  start: { x: number; y: number }; viewportMovedBefore: boolean;
};

/** How far one arrow press moves a node, in world px, and what Shift
 * multiplies it by. The pair mirrors the tile nudge's one-step / eight-step
 * relationship, so the two sets of arrows behave alike even though they move
 * different things. A single px would be true to the tile nudge's literal step
 * and useless here: a node is furniture, not a coordinate. */
const NUDGE_PX = 8;
const NUDGE_FAST = 8;

const ARROW_DELTAS: Record<string, { x: number; y: number }> = {
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
};

type GestureContext = {
  sceneRef: React.MutableRefObject<GraphScene | null>;
  baseLayoutRef: React.MutableRefObject<BaseGraphLayout | null>;
  tfRef: React.MutableRefObject<{ k: number }>;
  movedRef: React.MutableRefObject<boolean>;
  interactive: boolean;
};

/** Owns node move/resize transactions. Geometry stays in the pure view helpers;
 * pointer updates preview in the store, release commits one history entry, and
 * cancellation restores the snapshot only while its graph is still current. */
export function useGraphGestures({ sceneRef, baseLayoutRef, tfRef, movedRef, interactive }: GestureContext) {
  const resolved = useStore((s) => s.resolved);
  const setDragging = useStore((s) => s.setDragging);
  const resizingTensor = useStore((s) => s.resizingTensor);
  const setNodeOffset = useStore((s) => s.setNodeOffset);
  const commitNodeMove = useStore((s) => s.commitNodeMove);
  const [movingNode, setMovingNode] = useState<string | null>(null);
  const [blockedNode, setBlockedNode] = useState<string | null>(null);
  const nodeDragRef = useRef<NodeDrag | null>(null);
  const resizeRef = useRef<CardResize | null>(null);
  /* Stable across renders. Everything the drag handlers read that changes - the
     scene, the viewport scale, the stored offsets - is reached through a ref or
     the store rather than captured.

     Each node calls this from its capture phase, so a press that moves the node
     stops before anything inside it - a card's canvas, a slider, the substitute
     button - sees it. The return value says whether it did. */
  const startNodeDrag = useCallback((
    e: React.PointerEvent<HTMLElement>,
    kind: GraphNodeKind,
    id: string
  ): boolean => {
    if (nodeDragRef.current || resizeRef.current) return false;
    if (!graphIsInteractive(useStore.getState())) return false;
    if (!isPrimaryPress(e)) return false;
    // Portal events follow the React tree, but the card's menu lives outside
    // this DOM node and must keep its own pointer gestures.
    if (!e.currentTarget.contains(e.target as Node)) return false;
    if ((e.target as HTMLElement | null)?.closest?.("[data-card-resize]")) return false;
    // The tool is read here rather than captured, so switching it mid-session
    // takes effect on the next press. The modifier comes off the press itself,
    // and borrows nothing where the layout cannot move.
    const state = useStore.getState();
    if (!layoutMovable(state.inspectorTab)) return false;
    if (activeCanvasTool(state) !== "move" && !borrowsMoveTool(e)) return false;
    const placed = sceneRef.current?.nodes.find(
      (node) => node.kind === kind && node.id === id
    );
    if (!placed) return false;
    e.preventDefault();
    e.stopPropagation();
    // Cancelling the press also cancels native focus. Under Move, subsequent
    // arrow presses should fine-tune the node that was just dragged.
    if (activeCanvasTool(useStore.getState()) === "move") {
      e.currentTarget.focus({ preventScroll: true });
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    const key = nodeKey(kind, id);
    const before = useStore.getState().nodeOffsets[key] ?? { dx: 0, dy: 0 };
    nodeDragRef.current = {
      graph: state.resolved!,
      host: e.currentTarget,
      key,
      pointerId: e.pointerId,
      lastClient: { x: e.clientX, y: e.clientY },
      rect: { x: placed.x, y: placed.y, w: placed.w, h: placed.h },
      offset: before,
      before,
      blockers: (sceneRef.current?.nodes ?? [])
        .filter((other) => !(other.kind === kind && other.id === id))
        .map(({ x, y, w, h }) => ({ x, y, w, h })),
      moved: false,
      viewportMovedBefore: movedRef.current,
    };
    setMovingNode(key);
    setBlockedNode(null);
    setDragging(true);
    return true;
  }, [setDragging]);

  const cardPointRef = useRef<{ x: number; y: number } | null>(null);
  const applyCardMove = useCallback(() => {
    if (!graphIsInteractive(useStore.getState())) return;
    const drag = nodeDragRef.current;
    const point = cardPointRef.current;
    if (!drag || !point || useStore.getState().resolved !== drag.graph) return;
    const delta = {
      x: (point.x - drag.lastClient.x) / tfRef.current.k,
      y: (point.y - drag.lastClient.y) / tfRef.current.k,
    };
    if (delta.x === 0 && delta.y === 0) return;
    // Each axis is still swept over the whole requested distance, so coalescing
    // moves cannot tunnel a card through a neighbour: a frame's worth of
    // movement is one larger request, not a skipped one.
    const rect = constrainRectMotion(drag.rect, delta, drag.blockers);
    const accepted = { x: rect.x - drag.rect.x, y: rect.y - drag.rect.y };
    drag.lastClient = { x: point.x, y: point.y };
    drag.rect = rect;
    if (accepted.x === 0 && accepted.y === 0) {
      if (Math.abs(delta.x) > 0.1 || Math.abs(delta.y) > 0.1) setBlockedNode(drag.key);
      return;
    }
    setBlockedNode(null);
    drag.moved = true;
    movedRef.current = true;
    drag.offset = { dx: drag.offset.dx + accepted.x, dy: drag.offset.dy + accepted.y };
    setNodeOffset(drag.key, drag.offset);
  }, [setNodeOffset]);
  const requestCardMove = useFrameThrottle(applyCardMove);

  const moveCard = useCallback((e: React.PointerEvent<HTMLElement>) => {
    const drag = nodeDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    cardPointRef.current = { x: e.clientX, y: e.clientY };
    requestCardMove();
  }, [requestCardMove]);

  const finishCardDrag = useCallback((commit: boolean) => {
    const drag = nodeDragRef.current;
    if (!drag) return;
    const sameGraph = useStore.getState().resolved === drag.graph;
    commit = commit && sameGraph && graphIsInteractive(useStore.getState());
    // The last coalesced position is part of the gesture, so it is applied
    // before the gesture is read: a drag released inside the same frame as its
    // final move must commit where the pointer left it.
    if (commit) applyCardMove();
    cardPointRef.current = null;
    nodeDragRef.current = null;
    if (commit && drag.moved) commitNodeMove(drag.key, drag.before);
    else if (!commit && sameGraph && drag.moved) {
      setNodeOffset(drag.key, drag.before);
      movedRef.current = drag.viewportMovedBefore;
    }
    if (drag.host.hasPointerCapture(drag.pointerId)) drag.host.releasePointerCapture(drag.pointerId);
    setMovingNode(null);
    setBlockedNode(null);
    setDragging(false);
  }, [applyCardMove, commitNodeMove, setDragging, setNodeOffset]);

  const resizePointRef = useRef<{ x: number; y: number } | null>(null);
  const applyResize = useCallback(() => {
    const drag = resizeRef.current;
    const point = resizePointRef.current;
    if (!drag || !point || !graphIsInteractive(useStore.getState()) || useStore.getState().resolved !== drag.graph) return;
    const scale = resizeScale(drag.fromScale, drag.canvas,
      { x: (point.x - drag.start.x) / tfRef.current.k, y: (point.y - drag.start.y) / tfRef.current.k }, drag.corner);
    const rect = drag.measure(scale);
    const state = useStore.getState();
    // The card takes the size asked for, and its neighbours make room for it.
    state.setCardGeometry(drag.id, scale, { dx: drag.beforeOffset.dx, dy: drag.beforeOffset.dy + rect.y - drag.placed.y }, false,
      { layout: drag.layout, from: drag.beforeOffsets });
    movedRef.current = true;
  }, []);
  const requestResize = useFrameThrottle(applyResize);
  const finishResize = useCallback((commit: boolean) => {
    const drag = resizeRef.current;
    if (!drag) return;
    commit = commit && graphIsInteractive(useStore.getState()) && useStore.getState().resolved === drag.graph;
    if (commit) applyResize();
    resizeRef.current = null;
    resizePointRef.current = null;
    const state = useStore.getState();
    if (commit) state.commitCardResize(drag.id, drag.beforeScale, drag.beforeOffsets);
    else if (state.resolved === drag.graph) {
      state.restoreCardGeometry(drag.id, drag.beforeScale, drag.beforeOffsets);
      movedRef.current = drag.viewportMovedBefore;
    }
    if (drag.host.hasPointerCapture(drag.pointerId)) drag.host.releasePointerCapture(drag.pointerId);
    setMovingNode(null); setBlockedNode(null); setDragging(false);
  }, [applyResize, setDragging]);
  const resizeTarget = useCallback((id: string, corner: ResizeCorner): ResizeTarget | null => {
    const state = useStore.getState();
    const tensor = state.resolved?.tensors[id];
    const placed = sceneRef.current?.nodes.find((node) => node.kind === "tensor" && node.id === id);
    const layout = baseLayoutRef.current;
    if (!tensor || !placed || !layout || !graphIsInteractive(state) || state.resizingTensor !== id) return null;
    const key = nodeKey("tensor", id), cfg = state.viewCfgs[id];
    if (!cardCanResize(tensor.resolved, cfg, state.graphPx)) return null;
    const fromScale = cfg.scale ?? 1;
    const measure = (scale: number): Rect => {
      const size = tensorCardSize(tensor, state.graphPx, { ...cfg, scale });
      return { ...size, x: placed.x, y: placed.y + (corner === "top" ? placed.h - size.h : 0) };
    };
    return {
      placed, fromScale, beforeScale: cfg.scale ?? 1, beforeOffset: state.nodeOffsets[key] ?? { dx: 0, dy: 0 },
      beforeOffsets: state.nodeOffsets, layout,
      canvas: canvasSize(tensor.resolved, { ...cfg, scale: fromScale }, state.graphPx), measure,
    };
  }, []);

  const startResize = useCallback((e: React.PointerEvent<HTMLElement>, id: string, corner: ResizeCorner) => {
    if (nodeDragRef.current || resizeRef.current) return;
    const target = isPrimaryPress(e) ? resizeTarget(id, corner) : null;
    if (!target) return;
    e.preventDefault(); e.stopPropagation();
    resizeRef.current = { ...target, id, graph: useStore.getState().resolved!, pointerId: e.pointerId,
      host: e.currentTarget, corner, start: { x: e.clientX, y: e.clientY }, viewportMovedBefore: movedRef.current };
    e.currentTarget.setPointerCapture(e.pointerId);
    setMovingNode(nodeKey("tensor", id)); setDragging(true);
  }, [resizeTarget, setDragging]);

  const resizeByKey = useCallback((e: React.KeyboardEvent, id: string, corner: ResizeCorner) => {
    const delta = ARROW_DELTAS[e.key];
    const target = delta ? resizeTarget(id, corner) : null;
    if (!target) return;
    e.preventDefault(); e.stopPropagation();
    const { fromScale, beforeOffset, beforeOffsets, layout, measure, placed } = target;
    const step = e.shiftKey ? NUDGE_PX * NUDGE_FAST : NUDGE_PX;
    const scale = resizeScale(fromScale, target.canvas, { x: delta.x * step, y: delta.y * step }, corner);
    const state = useStore.getState();
    state.setCardGeometry(id, scale, { dx: beforeOffset.dx, dy: beforeOffset.dy + measure(scale).y - placed.y }, !e.repeat,
      { layout, from: beforeOffsets });
    movedRef.current = true;
  }, [resizeTarget]);

  useEffect(() => {
    if (!interactive || resizeRef.current?.id !== resizingTensor) finishResize(false);
    if (!interactive && resizingTensor !== null) useStore.getState().setResizingTensor(null);
  }, [interactive, resizingTensor, finishResize]);
  useEffect(() => {
    if (resizingTensor === null) return;
    const outside = (event: PointerEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest("[role=menu]")) return;
      if (target?.closest("[data-tensor]")?.getAttribute("data-tensor") === resizingTensor) return;
      finishResize(false);
      useStore.getState().setResizingTensor(null);
    };
    document.addEventListener("pointerdown", outside, true);
    return () => document.removeEventListener("pointerdown", outside, true);
  }, [resizingTensor, finishResize]);
  useEffect(() => () => {
    finishResize(false);
    finishCardDrag(false);
  }, [finishResize, finishCardDrag]);

  useEffect(() => {
    if (nodeDragRef.current && nodeDragRef.current.graph !== resolved) finishCardDrag(false);
    if (resizeRef.current && resizeRef.current.graph !== resolved) finishResize(false);
  }, [resolved, finishCardDrag, finishResize]);

  useEffect(() => {
    if (!interactive) {
      finishCardDrag(false);
      useStore.getState().setPreviewBox(null);
    }
  }, [interactive, finishCardDrag]);

  /**
   * Move a node by keyboard, one press at a time.
   *
   * A pointer drag is the same motion sampled continuously: same collision
   * sweep, same accepted-delta accounting, same key space. It differs in what
   * a gesture is - a press is complete on arrival, so each one records its own
   * undo step rather than one per press-to-release.
   */
  const nudgeNode = useCallback((
    e: React.KeyboardEvent<HTMLElement>,
    kind: GraphNodeKind,
    id: string
  ) => {
    if (!graphIsInteractive(useStore.getState())) return;
    const step = ARROW_DELTAS[e.key];
    if (!step || e.ctrlKey || e.metaKey || e.altKey) return;
    if (activeCanvasTool(useStore.getState()) !== "move") return;
    const placed = sceneRef.current?.nodes.find(
      (node) => node.kind === kind && node.id === id
    );
    if (!placed) return;
    e.preventDefault();
    e.stopPropagation();
    if (kind === "op") useStore.getState().setSelectedOp(id);
    const reach = NUDGE_PX * (matchesShortcut(e, SHORTCUTS.moveNodeFast) ? NUDGE_FAST : 1);
    const rect = constrainRectMotion(
      { x: placed.x, y: placed.y, w: placed.w, h: placed.h },
      { x: step.x * reach, y: step.y * reach },
      (sceneRef.current?.nodes ?? [])
        .filter((other) => !(other.kind === kind && other.id === id))
        .map(({ x, y, w, h }) => ({ x, y, w, h }))
    );
    const accepted = { x: rect.x - placed.x, y: rect.y - placed.y };
    const key = nodeKey(kind, id);
    if (accepted.x === 0 && accepted.y === 0) {
      setBlockedNode(key);
      return;
    }
    const before = useStore.getState().nodeOffsets[key] ?? { dx: 0, dy: 0 };
    setBlockedNode(null);
    movedRef.current = true;
    setNodeOffset(key, { dx: before.dx + accepted.x, dy: before.dy + accepted.y });
    commitNodeMove(key, before);
  }, [commitNodeMove, setNodeOffset]);

  /** Blocked is feedback on the last attempt, so it belongs to the element that
   * made it: leaving clears it rather than leaving a stale mark behind. */
  const clearBlocked = useCallback((key: string) => {
    setBlockedNode((current) => (current === key ? null : current));
  }, []);

  /** What follows a press that started a drag, the same for every node. The
   * node holds pointer capture, so these arrive on it wherever the pointer goes. */
  const ownsPointer = useCallback((e: React.PointerEvent<HTMLElement>) =>
    (resizeRef.current ?? nodeDragRef.current)?.pointerId === e.pointerId, []);
  const cancel = useCallback(() => {
    finishResize(false);
    finishCardDrag(false);
  }, [finishResize, finishCardDrag]);
  const commit = useCallback(() => {
    finishResize(true);
    finishCardDrag(true);
  }, [finishResize, finishCardDrag]);
  const dragFollow = useMemo(
    () => ({
      onPointerMove: (e: React.PointerEvent<HTMLElement>) => {
        if (!resizeRef.current) return moveCard(e);
        if (resizeRef.current.pointerId !== e.pointerId) return;
        e.preventDefault(); e.stopPropagation();
        resizePointRef.current = { x: e.clientX, y: e.clientY }; requestResize();
      },
      onPointerUp: (e: React.PointerEvent<HTMLElement>) => {
        if (!ownsPointer(e)) return;
        // Pointerup can carry a newer position than the last animation frame.
        const point = { x: e.clientX, y: e.clientY };
        if (resizeRef.current) resizePointRef.current = point;
        else cardPointRef.current = point;
        commit();
      },
      onPointerCancel: (e: React.PointerEvent<HTMLElement>) => { if (ownsPointer(e)) cancel(); },
      onLostPointerCapture: (e: React.PointerEvent<HTMLElement>) => { if (ownsPointer(e)) cancel(); },
    }),
    [moveCard, commit, cancel, ownsPointer, requestResize]
  );

  useEffect(() => {
    const cancel = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || (!nodeDragRef.current && !resizeRef.current)) return;
      e.preventDefault();
      finishResize(false);
      finishCardDrag(false);
    };
    window.addEventListener("keydown", cancel);
    return () => window.removeEventListener("keydown", cancel);
  }, [finishCardDrag, finishResize]);

  return { movingNode, blockedNode, startNodeDrag, startResize, resizeByKey,
    nudgeNode, clearBlocked, dragFollow,
    isDraggingNode: () => nodeDragRef.current !== null,
    isResizing: () => resizeRef.current !== null };
}
