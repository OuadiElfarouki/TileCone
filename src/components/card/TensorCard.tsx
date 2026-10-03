import React, { useEffect, useMemo, useRef, useState } from "react";
import { Tensor } from "../../core/graph";
import { elementCount } from "../../core/shapes";
import { DTYPE_BYTES } from "../../core/dtypes";
import { reuseReachAt } from "../../core/reuse";
import { useFrameThrottle } from "../hooks/useFrameThrottle";
import { useDebounced } from "../hooks/useDebounced";
import { Box, formatBoxIndices, fromBox, iv } from "../../core/region";
import { paintScale, elementFromEvent, gridGeometry, Layer, PlanPaint, tileOf } from "../../view/tensor/grid";
import { drawGrid } from "./draw-grid";
import { useDark, useStore } from "../../state/store";
import { axisName, shapeLabel, shapeReadings } from "../../view/tensor/shape-label";
import { OVERVIEW_SCALE } from "../../view/graph/overview-labels";
import { formatBytes } from "../../view/format";
import { axesWith, hiddenAxisPositioned, remapped, viewAxes } from "../../view/tensor/tensor-view";
import { cardViewMenu, type CardViewAction, type CardViewChoice } from "../../view/tensor/menus";
import { OptionsMenu, useOptionsMenu } from "../chrome/OptionsMenu";
import type { MenuHandlers } from "../../view/menu";
import { cardScaleFor } from "../../view/tensor/card-size";
import { seedTile } from "../../view/tensor/tile-spec";
import { partsOn } from "../../view/workspace";
import { buildPlanPaint, buildLayers } from "../../view/tensor/layers";
import { CellDrag, planGesture, selectionBoxFromDrag, planElementFromCell, visibleApproximation } from "../../view/tensor/gesture";
import { buildExecutionPaint } from "../../view/tensor/execution-paint";
import { isPrimaryPress } from "../pointer";

export { cardSize } from "../../view/tensor/card-size";

/** `4`, `2.5`: a scale ratio as short as it can be stated without lying. */
const formatRatio = (ratio: number): string =>
  Number.isInteger(Math.round(ratio * 10) / 10) ? String(Math.round(ratio)) : ratio.toFixed(1);

/**
 * The card-moving gestures, as callbacks that do not change between renders.
 *
 * The tensor is an argument rather than a closure, so one object serves every
 * card. Binding it per card produced a fresh handler object on each of the
 * graph's renders, which is every pointer event of a pan, and a changing prop
  onPointerMove: (e: React.PointerEvent<HTMLElement>) => void;
  onPointerUp: () => void;
  onPointerCancel: () => void;
  onLostPointerCapture: () => void;
};

/**
 * The card-moving gestures, as callbacks that do not change between renders.
 *
 * The tensor is an argument rather than a closure, so one object serves every
 * card. Binding it per card produced a fresh handler object on each of the
 * graph's renders, which is every pointer event of a pan, and a changing prop
 * is what stops a card from being skipped by `React.memo`.
 */
export type CardGestures = {
  onPointerDown: (e: React.PointerEvent<HTMLElement>, tensorId: string) => void;
  onPointerMove: (e: React.PointerEvent<HTMLElement>) => void;
  onPointerUp: () => void;
  onPointerCancel: () => void;
  onLostPointerCapture: () => void;
};

/**
 * How long the view must hold still before a card re-rasterises, in ms.
 *
 * Long enough that a continuous wheel gesture rasterises once at its end
 * rather than at each of the 32 paint buckets in an octave, short enough that
 * the sharper raster arrives while the reader is still looking at what they
 * zoomed to. Between the two the CSS transform scales the existing raster, so
 * the picture is never absent, only briefly softer.
 */
export const ZOOM_SETTLE_MS = 140;

function TensorCardView({
  tensor,
  renderScale = 1,
  viewScale = 1,
  overviewWidth,
  uniformTile = null,
  gestures,
}: {
  tensor: Tensor;
  renderScale?: number;
  viewScale?: number;
  overviewWidth?: number;
  uniformTile?: number | null;
  gestures?: CardGestures;
}): React.ReactElement {
  const shape = tensor.resolved!;
  const rank = shape.length;
  const cfg = useStore((s) => s.viewCfgs[tensor.id]);
  const setViewCfg = useStore((s) => s.setViewCfg);
  const setViewAxes = useStore((s) => s.setViewAxes);
  /** How the tensor is drawn is its own setting, so it is changed here, on the
   *  card, from a right-click rather than from the inspected tile's table. */
  const viewMenu = useOptionsMenu();
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  // Subscribe only to this tensor's preview entry. A hover query can touch a
  // subset of the graph; cards outside it retain `undefined` and do not render.
  const prev = useStore((s) => s.preview?.backward?.tensors.get(tensor.id));
  const prevForward = useStore((s) => s.preview?.forward?.tensors.get(tensor.id));
  /**
   * The scale this card is rasterised for, which trails the scale it is shown
   * at while the view is moving.
   *
   * Zoom changes both of the paint inputs - the backing-store multiplier and
   * the fine paint bucket - and a wheel gesture crosses 32 buckets per octave,
   * so following it live re-rasterised every card in the graph dozens of times
   * for one gesture. The CSS transform scales what is already drawn in the
   * meantime, so nothing disappears; the raster catches up once the view
   * stops. The first value is taken as it arrives, so a card is never blank
   * and a headless caller sees exactly what it always did.
   */
  const [paintAt, setPaintAt] = useState({ view: viewScale, render: renderScale });
  const settlePaint = useDebounced(setPaintAt, ZOOM_SETTLE_MS);
  useEffect(() => {
    if (paintAt.view === viewScale && paintAt.render === renderScale) return;
    settlePaint({ view: viewScale, render: renderScale });
  }, [viewScale, renderScale, paintAt, settlePaint]);
  const drawScale = paintScale(paintAt.view);
  const paintRenderScale = paintAt.render;
  const setSelection = useStore((s) => s.setSelection);
  const setPreviewBox = useStore((s) => s.setPreviewBox);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const direction = useStore((s) => s.direction);
  const snapToGrid = useStore((s) => s.snapToGrid);
  const axisMode = useStore((s) => s.axisMode);
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);
  const setDragging = useStore((s) => s.setDragging);
  const showEntangled = useStore((s) => s.showEntangled);
  const entangledAll = useStore((s) => s.entangled);
  const inspectorTab = useStore((s) => s.inspectorTab);
  const planView = inspectorTab === "plan";
  // Only the studied tensor and the ones some probe reaches repaint on each
  // playback frame. Unrelated cards keep selecting null and React.memo can
  // leave their canvases alone.
  const executionPlayback = useStore((s) => {
    const playback = s.executionPlayback;
    return playback && (
      playback.tensorId === tensor.id ||
      playback.frames.some((frame) =>
        Object.values(frame.surfaces).some((byTensor) => !!reuseReachAt(byTensor, tensor.id)))
    ) ? playback : null;
  });
  const plan = useStore((s) => s.plan);
  const planSupply = useStore((s) => s.planSupply);
  const planTaskAt = useStore((s) => s.planTaskAt);
  const setPlanTileAt = useStore((s) => s.setPlanTileAt);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewKeyRef = useRef<string | null>(null);
  const [drag, setDrag] = useState<{ r0: number; c0: number; r1: number; c1: number } | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  /** The element under the pointer: the Plan view outlines the tile a click takes. */
  const [hoverCell, setHoverCell] = useState<{ row: number; col: number } | null>(null);

  const geom = useMemo(
    () => gridGeometry(shape, cfg, tileScale, graphPx),
    [shape, cfg, tileScale, graphPx]
  );
  const { rowAxis, colAxis } = viewAxes(shape, cfg);

  const parts = useMemo(() => partsOn(selection, tensor.id), [selection, tensor.id]);
  /* One entry per (part, meeting operation) that lands on this card. A part can
     be entangled with this tensor at more than one node, and those are separate
     facts, so they are separate layers rather than a union. */
  const entangled = useMemo(
    () =>
      entangledAll?.flatMap((forPart, index) =>
        forPart
          .filter((e) => e.tensorId === tensor.id)
          .map((e) => ({ index, region: e.region }))
      ) ?? [],
    [entangledAll, tensor.id]
  );
  const isSelected = parts.length > 0;
  const back = backwardRes?.tensors.get(tensor.id);
  const fwd = forwardRes?.tensors.get(tensor.id);
  const dark = useDark();

  /** The extents a click would divide this tensor at while it is untiled. */
  const planProposal = useMemo(
    () => seedTile(shape, cfg, tileOf(shape, tileScale, graphPx, cfg)),
    [shape, cfg, tileScale, graphPx]
  );

  const commitPlan = (d: CellDrag) => {
    if (!tensor.producer) return;
    const gesture = planGesture(shape, cfg, geom, d, !!plan?.families.has(tensor.id));
    if (gesture.kind === "inspect") planTaskAt(tensor.id, gesture.element);
    else setPlanTileAt(tensor.id, gesture.tile, gesture.element);
  };

  /** The tile the pointer is over, or the band being dragged, in the Plan view. */
  const planPointer = useMemo(() => {
    if (!planView || !tensor.producer) return null;
    const tiled = !!plan?.families.get(tensor.id);
    const moved = drag && (drag.r0 !== drag.r1 || drag.c0 !== drag.c1);
    // The tile as it will commit. A tensor that already has a tiling cannot be
    // redrawn, so a drag over one previews the task it will inspect, not a band.
    if (drag && moved && !tiled) return fromBox(selectionBoxFromDrag(shape, cfg, geom, drag, true));
    const cell = drag ? { row: drag.r0, col: drag.c0 } : hoverCell;
    if (!cell) return null;
    const extents = plan?.families.get(tensor.id)?.tile ?? planProposal;
    const element = planElementFromCell(shape, cfg, geom, cell);
    return fromBox(
      element.map((index, axis) => {
        const lo = Math.floor(index / extents[axis]) * extents[axis];
        return iv(lo, Math.min(lo + extents[axis], shape[axis]));
      })
    );
  }, [planView, tensor.producer, tensor.id, drag, hoverCell, plan, planProposal, shape, cfg, geom]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let layers: Layer[];
    let paint: PlanPaint | undefined;
    if (planView) {
      const planned = buildPlanPaint({
        tensorId: tensor.id,
        rowAxis,
        colAxis,
        dark,
        plan,
        supply: planSupply,
        proposed: tensor.producer ? planProposal : null,
        pointer: planPointer,
      });
      layers = planned.layers;
      paint = planned.paint;
    } else {
      layers = buildLayers({
        tensorId: tensor.id,
        dark,
        direction,
        isSelected,
        parts,
        partCount: selection?.parts.length ?? 0,
        perBox,
        hiddenBoxes,
        focusedBox,
        back,
        fwd,
        prev,
        prevForward,
        dragRegion: drag ? fromBox(dragToBox(drag)) : null,
        entangled,
        showEntangled,
      });
    }

    if (executionPlayback)
      layers.push(...buildExecutionPaint({
        tensorId: tensor.id,
        dark,
        playback: executionPlayback,
      }).layers);
    drawGrid(canvas, shape, cfg, geom, layers, dark, paintRenderScale, drawScale, paint);
  }, [
    back,
    cfg,
    dark,
    direction,
    drag,
    entangled,
    showEntangled,
    focusedBox,
    fwd,
    geom,
    hiddenBoxes,
    isSelected,
    executionPlayback,
    parts,
    perBox,
    drawScale,
    prev,
    prevForward,
    paintRenderScale,
    selection?.parts.length,
    shape,
    snapToGrid,
    tensor.id,
    planView,
    plan,
    planSupply,
    planPointer,
    rowAxis,
    colAxis,
    planProposal,
    tensor.producer,
  ]);

  useEffect(() => {
    if (!drag) return;
    setDragging(true);
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      setDrag(null); // abandon the rubber-band; the selection is left untouched
      setDragging(false);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      setDragging(false);
    };
  }, [drag, setDragging]);

  /** Drag rectangle in tile-cell coordinates -> element-space box. */
  function dragToBox(d: CellDrag): Box {
    // The drag is tracked in elements; snapping is a presentation choice applied
    // at the end, so turning it off costs no precision that was ever available.
    return selectionBoxFromDrag(shape, cfg, geom, d, snapToGrid);
  }

  /** Direct drawing adds by default; Alt turns the gesture into subtraction. */
  function composeOf(e: React.MouseEvent): "union" | "subtract" | undefined {
    if (e.altKey) return "subtract";
    if (e.shiftKey) return "union";
    return undefined;
  }

  function commit(box: Box, e: React.MouseEvent) {
    setSelection(tensor.id, fromBox(box), composeOf(e));
  }

  /* The preview runs a full dependency query, and pointer events outrun frames
     by an order of magnitude on a high-polling-rate mouse. Only the last
     position in a frame can be seen, so only that one is computed.
     Every preview update goes through this, clears included: a clear that
     bypassed it would be overtaken by a move still pending for the frame. */
  const requestPreview = useFrameThrottle(setPreviewBox);

  useEffect(() => {
    setHover(null);
    if (previewKeyRef.current !== null) {
      previewKeyRef.current = null;
      requestPreview(null);
    }
  }, [cfg, requestPreview]);

  /**
   * The Plan view takes a click as "inspect the task here". The element is
   * resolved as an unsnapped one-element selection, so the task named is the
   * one under the pointer even when the plan lattice differs from the display
   * lattice. An untiled produced tensor is tiled first, at the displayed tile
   * on the visible axes and one element on the others: one task per hidden-axis
   * index, as a kernel grid usually assigns batch and head.
   */
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isPrimaryPress(e)) return;
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    if (!cell) return;
    e.preventDefault();
    if (planView && !tensor.producer) return; // a graph input has no tasks to divide
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    if (previewKeyRef.current !== null) requestPreview(null);
    previewKeyRef.current = null;
    setDrag({ r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col });
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    // Only a change of cell changes anything drawn. Pointer events outrun the
    // cells they land in by an order of magnitude on a large card, and a fresh
    // object for the same cell repainted the canvas for a picture identical to
    // the one already on it.
    if (drag && cell && (cell.row !== drag.r1 || cell.col !== drag.c1))
      setDrag({ ...drag, r1: cell.row, c1: cell.col });
    if (cell) {
      // A hover is the click that has not happened yet. Both the readout and the
      // preview cone are therefore built from the box that click would commit,
      // by the same function the commit uses, so neither can drift from it.
      const box = dragToBox({ r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col });
      const key = formatBoxIndices(box);
      setHover(`(${key})`);
      setHoverCell((current) =>
        current && current.row === cell.row && current.col === cell.col ? current : cell
      );
      if (planView) return; // the Plan view has no cone preview
      if (!drag && previewKeyRef.current !== key) {
        previewKeyRef.current = key;
        requestPreview(tensor.id, box, cfg);
      }
    } else {
      setHover(null);
      if (previewKeyRef.current !== null) requestPreview(null);
      previewKeyRef.current = null;
    }
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag) return;
    setDrag(null);
    if (planView) return commitPlan(drag);
    commit(dragToBox(drag), e);
  };

  const cancelPointerDrag = () => {
    if (!drag) return;
    setDrag(null);
    setDragging(false);
  };

  const onLeave = () => {
    setHover(null);
    setHoverCell(null);
    if (previewKeyRef.current !== null) requestPreview(null);
    previewKeyRef.current = null;
  };

  const totalBytes = elementCount(shape) * DTYPE_BYTES[tensor.dtype];
  const planeRemapped = remapped(shape, cfg);
  /** How much smaller than the graph's scale this card draws; 1 for every card
   *  on the default plane. */
  const scaleRatio = graphPx / cardScaleFor(shape, cfg, graphPx);
  const nameOf = (ax: number) => axisName(tensor, ax);
  const viewSpec = cardViewMenu(shape, cfg, nameOf);
  const viewHandlers: MenuHandlers<CardViewAction, CardViewChoice> = {
    action: {
      swap: () => setViewAxes(tensor.id, [colAxis, rowAxis]),
      default: () => setViewAxes(tensor.id, null),
    },
    choice: {
      rows: (axis) => setViewAxes(tensor.id, axesWith(shape, cfg, axis, "rows")),
      cols: (axis) => setViewAxes(tensor.id, axesWith(shape, cfg, axis, "cols")),
    },
  };
  // Keep the compact header to one reading. The details popover preserves the
  // separate axis-label, symbolic-extent, and numeric-extent facts.
  const symbolicShape = shapeLabel(tensor, "symbolic");
  const numericShape = shapeLabel(tensor, "numeric");
  const shownShape = axisMode === "numeric" ? numericShape : symbolicShape;
  const tileSpanRows = Math.min(geom.rows, geom.rowTile);
  const tileSpanCols = Math.min(geom.cols, geom.colTile);
  const ownTile = cfg.tile;
  /**
   * The span is only worth a slot in the header when it says something the
   * setup strip does not. Two cases do: a tensor short enough to clip the
   * square tile (C5), and a graph whose tensors did not all settle on the same
   * one, where the strip can only print a range. On a uniform square lattice
   * the strip already names the tile, and repeating it on every card put a
   * second bracketed number pair beside the shape on the whole canvas.
   *
   * `cardSize` still reserves this label's width whether or not it is drawn, so
   * detail changes re-rasterise in place (C4).
   */
  const showTileSpan = !!ownTile || tileSpanRows !== tileSpanCols || uniformTile !== tileSpanRows;
  const roleTag = tensor.producer ? null : tensor.role === "weight" ? "weight" : "input";
  /** This card's own header handlers, bound once to its tensor. */
  const moveHandlers = useMemo(
    () =>
      gestures
        ? {
            onPointerDown: (e: React.PointerEvent<HTMLElement>) =>
              gestures.onPointerDown(e, tensor.id),
            onPointerMove: gestures.onPointerMove,
            onPointerUp: gestures.onPointerUp,
            onPointerCancel: gestures.onPointerCancel,
            onLostPointerCapture: gestures.onLostPointerCapture,
          }
        : undefined,
    [gestures, tensor.id]
  );
  // Exactness is carried by hatching on the canvas; this repeats it in the
  // header because an over-approximation must never be mistakable for ground
  // truth, and hatching is easy to miss on a small or sparsely covered card.
  const planDemand = planView
    ? [...(planSupply?.demand ?? []), ...(planSupply?.computes ?? [])].filter((d) => d.tensorId === tensor.id)
    : [];
  const planFamily = planView ? plan?.families.get(tensor.id) : undefined;
  const approximation = planView
    ? visibleApproximation(...planDemand.map((d) => d.region))
    : visibleApproximation(
    back?.region,
    fwd?.region,
    ...(showEntangled
      ? entangled
          .filter(({ index }) => !hiddenBoxes.has(index))
          .map(({ region }) => region)
      : [])
  );

  return (
    <div
      className={`tensor-card${isSelected ? " selected" : ""}${viewScale < OVERVIEW_SCALE ? " overview" : ""}`}
      data-tensor={tensor.id}
      style={{ "--view-scale": viewScale } as React.CSSProperties}
      onContextMenu={viewSpec.length ? viewMenu.onContextMenu : undefined}
    >
      <OptionsMenu menu={viewMenu} label={`${tensor.name} view`} spec={viewSpec} handlers={viewHandlers} />
      {/* The tensor plate is deliberately frameless. Its persistent label is the
          name, the resolved numeric shape, and the two facts that change how the
          grid below should be read: where the tensor comes from, and whether its
          highlight is exact. */}
      <div className={`tc-header${moveHandlers ? " movable" : ""}`} {...moveHandlers}>
        <span className={`tc-name-wrap${overviewWidth ? " overview-name" : ""}`} style={overviewWidth ? { width: overviewWidth } : undefined}>
          <span className="tc-name" tabIndex={0} title={tensor.name}>{tensor.name}</span>
          <span className="tc-info" role="tooltip">
            {shapeReadings(tensor).map((reading) => (
              <React.Fragment key={reading.label}>
                <span>{reading.label}</span><b>{reading.value}</b>
              </React.Fragment>
            ))}
            <span>dtype</span><b>{tensor.dtype}</b>
            <span>size</span><b>{formatBytes(totalBytes)}</b>
          </span>
        </span>
        <span className="tc-shape">{shownShape}</span>
        {planView ? (
          planFamily && (
            <span className="tc-tile plan" title={`plan tile · ${planFamily.count} task${planFamily.count === 1 ? "" : "s"}`}>
              ⊞ {Math.min(geom.rows, rowAxis >= 0 ? planFamily.tile[rowAxis] : 1)}×
              {Math.min(geom.cols, colAxis >= 0 ? planFamily.tile[colAxis] : 1)}
            </span>
          )
        ) : (
          showTileSpan && (
            <span
              className={`tc-tile${ownTile ? " own" : ""}`}
              title={ownTile
                ? `this tensor's tile: [${ownTile.join(", ")}]`
                : "current visible-plane tile size"}
            >
              ⊞ {tileSpanRows}×{tileSpanCols}
            </span>
          )
        )}
        {roleTag && <span className="tc-role">{roleTag}</span>}
        {approximation.approximate && (
          <span
            className="tc-approx"
            title={`over-approximation: ${approximation.reasons.join(", ") || "conservative bound"}`}
            aria-label="highlight is a conservative over-approximation"
          >
            ≈
          </span>
        )}
      </div>
      {(rank > 2 || planeRemapped) && (
        <div className="tc-axes">
          {rank > 2 && (
            <button
              className={`mini ${cfg.projection ? "on" : ""}`}
              title="this tensor only - projection unions hidden axes; slice uses the slider index"
              onClick={() => setViewCfg(tensor.id, { projection: !cfg.projection })}
            >
              {cfg.projection ? "proj" : "slice"}
            </button>
          )}
          {/* A chosen pair is stated on the card itself, so the canvas cannot be
              read as the default plane or as a transposed tensor. */}
          {planeRemapped && (
            <span
              className="tc-plane"
              title={`display only: the card draws ${nameOf(rowAxis)} down and ${nameOf(colAxis)} across; the graph is unchanged${
                scaleRatio > 1 ? `. Drawn at 1/${formatRatio(scaleRatio)} of the graph's scale to fit, so its lengths are not comparable with other cards` : ""}. Right-click the card to change it.`}
            >
              rows {nameOf(rowAxis)} · cols {nameOf(colAxis)}
              {scaleRatio > 1 && <b> · scale ÷{formatRatio(scaleRatio)}</b>}
            </span>
          )}
        </div>
      )}
      {shape.map((e, ax) => {
        if (ax === rowAxis || ax === colAxis) return null;
        const positioned = hiddenAxisPositioned(shape, cfg, ax);
        return (
          <div className="tc-slider" key={ax}>
            <span>{nameOf(ax)}</span>
            <input
              type="range"
              disabled={!positioned}
              aria-label={`${nameOf(ax)} ${cfg.projection ? "tile position" : "slice index"}`}
              title={!positioned
                ? "Switch to slice mode to choose an index"
                : cfg.projection
                  ? "Which tile a gesture takes on this axis"
                  : "Displayed slice index"}
              min={0}
              max={e - 1}
              step={1}
              value={cfg.sliders[ax] ?? 0}
              onChange={(ev) => {
                const sliders = cfg.sliders.slice();
                sliders[ax] = Number(ev.target.value);
                setViewCfg(tensor.id, { sliders });
              }}
            />
            <span className="tc-slider-val">{cfg.sliders[ax] ?? 0}</span>
          </div>
        );
      })}
      <div className="tc-canvas-wrap">
        <canvas
          ref={canvasRef}
          style={{ width: geom.canvasW, height: geom.canvasH, cursor: "crosshair" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={cancelPointerDrag}
          onLostPointerCapture={cancelPointerDrag}
          onPointerLeave={onLeave}
        />
        {hover && <div className="tc-tooltip">{hover}</div>}
      </div>
    </div>
  );
}

/**
 * Cards re-render only when their own props or store slices change.
 *
 * The graph re-renders on every pointer event of a pan and on every frame of a
 * card drag, because the viewport transform and the moved card's position live
 * there. Neither changes anything about the other cards, but every card was
 * reconciled anyway - twenty subtrees, each with its own subscriptions, for a
 * translate. The props below are primitives, or objects held stable by the
 * graph for exactly this reason, so the comparison is sound: what a card draws
 * comes from the store, which `useStore` re-subscribes to on its own.
 */
export const TensorCard = React.memo(TensorCardView);
TensorCard.displayName = "TensorCard";
