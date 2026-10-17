import React, { useEffect, useMemo, useRef, useState } from "react";
import { Tensor } from "../../core/graph";
import { elementCount } from "../../core/shapes";
import { DTYPE_BYTES } from "../../core/dtypes";
import { useFrameThrottle } from "../hooks/useFrameThrottle";
import { useDebounced } from "../hooks/useDebounced";
import { Box, formatBoxIndices, fromBox, iv, subtract } from "../../core/region";
import { CardCell, paintScale, elementFromEvent, gridGeometry, HeatPaint, Layer, PlanPaint } from "../../view/tensor/grid";
import { displayTile } from "../../view/tensor/tiling";
import { lensColor, lensReading, lensReadingText } from "../../view/reuse-lens";
import { useLensPaint } from "./useLensPaint";
import { drawGrid } from "./draw-grid";
import { useDark, useStore } from "../../state/store";
import { axisName, shapeLabel, shapeReadings } from "../../view/tensor/shape-label";
import { OVERVIEW_SCALE, overviewTextScale } from "../../view/graph/overview-labels";
import { formatBytes } from "../../view/format";
import {
  axesWith,
  hiddenAxes,
  hiddenMode,
  projectedTileAxes,
  sliceOrdinal,
  viewAxes,
} from "../../view/tensor/tensor-view";
import { cardDisplayMenu, cardViewMenu, hiddenViewCfg, type CardViewAction, type CardViewChoice } from "../../view/tensor/menus";
import { OptionsMenu, useOptionsMenu } from "../chrome/OptionsMenu";
import type { MenuHandlers } from "../../view/menu";
import { scaleRatioLabel, viewTags } from "../../view/tensor/card-size";
import type { ResizeCorner } from "../../view/tensor/resize";
import { DECK_ROOM_PX, DECK_STEP_PX, deckSheets } from "../../view/tensor/deck";
import { cancelDeal, deal, holdPicture, prefersStill } from "./deal";
import { seedTile } from "../../view/tensor/tile-spec";
import { graphIsInteractive, keyboardTile, layoutMovable, nextHue, partHue, partsOn } from "../../view/workspace";
import { activeGrowth } from "../../view/growth";
import { tilesBox } from "../../core/plan/tile-family";
import { buildPlanPaint, buildLayers, composeCardLayers } from "../../view/tensor/layers";
import {
  CellDrag,
  dragAt,
  dragMoved,
  planGesture,
  planElementFromCell,
  samePanel,
  selectionBoxFromDrag,
  visibleApproximation,
} from "../../view/tensor/gesture";
import { buildReplayPaint } from "../../view/tensor/replay-paint";
import { isPrimaryPress } from "../pointer";
import { CardScaleDialog } from "./CardScaleDialog";

export { cardSize } from "../../view/tensor/card-size";

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
  rasterize = true,
  rasterPixels,
  displayScale,
  onResizeStart,
  onResizeKeyDown,
  onMakeRoom,
}: {
  tensor: Tensor;
  renderScale?: number;
  viewScale?: number;
  overviewWidth?: number;
  rasterize?: boolean;
  rasterPixels?: number;
  /** The scale the graph has room to draw a resized card at; see `displayedCardScales`. */
  displayScale?: number;
  onResizeStart?: (event: React.PointerEvent<HTMLElement>, id: string, corner: ResizeCorner) => void;
  onResizeKeyDown?: (event: React.KeyboardEvent, id: string, corner: ResizeCorner) => void;
  onMakeRoom?: (id: string) => void;
}): React.ReactElement {
  const shape = tensor.resolved!;
  const rank = shape.length;
  const interactive = useStore(graphIsInteractive);
  const graph = useStore((s) => s.resolved);
  const [scaleEditor, setScaleEditor] = useState<number | null>(null);
  useEffect(() => { setScaleEditor(null); }, [graph, interactive]);
  const storedCfg = useStore((s) => s.viewCfgs[tensor.id]);
  const cfg = useMemo(
    () => displayScale === undefined || displayScale === storedCfg.scale ? storedCfg : { ...storedCfg, scale: displayScale },
    [storedCfg, displayScale]
  );
  const setViewCfg = useStore((s) => s.setViewCfg);
  const setViewAxes = useStore((s) => s.setViewAxes);
  const resizing = useStore((s) => s.resizingTensor === tensor.id);
  const movable = useStore((s) => layoutMovable(s.inspectorTab));
  /** How the tensor is drawn is its own setting, so it is changed here, on the
   *  card, from a right-click rather than from the inspected tile's table. */
  const viewMenu = useOptionsMenu(interactive);
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
  const keyTile = useStore(keyboardTile);
  // What the keyboard tile's growth added on this tensor. Selected as the
  // stored region itself, so a card the growth does not reach never repaints.
  const grownRegion = useStore((s) => {
    const growth = activeGrowth(s.growth, s.selection?.parts, keyboardTile(s));
    const row = growth?.delta.tensors.find((t) => t.tensorId === tensor.id);
    return row && row.elements > 0 ? row.added : null;
  });
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
  // Only the family's tensor and the ones some replayed task reaches repaint
  // on each frame. Unrelated cards keep selecting null and React.memo can
  // leave their canvases alone. The replay belongs to the Plan view, whose
  // paint it overlays.
  const replayPlayback = useStore((s) => {
    const playback = s.replayPlayback;
    return playback && s.inspectorTab === "plan" && (
      playback.replay.task.tensorId === tensor.id ||
      playback.replay.frames.some((frame) => frame.reach.has(tensor.id))
    ) ? playback : null;
  });
  const plan = useStore((s) => s.plan);
  const planSupply = useStore((s) => s.planSupply);
  /** The tasks of this tensor a lineage row in the Plan view points at. */
  const lineage = useStore((s) =>
    s.inspectorTab === "plan" && s.lineageFocus?.tensorId === tensor.id ? s.lineageFocus : null
  );
  /** The reuse lens, when it is on this card: counts take the place of cones. */
  const lens = useLensPaint(tensor.id);
  const setReuseLens = useStore((s) => s.setReuseLens);
  const planTaskAt = useStore((s) => s.planTaskAt);
  const setPlanTileAt = useStore((s) => s.setPlanTileAt);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  /** The picture from before a slice change, which the deal carries away. */
  const outRef = useRef<HTMLCanvasElement>(null);
  /** The slice last drawn, so a redraw knows whether it is a deal. */
  const dealtRef = useRef<number | null>(null);
  const previewKeyRef = useRef<string | null>(null);
  const [drag, setDrag] = useState<CellDrag | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  /** The element under the pointer: the Plan view outlines the tile a click takes. */
  const [hoverCell, setHoverCell] = useState<CardCell | null>(null);

  const geom = useMemo(
    () => gridGeometry(shape, cfg, tileScale, graphPx),
    [shape, cfg, tileScale, graphPx]
  );
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const mode = rank > 2 ? hiddenMode(cfg) : "project";
  /** A slice card is a deck: the slices still to come lie behind the one shown. */
  const slicing = mode === "slice";
  const ordinal = sliceOrdinal(shape, cfg);
  const sheets = slicing ? deckSheets(ordinal.index, ordinal.count) : 0;

  const parts = useMemo(() => partsOn(selection, tensor.id), [selection, tensor.id]);
  const hues = useMemo(
    () => selection?.parts.map((_, index) => partHue(selection.parts, index)) ?? [],
    [selection]
  );
  const newHue = useMemo(() => nextHue(selection?.parts ?? []), [selection]);
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
    () => seedTile(shape, cfg, displayTile(shape, tileScale)),
    [shape, cfg, tileScale]
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
    const moved = drag && dragMoved(drag);
    // The tile as it will commit. A tensor that already has a tiling cannot be
    // redrawn, so a drag over one previews the task it will inspect, not a band.
    if (drag && moved && !tiled) return fromBox(selectionBoxFromDrag(shape, cfg, geom, drag, true));
    const cell = drag ? { row: drag.r0, col: drag.c0, panel: drag.p0 } : hoverCell;
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
    if (!rasterize) {
      // Setting dimensions releases the previous backing store. Keep the DOM
      // canvas and its CSS geometry for focus, gestures, and graph placement.
      if (canvas.width !== 0) canvas.width = 0;
      if (canvas.height !== 0) canvas.height = 0;
      cancelDeal(canvas, outRef.current);
      dealtRef.current = null;
      return;
    }
    let layers: Layer[];
    let paint: PlanPaint | undefined;
    const dragRegion = drag ? fromBox(dragToBox(drag)) : null;
    // The box a click here would take, outlined; the Plan view has its own.
    const hoverRegion = !planView && !drag && hoverCell && interactive
      ? fromBox(dragToBox(dragAt(hoverCell)))
      : null;
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
        recede: replayPlayback?.opacity ?? 0,
      });
      layers = planned.layers;
      paint = planned.paint;
      // A lineage row being pointed at outlines the tasks it names here, in
      // place of the inspected task's direct producers: solid where a task is
      // reached for certain, dashed where it is only possible. The two sets
      // are kept apart, since the strokes are translucent and would compound.
      const lineageFamily = lineage && plan?.families.get(tensor.id);
      if (lineage && lineageFamily)
        paint = {
          ...paint,
          tiles: [
            ...lineage.definite.boxes.map((b) => ({ box: tilesBox(lineageFamily, b), definite: true })),
            ...subtract(lineage.coordinates, lineage.definite).boxes
              .map((b) => ({ box: tilesBox(lineageFamily, b), definite: false })),
          ],
        };
    } else {
      layers = buildLayers({
        tensorId: tensor.id,
        dark,
        direction,
        isSelected,
        parts,
        partCount: selection?.parts.length ?? 0,
        hues,
        nextHue: newHue,
        perBox,
        hiddenBoxes,
        focusedBox,
        keyTile,
        back,
        fwd,
        prev,
        prevForward,
        dragRegion,
        hoverRegion,
        grown: grownRegion && keyTile !== null ? { index: keyTile, region: grownRegion } : null,
        entangled,
        showEntangled,
      });
    }

    // Under the reuse lens this card shows counts instead of cones and
    // replay fills; its tiles keep their outlines (`composeCardLayers`).
    const heat: HeatPaint | undefined = lens?.complete
      ? { pieces: lens.pieces.map((p) => ({ box: p.box, color: lensColor(p.count, lens.max, dark) })), hatch: !lens.exact }
      : undefined;
    const replayed = replayPlayback && !lens
      ? buildReplayPaint({ tensorId: tensor.id, dark, playback: replayPlayback })
      : null;
    // The run's producer tiles take the place of the inspected task's, in
    // the same neutral outline.
    if (paint && replayed?.tiles.length) paint = { ...paint, tiles: replayed.tiles };
    layers = composeCardLayers({
      layers,
      replay: replayed?.layers ?? [],
      lens: !!lens,
      gesture: planView ? planPointer : dragRegion ?? hoverRegion,
    });
    // A redraw that changes the slice shown is a deal. Hold the old picture
    // before it is painted over, so it can leave while the new one arrives.
    const last = dealtRef.current;
    dealtRef.current = slicing ? ordinal.index : null;
    const way = slicing && last !== null && last !== ordinal.index && !prefersStill()
      ? (Math.sign(ordinal.index - last) as 1 | -1)
      : 0;
    const out = outRef.current;
    const held = way !== 0 && !!out && holdPicture(canvas, out);
    drawGrid(canvas, shape, cfg, geom, layers, dark, paintRenderScale, drawScale, paint, rasterPixels, heat);
    if (held && out) return deal(canvas, out, way as 1 | -1);
  }, [
    slicing,
    ordinal.index,
    back,
    rasterize,
    rasterPixels,
    cfg,
    dark,
    direction,
    drag,
    entangled,
    showEntangled,
    focusedBox,
    keyTile,
    fwd,
    geom,
    hiddenBoxes,
    isSelected,
    replayPlayback,
    lens,
    parts,
    hues,
    newHue,
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
    hoverCell,
    interactive,
    grownRegion,
    lineage,
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
  const requestPreview = useFrameThrottle((...args: Parameters<typeof setPreviewBox>) => {
    if (args[0] === null || graphIsInteractive(useStore.getState())) setPreviewBox(...args);
  });

  useEffect(() => {
    if (!interactive) {
      setDrag(null);
      setHover(null);
      setHoverCell(null);
      previewKeyRef.current = null;
    }
  }, [interactive]);

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
    if (!graphIsInteractive(useStore.getState())) return;
    if (!isPrimaryPress(e)) return;
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    if (!cell) return;
    e.preventDefault();
    if (planView && !tensor.producer) return; // a graph input has no tasks to divide
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    if (previewKeyRef.current !== null) requestPreview(null);
    previewKeyRef.current = null;
    setDrag(dragAt(cell));
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!graphIsInteractive(useStore.getState())) return;
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    // Only a change of cell changes anything drawn. Pointer events outrun the
    // cells they land in by an order of magnitude on a large card, and a fresh
    // object for the same cell repainted the canvas for a picture identical to
    // the one already on it.
    if (drag && cell && (cell.row !== drag.r1 || cell.col !== drag.c1 || !samePanel(cell.panel, drag.p1)))
      setDrag({ ...drag, r1: cell.row, c1: cell.col, p1: cell.panel });
    if (cell) {
      // A hover is the click that has not happened yet. Both the readout and the
      // preview cone are therefore built from the box that click would commit,
      // by the same function the commit uses, so neither can drift from it.
      const box = dragToBox(dragAt(cell));
      const key = formatBoxIndices(box);
      if (lens) {
        // Under the lens the reading is the element's count, so the position
        // is the element's too, not the tile a click would take.
        const element = selectionBoxFromDrag(shape, cfg, geom, dragAt(cell), false);
        setHover(`(${formatBoxIndices(element)}) · ${
          lens.complete ? lensReadingText(lensReading(lens, element)) : "counts too fine to draw"
        }`);
      } else setHover(`(${key})`);
      setHoverCell((current) =>
        current && current.row === cell.row && current.col === cell.col && samePanel(current.panel, cell.panel)
          ? current
          : cell
      );
      if (planView) return; // the Plan view has no cone preview
      if (!drag && previewKeyRef.current !== key) {
        previewKeyRef.current = key;
        requestPreview(tensor.id, box, storedCfg);
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
    if (!graphIsInteractive(useStore.getState())) return;
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
  const nameOf = (ax: number) => axisName(tensor, ax);
  const scaleNote = scaleRatioLabel(shape, cfg, graphPx);
  const planeSpec = cardViewMenu(shape, cfg, nameOf);
  const viewSpec = [...planeSpec, ...(planeSpec.length ? [{ kind: "separator" as const }] : []),
    ...cardDisplayMenu(shape, cfg, graphPx, movable, resizing, storedCfg, !!onMakeRoom)];
  const viewHandlers: MenuHandlers<CardViewAction, CardViewChoice> = {
    action: {
      swap: () => setViewAxes(tensor.id, [colAxis, rowAxis]),
      default: () => setViewAxes(tensor.id, null),
      resize: () => useStore.getState().setResizingTensor(resizing ? null : tensor.id),
      resetSize: () => useStore.getState().setCardScale(tensor.id, 1),
      setSize: () => {
        useStore.getState().setResizingTensor(null);
        setScaleEditor(storedCfg.scale ?? 1);
      },
      makeRoom: () => onMakeRoom?.(tensor.id),
    },
    choice: {
      hidden: (value) => setViewCfg(tensor.id, hiddenViewCfg(value)),
      rows: (axis) => setViewAxes(tensor.id, axesWith(shape, cfg, axis, "rows")),
      cols: (axis) => setViewAxes(tensor.id, axesWith(shape, cfg, axis, "cols")),
    },
  };
  const tags = viewTags(shape, cfg, nameOf);
  const hiddenNames = hiddenAxes(shape, cfg).map(nameOf);
  const tilePositions = projectedTileAxes(shape, cfg);
  /** In projection, the position controls choose the tile a gesture takes. */
  const heldTiles = mode === "project"
    ? tilePositions
        .map((axis) => {
          const lo = Math.floor((cfg.sliders[axis] ?? 0) / cfg.tile![axis]) * cfg.tile![axis];
          return `${nameOf(axis)} ${lo}:${Math.min(shape[axis], lo + cfg.tile![axis])}`;
        })
    : [];
  const hiddenTitle =
    mode === "unfold"
      ? `unfolded: every index of ${hiddenNames.join(", ")} has a panel of its own, the slowest axis outermost, so the panels read in the order the tensor is stored. A drag can span panels. Right-click the card to project or slice instead.`
      : mode === "slice"
        ? `sliced: one index of ${hiddenNames.join(", ")} at a time, chosen by the sliders or [ and ]. The sheets behind are the slices still to come in row-major order. Right-click the card to project or unfold instead.`
        : `projected: each cell stands for every index of ${hiddenNames.join(", ")}, shaded by how much of them a region covers; a drag takes them all${
            heldTiles.length ? `, except where the tensor's tile is narrower - here it takes ${heldTiles.join(", ")}. The tile position controls choose each axis; [ and ] step the first` : ""
          }. Right-click the card to slice or unfold instead.`;
  // Keep the compact header to one reading. The details popover preserves the
  // separate axis-label, symbolic-extent, and numeric-extent facts.
  const symbolicShape = shapeLabel(tensor, "symbolic");
  const numericShape = shapeLabel(tensor, "numeric");
  const shownShape = axisMode === "numeric" ? numericShape : symbolicShape;
  const tileSpanRows = Math.min(geom.rows, geom.rowTile);
  const tileSpanCols = Math.min(geom.cols, geom.colTile);
  const ownTile = cfg.tile;
  /**
   * The tile this card snaps to, always in the header: the setup strip names
   * how many tiles cross an axis, and axes of different lengths settle on
   * different tiles, so only the card can say which one it has. `cardSize`
   * reserves the label's width, so detail changes re-rasterise in place (C4).
   */
  const roleTag = tensor.producer ? null : tensor.role === "weight" ? "weight" : "input";
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

  // While its menu or size dialog is open, the card shows it is the one being
  // set up, in the grey a card takes while pointed at to move.
  const settingUp = (!!viewMenu.anchor || scaleEditor !== null) && interactive;

  return (
    <div
      className={`tensor-card${isSelected ? " selected" : ""}${settingUp ? " setting-up" : ""}${viewScale < OVERVIEW_SCALE ? " overview" : ""}`}
      data-tensor={tensor.id}
      style={{
        "--view-scale": viewScale,
        "--label-scale": overviewTextScale(viewScale),
      } as React.CSSProperties}
      onContextMenu={viewSpec.length ? viewMenu.onContextMenu : undefined}
    >
      <OptionsMenu menu={viewMenu} label={`${tensor.name} view`} spec={viewSpec} handlers={viewHandlers} />
      {scaleEditor !== null && interactive && <CardScaleDialog name={tensor.name} scale={scaleEditor}
        onClose={() => setScaleEditor(null)}
        onApply={(scale) => { useStore.getState().setCardScale(tensor.id, scale); setScaleEditor(null); }} />}
      {/* The tensor plate is deliberately frameless. Its persistent label is the
          name, the resolved numeric shape, and the two facts that change how the
          grid below should be read: where the tensor comes from, and whether its
          highlight is exact. */}
      <div className="tc-header">
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
          (
            <span
              className={`tc-tile${ownTile ? " own" : ""}`}
              title={ownTile
                ? `this tensor's tile: [${ownTile.join(", ")}]`
                : "the tile a click or snapped drag takes here, rows × columns"}
            >
              ⊞ {tileSpanRows}×{tileSpanCols}
            </span>
          )
        )}
        {/* Lengths on a card off the graph's scale do not compare with other
            cards', so the card says so wherever it is read. */}
        {scaleNote && (
          <span
            className="tc-scale"
            title={`drawn at ${scaleNote} the graph's scale, so its lengths are not comparable with other cards` +
              (storedCfg.scale !== cfg.scale ? `; resized to ${scaleRatioLabel(shape, storedCfg, graphPx) ?? "×1"}, it is drawn smaller until its neighbours leave room` : "")}
          >
            scale {scaleNote}
          </span>
        )}
        {roleTag && <span className="tc-role">{roleTag}</span>}
        {/* The shading is scaled to this lens's own largest count, so the
            card states that count: two tiles' strongest shades can stand for
            8 uses and for 64. */}
        {lens && (
          <span
            className="tc-lens"
            title={`shaded by how many outputs use each element, 0 to ${lens.max}${lens.exact ? "" : " (upper bounds)"}; the strongest shade is this lens's largest count, so compare the numbers across tiles, not the shades`}
          >
            uses {lens.exact ? "" : "≤"}{lens.min === lens.max ? lens.max : `${lens.min}–${lens.max}`}
            <button
              type="button"
              aria-label={`clear the reuse shading on ${tensor.name}`}
              title="clear the reuse shading (Esc)"
              onClick={() => setReuseLens(null)}
            >
              ×
            </button>
          </span>
        )}
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
      {(tags.hidden || tags.plane) && (
        <div className="tc-axes">
          {/* How the card reads the axes it does not draw is always stated,
              so a union, a slice and an unfolded strip cannot be mistaken for
              one another. It is changed from the card's right-click menu. */}
          {tags.hidden && (
            <span className="tc-plane" title={hiddenTitle}>
              {tags.hidden}
            </span>
          )}
          {/* A chosen pair is stated on the card itself, so the canvas cannot be
              read as the default plane or as a transposed tensor. */}
          {tags.plane && (
            <span
              className="tc-plane"
              title={`display only: the card draws ${nameOf(rowAxis)} down and ${nameOf(colAxis)} across; the graph is unchanged. Right-click the card to change it.`}
            >
              {tags.plane}
            </span>
          )}
        </div>
      )}
      {slicing && shape.map((e, ax) => {
        if (ax === rowAxis || ax === colAxis) return null;
        return (
          <div className="tc-slider" key={ax}>
            <span>{nameOf(ax)}</span>
            <input
              type="range"
              aria-label={`${nameOf(ax)} slice index`}
              title="Displayed slice index"
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
      {tilePositions.map((ax) => {
        const tile = cfg.tile![ax];
        const index = Math.floor((cfg.sliders[ax] ?? 0) / tile);
        const lo = index * tile;
        return (
          <div className="tc-slider tc-tile-position" key={ax}>
            <span>{nameOf(ax)} tile</span>
            <input
              type="number"
              aria-label={`${nameOf(ax)} tile position`}
              title="Which tile a gesture takes on this projected axis (zero-based)"
              min={0}
              max={Math.ceil(shape[ax] / tile) - 1}
              step={1}
              value={index}
              onChange={(ev) => {
                const value = ev.target.valueAsNumber;
                if (!Number.isSafeInteger(value)) return;
                const sliders = cfg.sliders.slice();
                sliders[ax] = Math.max(0, Math.min(Math.ceil(shape[ax] / tile) - 1, value)) * tile;
                setViewCfg(tensor.id, { sliders });
              }}
            />
            <span className="tc-slider-val">{lo}:{Math.min(shape[ax], lo + tile)}</span>
          </div>
        );
      })}
      <div
        className={`tc-canvas-wrap${slicing ? " deck" : ""}`}
        style={slicing ? { paddingTop: DECK_ROOM_PX, paddingRight: DECK_ROOM_PX } : undefined}
      >
        {/* The deck: one sheet per slice still to come, nearest in front. They
            hold no elements; their count is the depth behind this slice. */}
        {Array.from({ length: sheets }, (_, i) => sheets - i).map((depth) => (
          <div
            key={depth}
            className="tc-deck-sheet"
            aria-hidden
            style={{
              left: depth * DECK_STEP_PX,
              top: DECK_ROOM_PX - depth * DECK_STEP_PX,
              width: geom.canvasW,
              height: geom.canvasH,
              opacity: 1 - (depth - 1) * 0.14,
            }}
          />
        ))}
        {slicing && (
          <canvas
            ref={outRef}
            width={0}
            height={0}
            className="tc-deck-out"
            aria-hidden
            style={{ top: DECK_ROOM_PX, width: geom.canvasW, height: geom.canvasH }}
          />
        )}
        <canvas
          ref={canvasRef}
          className="tc-front"
          style={{ width: geom.canvasW, height: geom.canvasH, cursor: "crosshair" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={cancelPointerDrag}
          onLostPointerCapture={cancelPointerDrag}
          onPointerLeave={onLeave}
        />
        {resizing && onResizeStart && (["top", "bottom"] as const).map((corner) => (
          <button key={corner} className={`tc-resize-handle ${corner}`} data-card-resize={corner}
            aria-label={`resize ${tensor.name} from ${corner} right corner`}
            title="drag to resize · Escape finishes resizing"
            style={{ left: geom.canvasW, top: (slicing ? DECK_ROOM_PX : 0) + (corner === "bottom" ? geom.canvasH : 0) }}
            onPointerDown={(event) => onResizeStart(event, tensor.id, corner)}
            onKeyDown={(event) => onResizeKeyDown?.(event, tensor.id, corner)} />
        ))}
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
