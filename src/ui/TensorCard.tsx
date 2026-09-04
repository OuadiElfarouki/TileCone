import React, { useEffect, useMemo, useRef, useState } from "react";
import { Tensor } from "../core/graph";
import { DTYPE_BYTES } from "../core/dtypes";
import { Box, formatBoxIndices, fromBox, iv, Region } from "../core/region";
import {
  drawGrid,
  elementFromEvent,
  gridGeometry,
  GridGeom,
  Layer,
  snapSpan,
  stripeAngleDeg,
} from "./grid";
import { aggregateColors, boxColor } from "./palette";
import { BoxProp, Direction, partsOn, useStore, ViewCfg, viewAxes } from "./store";
import { cardPx, planeExtents } from "./tiling";
import { shapeLabel, shapeReadings } from "./shape-label";

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1 << 20) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1 << 30) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${(n / (1 << 30)).toFixed(2)} GB`;
}

type CellDrag = { r0: number; c0: number; r1: number; c1: number };

/** Convert a visible-plane drag to the tensor region it visually promises.
 * Projection represents the union across hidden axes, so a projection gesture
 * must select their full extent; slice mode remains pinned to its sliders. */
/** @internal Pure interaction seam exported for tensor-card tests. */
export function selectionBoxFromDrag(
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom,
  drag: CellDrag,
  snapToGrid: boolean
): Box {
  const [rLo, rHi] = snapToGrid
    ? snapSpan(drag.r0, drag.r1, geom.tile, geom.rows)
    : [Math.min(drag.r0, drag.r1), Math.max(drag.r0, drag.r1) + 1];
  const [cLo, cHi] = snapToGrid
    ? snapSpan(drag.c0, drag.c1, geom.tile, geom.cols)
    : [Math.min(drag.c0, drag.c1), Math.max(drag.c0, drag.c1) + 1];
  return shape.map((extent, ax) => {
    if (ax === geom.rowAxis) return iv(rLo, rHi);
    if (ax === geom.colAxis) return iv(cLo, cHi);
    if (cfg.projection) return iv(0, extent);
    const slider = cfg.sliders[ax] ?? 0;
    return iv(slider, slider + 1);
  });
}

/** Combine exactness across every cone currently visible on a tensor card. */
/** @internal Pure rendering seam exported for deterministic canvas tests. */
export function visibleApproximation(...regions: (Region | undefined)[]): {
  approximate: boolean;
  reasons: string[];
} {
  const inexact = regions.filter((region): region is Region => !!region && !region.exact);
  return {
    approximate: inexact.length > 0,
    reasons: [...new Set(inexact.flatMap((region) => region.reasons))],
  };
}

const CONE_ALPHA = 0.72;
const previewAlpha = (depth: number) =>
  (0.22 * CONE_ALPHA) / (1 + 0.35 * Math.max(0, depth - 1));

/** How far a non-focused box's cone fades. Fading, never hiding: the peers have
 * to stay legible or there is nothing to compare the focused one against. */
const PEER_FADE = 0.4;
/** Outline weight on the focused box's cone. */
const EMPHASIS_LINE_PX = 2.7;
/** Placeholder until supplied-share is quantitative. Direction owns the ruling's
 * slope; a later analysis may vary this full 0–1 channel, which the renderer
 * spends on the spacing between lines rather than on their weight. */
export const DEFAULT_DOWNSTREAM_DENSITY = 0.46;

const downstreamPattern = (boxIndex: number): NonNullable<Layer["pattern"]> => ({
  kind: "stripe",
  density: DEFAULT_DOWNSTREAM_DENSITY,
  angle: stripeAngleDeg(boxIndex),
});

/** Everything the canvas needs to decide what to paint, as plain data. */
/** @internal Input contract for the directly tested layer builder. */
export type LayerInputs = {
  tensorId: string;
  dark: boolean;
  direction: Direction;
  isSelected: boolean;
  /**
   * The selection parts drawn on *this* tensor, each carrying the index it has
   * in the whole workspace. Parts on other tensors are not this card's business
   * to draw, but the index is: hue, focus and visibility are all keyed by it,
   * so a tile must keep the same colour whichever card it lives on.
   */
  parts: { index: number; box: Box }[];
  /** Parts in the whole workspace, so the rubber band previews the next hue. */
  partCount: number;
  perBox: BoxProp[] | null;
  hiddenBoxes: Set<number>;
  focusedBox: number | null;
  back?: { region: Region; depth: number };
  fwd?: { region: Region; depth: number };
  prev?: { region: Region; depth: number };
  /** The in-progress rubber band, already in element space. */
  dragRegion: Region | null;
};

/**
 * Decide the paint stack for one tensor. Pure so the encoding rules — which
 * cone is filled, which is outlined, what fades, what is hidden — can be
 * asserted directly instead of inferred from pixels.
 */
/** @internal Pure rendering seam exported for deterministic canvas tests. */
export function buildLayers({
  tensorId,
  dark,
  direction,
  isSelected,
  parts,
  partCount,
  perBox,
  hiddenBoxes,
  focusedBox,
  back,
  fwd,
  prev,
  dragRegion,
}: LayerInputs): Layer[] {
  const layers: Layer[] = [];
  const agg = aggregateColors(dark);
  // Both cones are always analysed; `direction` decides which are painted. A
  // Direction filtering happens at paint time; per-tile visibility is shared
  // with the inspector so the picture and merged numbers describe the same set.
  const showBack = direction === "backward" || direction === "both";
  const showFwd = direction === "forward" || direction === "both";

  // Transient hover preview, drawn faintly under everything else.
  if (prev && !isSelected && showBack)
    layers.push({ region: prev.region, color: agg.upstream, alpha: previewAlpha(prev.depth), hatch: false });

  if (perBox) {
    // Hue identifies which selected box produced this region.
    // Focusing a box fades its peers rather than hiding them: this feature
    // exists to compare cones, and blanking every other cone destroys the
    // comparison being made. Only the explicit visibility toggle removes paint.
    const plain: Layer[] = [];
    const emphasised: Layer[] = [];
    perBox.forEach((bp, i) => {
      if (hiddenBoxes.has(i)) return;
      const emph = focusedBox === i;
      const alphaScale = focusedBox !== null && !emph ? PEER_FADE : 1;
      const color = boxColor(i, dark);
      const bTr = bp.backward?.tensors.get(tensorId);
      const fTr = bp.forward?.tensors.get(tensorId);
      const into = emph ? emphasised : plain;
      // Hue means "which box", so direction lives in fill geometry in every
      // view: required input is uniform; downstream reach is ruled.
      if (showBack && bTr && !(isSelected && bTr.depth === 0))
        into.push({
          region: bTr.region,
          color,
          // Persistent cone alpha means hidden-axis coverage in both directions;
          // graph distance is stated exactly by the inspector's dN badge.
          alpha: CONE_ALPHA * alphaScale,
          hatch: !bTr.region.exact,
          outline: emph,
          lineWidth: emph ? EMPHASIS_LINE_PX : undefined,
        });
      if (showFwd && fTr && !(isSelected && fTr.depth === 0))
        into.push({
          region: fTr.region,
          color,
          // Distance is already stated exactly as dN in the inspector. Keeping
          // it out of downstream alpha leaves alpha to hidden-axis coverage and
          // density to the future supplied-share measurement.
          alpha: CONE_ALPHA * alphaScale,
          hatch: !fTr.region.exact,
          // Angle follows the box index, like the hue does. Where two boxes
          // reach the same elements the rulings cross instead of the later one
          // hiding the earlier, which is the only place that overlap is visible.
          pattern: downstreamPattern(i),
        });
    });
    // The emphasised cone draws last so it sits over its faded peers. Scoped
    // to this group on purpose: sorting all layers would also lift it over the
    // selection outline, which is meant to stay on top.
    layers.push(...plain, ...emphasised);
  } else {
    // Too many boxes to attribute: fall back to one hue per direction.
    if (showBack && back && !(isSelected && back.depth === 0))
      layers.push({ region: back.region, color: agg.upstream, alpha: CONE_ALPHA, hatch: !back.region.exact });
    if (showFwd && fwd && !(isSelected && fwd.depth === 0))
      layers.push({
        region: fwd.region,
        color: agg.downstream,
        alpha: CONE_ALPHA,
        hatch: !fwd.region.exact,
        // One merged cone, so there is nothing to tell apart: the base slope.
        pattern: downstreamPattern(0),
      });
  }

  // The selection itself: each box in its own hue, dimmed when another is
  // focused. A hidden box keeps its rectangle — hiding removes the *cone*, and
  // a probe you cannot see is a probe you cannot move back.
  parts.forEach(({ index, box: b }) => {
    const isFocused = focusedBox === null || focusedBox === index;
    const hidden = hiddenBoxes.has(index);
    layers.push({
      region: fromBox(b),
      color: boxColor(index, dark),
      alpha: hidden ? 0.18 : isFocused ? 0.9 : 0.35,
      hatch: false,
      outline: isFocused && !hidden,
    });
  });

  // The in-progress rubber band, drawn on top of everything it will replace.
  if (dragRegion)
    layers.push({
      region: dragRegion,
      color: boxColor(partCount, dark),
      alpha: 0.4,
      hatch: false,
      outline: true,
    });

  return layers;
}

export function TensorCard({
  tensor,
  renderScale = 1,
  viewScale = 1,
  moveHandlers,
}: {
  tensor: Tensor;
  renderScale?: number;
  viewScale?: number;
  moveHandlers?: Pick<
    React.HTMLAttributes<HTMLDivElement>,
    "onPointerDown" | "onPointerMove" | "onPointerUp" | "onPointerCancel" | "onLostPointerCapture"
  >;
}): React.ReactElement {
  const shape = tensor.resolved!;
  const rank = shape.length;
  const cfg = useStore((s) => s.viewCfgs[tensor.id]);
  const setViewCfg = useStore((s) => s.setViewCfg);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  // Subscribe only to this tensor's preview entry. A hover query can touch a
  // subset of the graph; cards outside it retain `undefined` and do not render.
  const prev = useStore((s) => s.preview?.tensors.get(tensor.id));
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
  const theme = useStore((s) => s.theme);
  const setDragging = useStore((s) => s.setDragging);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewKeyRef = useRef<string | null>(null);
  const [drag, setDrag] = useState<{ r0: number; c0: number; r1: number; c1: number } | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  const geom = useMemo(
    () => gridGeometry(shape, cfg, tileScale, graphPx),
    [shape, cfg, tileScale, graphPx]
  );
  const { rowAxis, colAxis } = viewAxes(shape);

  const parts = useMemo(() => partsOn(selection, tensor.id), [selection, tensor.id]);
  const isSelected = parts.length > 0;
  const back = backwardRes?.tensors.get(tensor.id);
  const fwd = forwardRes?.tensors.get(tensor.id);
  const dark = theme === "dark";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const layers = buildLayers({
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
      dragRegion: drag ? fromBox(dragToBox(drag)) : null,
    });
    drawGrid(canvas, shape, cfg, geom, layers, dark, renderScale, viewScale);
  }, [
    back,
    cfg,
    dark,
    direction,
    drag,
    focusedBox,
    fwd,
    geom,
    hiddenBoxes,
    isSelected,
    parts,
    perBox,
    viewScale,
    prev,
    renderScale,
    selection?.parts.length,
    shape,
    snapToGrid,
    tensor.id,
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

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    if (!cell) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    if (previewKeyRef.current !== null) setPreviewBox(null);
    previewKeyRef.current = null;
    setDrag({ r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col });
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    if (drag && cell) {
      setDrag({ ...drag, r1: cell.row, c1: cell.col });
    }
    if (cell) {
      // A hover is the click that has not happened yet. Both the readout and the
      // preview cone are therefore built from the box that click would commit,
      // by the same function the commit uses, so neither can drift from it.
      const box = dragToBox({ r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col });
      const key = formatBoxIndices(box);
      setHover(`(${key})`);
      if (!drag && previewKeyRef.current !== key) {
        previewKeyRef.current = key;
        setPreviewBox(tensor.id, box);
      }
    } else {
      setHover(null);
      if (previewKeyRef.current !== null) setPreviewBox(null);
      previewKeyRef.current = null;
    }
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag) return;
    setDrag(null);
    commit(dragToBox(drag), e);
  };

  const cancelPointerDrag = () => {
    if (!drag) return;
    setDrag(null);
    setDragging(false);
  };

  const onLeave = () => {
    setHover(null);
    if (previewKeyRef.current !== null) setPreviewBox(null);
    previewKeyRef.current = null;
  };

  const totalBytes = shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[tensor.dtype];
  const axisName = (ax: number) => tensor.axisNames?.[ax] ?? `ax${ax}`;
  // Keep the compact header to one reading. The details popover preserves the
  // separate axis-label, symbolic-extent, and numeric-extent facts.
  const symbolicShape = shapeLabel(tensor, "symbolic");
  const numericShape = shapeLabel(tensor, "numeric");
  const shownShape = axisMode === "numeric" ? numericShape : symbolicShape;
  const tileSpanRows = Math.min(geom.rows, geom.tile);
  const tileSpanCols = Math.min(geom.cols, geom.tile);
  const roleTag = tensor.producer ? null : tensor.role === "weight" ? "weight" : "input";
  // Exactness is carried by hatching on the canvas; this repeats it in the
  // header because an over-approximation must never be mistakable for ground
  // truth, and hatching is easy to miss on a small or sparsely covered card.
  const approximation = visibleApproximation(back?.region, fwd?.region);

  return (
    <div className={`tensor-card${isSelected ? " selected" : ""}`} data-tensor={tensor.id}>
      {/* The tensor plate is deliberately frameless. Its persistent label is the
          name, the resolved numeric shape, and the two facts that change how the
          grid below should be read: where the tensor comes from, and whether its
          highlight is exact. */}
      <div className={`tc-header${moveHandlers ? " movable" : ""}`} {...moveHandlers}>
        <span className="tc-name-wrap">
          <span className="tc-name" tabIndex={0}>{tensor.name}</span>
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
        <span className="tc-tile" title="current visible-plane tile size">
          {tileSpanRows}×{tileSpanCols}
        </span>
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
      {rank > 2 && (
        <div className="tc-axes">
          <button
            className={`mini ${cfg.projection ? "on" : ""}`}
            title="this tensor only — projection unions hidden axes; slice uses the slider index"
            onClick={() => setViewCfg(tensor.id, { projection: !cfg.projection })}
          >
            {cfg.projection ? "proj" : "slice"}
          </button>
        </div>
      )}
      {shape.map((e, ax) =>
        ax === rowAxis || ax === colAxis ? null : (
          <div className="tc-slider" key={ax}>
            <span>{axisName(ax)}</span>
            <input
              type="range"
              min={0}
              max={e - 1}
              value={cfg.sliders[ax] ?? 0}
              onChange={(ev) => {
                const sliders = cfg.sliders.slice();
                sliders[ax] = Number(ev.target.value);
                setViewCfg(tensor.id, { sliders });
              }}
            />
            <span className="tc-slider-val">{cfg.sliders[ax] ?? 0}</span>
          </div>
        )
      )}
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

/** Layout footprint of a card at the graph's scale `px`. Independent of the
 * tile: detail never relayouts. */
export function cardSize(
  shape: number[],
  px: number,
  name = "",
  alternateShapeLabels: string[] = []
): { w: number; h: number } {
  const rank = shape.length;
  const { rowAxis, colAxis } = viewAxes(shape);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  const canvas = cardPx(rows, cols, px);
  const hiddenAxes = Math.max(0, rank - (rowAxis >= 0 ? 1 : 0) - (colAxis >= 0 ? 1 : 0));
  // Transparent label + optional higher-rank controls + canvas. There is no
  // decorative outer-card padding: this is the solid collision footprint.
  const h = 24 + (rank > 2 ? 24 : 0) + hiddenAxes * 20 + canvas.h;
  const numericShapeLabel = `[${shape.join(" × ")}]`;
  // Reserve the longest available reading once. Switching the workspace mode
  // must not move cards, and semantic labels may be wider than their numbers.
  const widestShapeLabel = [numericShapeLabel, ...alternateShapeLabels]
    .reduce((longest, label) => label.length > longest.length ? label : longest);
  // Reserve the longest possible clipped tile span too; changing lattice
  // detail must re-rasterise in place rather than trigger a graph relayout.
  const widestTileLabel = `${rows}×${cols}`;
  const labelW = name.length * 9 +
    (widestShapeLabel.length + widestTileLabel.length) * 6.5 + 22;
  const w = Math.max(canvas.w, labelW, 120);
  return { w, h };
}
