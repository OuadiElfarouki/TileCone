import React, { useEffect, useMemo, useRef, useState } from "react";
import { DTYPE_BYTES, Tensor } from "../core/graph";
import { Box, fromBox, Region } from "../core/region";
import { drawGrid, elementFromEvent, gridGeometry, Layer, snapSpan, tileSpan } from "./grid";
import { aggregateColors, boxColor } from "./palette";
import { BoxProp, Direction, partsOn, useStore, viewAxes } from "./store";
import { cardPx, planeExtents } from "./tiling";

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1 << 20) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1 << 30) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${(n / (1 << 30)).toFixed(2)} GB`;
}

/** Combine exactness across every cone currently visible on a tensor card. */
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

const depthAlpha = (depth: number, base = 0.72) => base / (1 + 0.35 * Math.max(0, depth - 1));

/** How far a non-focused box's cone fades. Fading, never hiding: the peers have
 * to stay legible or there is nothing to compare the focused one against. */
const PEER_FADE = 0.4;
/** Outline weight on the focused box's cone. */
const EMPHASIS_LINE_PX = 2.7;

/** Everything the canvas needs to decide what to paint, as plain data. */
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

  // Transient hover preview, drawn faintly under everything else.
  if (prev && !isSelected)
    layers.push({ region: prev.region, color: agg.upstream, alpha: 0.22 * depthAlpha(prev.depth), hatch: false });

  if (perBox) {
    // Hue identifies which selected box produced this region.
    // Focusing a box fades its peers rather than hiding them: this feature
    // exists to compare cones, and blanking every other cone destroys the
    // comparison being made. Only the explicit visibility toggle removes paint.
    const plain: Layer[] = [];
    const emphasised: Layer[] = [];
    perBox.forEach((bp, i) => {
      if (hiddenBoxes.has(i)) return; // parked by the user; metrics stay live
      const emph = focusedBox === i;
      const alphaScale = focusedBox !== null && !emph ? PEER_FADE : 1;
      const color = boxColor(i, dark);
      const bTr = bp.backward?.tensors.get(tensorId);
      const fTr = bp.forward?.tensors.get(tensorId);
      const into = emph ? emphasised : plain;
      // With both cones on screen at once, hue already means "which box", so
      // fill vs outline is what is left to carry up- versus downstream. With a
      // single direction there is nothing to disambiguate, so it stays filled.
      const outlineDownstream = direction === "both";
      if (bTr && !(isSelected && bTr.depth === 0))
        into.push({
          region: bTr.region,
          color,
          alpha: depthAlpha(bTr.depth) * alphaScale,
          hatch: !bTr.region.exact,
          outline: emph,
          lineWidth: emph ? EMPHASIS_LINE_PX : undefined,
        });
      if (fTr && !(isSelected && fTr.depth === 0))
        into.push({
          region: fTr.region,
          color,
          alpha: depthAlpha(fTr.depth) * alphaScale,
          hatch: !fTr.region.exact,
          strokeOnly: outlineDownstream,
          // a stroke-only layer never reaches drawGrid's outline branch, so
          // emphasis rides on the stroke width instead
          outline: emph && !outlineDownstream,
          lineWidth: emph ? EMPHASIS_LINE_PX : undefined,
        });
    });
    // The emphasised cone draws last so it sits over its faded peers. Scoped
    // to this group on purpose: sorting all layers would also lift it over the
    // selection outline, which is meant to stay on top.
    layers.push(...plain, ...emphasised);
  } else {
    // Too many boxes to attribute: fall back to one hue per direction.
    if (back && !(isSelected && back.depth === 0))
      layers.push({ region: back.region, color: agg.upstream, alpha: depthAlpha(back.depth), hatch: !back.region.exact });
    if (fwd && !(isSelected && fwd.depth === 0))
      layers.push({
        region: fwd.region,
        color: agg.downstream,
        alpha: depthAlpha(fwd.depth),
        hatch: !fwd.region.exact,
        strokeOnly: direction === "both",
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
}: {
  tensor: Tensor;
  renderScale?: number;
}): React.ReactElement {
  const shape = tensor.resolved!;
  const rank = shape.length;
  const cfg = useStore((s) => s.viewCfgs[tensor.id]);
  const setViewCfg = useStore((s) => s.setViewCfg);
  const selection = useStore((s) => s.selection);
  const backwardRes = useStore((s) => s.backwardRes);
  const forwardRes = useStore((s) => s.forwardRes);
  const preview = useStore((s) => s.preview);
  const setSelection = useStore((s) => s.setSelection);
  const setPreviewCell = useStore((s) => s.setPreviewCell);
  const focusTensor = useStore((s) => s.focusTensor);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);
  const hiddenBoxes = useStore((s) => s.hiddenBoxes);
  const direction = useStore((s) => s.direction);
  const snapToGrid = useStore((s) => s.snapToGrid);
  const tileScale = useStore((s) => s.tileScale);
  const graphPx = useStore((s) => s.graphPx);
  const theme = useStore((s) => s.theme);
  const setDragging = useStore((s) => s.setDragging);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
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
  const prev = preview?.tensors.get(tensor.id);

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
      dragRegion: drag
        ? fromBox(dragToBox(drag).map(([lo, hi]) => ({ lo, hi })))
        : null,
    });
    drawGrid(canvas, shape, cfg, geom, layers, dark, renderScale);
  });

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

  useEffect(() => {
    if (focusTensor === tensor.id)
      rootRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [focusTensor, tensor.id]);

  /** Drag rectangle in tile-cell coordinates -> element-space box. */
  function dragToBox(d: { r0: number; c0: number; r1: number; c1: number }): [number, number][] {
    // The drag is tracked in elements; snapping is a presentation choice applied
    // at the end, so turning it off costs no precision that was ever available.
    const [rLo, rHi] = snapToGrid
      ? snapSpan(d.r0, d.r1, geom.tile, geom.rows)
      : [Math.min(d.r0, d.r1), Math.max(d.r0, d.r1) + 1];
    const [cLo, cHi] = snapToGrid
      ? snapSpan(d.c0, d.c1, geom.tile, geom.cols)
      : [Math.min(d.c0, d.c1), Math.max(d.c0, d.c1) + 1];
    return shape.map((e, ax) => {
      if (ax === rowAxis) return [rLo, rHi];
      if (ax === colAxis) return [cLo, cHi];
      return [cfg.sliders[ax] ?? 0, (cfg.sliders[ax] ?? 0) + 1];
    });
  }

  /** Direct drawing adds by default; Alt turns the gesture into subtraction. */
  function composeOf(e: React.MouseEvent): "union" | "subtract" | undefined {
    if (e.altKey) return "subtract";
    if (e.shiftKey) return "union";
    return undefined;
  }

  function commit(box: [number, number][], e: React.MouseEvent) {
    setSelection(tensor.id, fromBox(box.map(([lo, hi]) => ({ lo, hi }))), composeOf(e));
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    if (!cell) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setDrag({ r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col });
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = elementFromEvent(e, canvasRef.current!, geom);
    if (drag && cell) {
      setDrag({ ...drag, r1: cell.row, c1: cell.col });
    }
    if (cell) {
      // The readout follows the gesture: whole cells when snapping, the element
      // under the pointer when not.
      const [rLo, rHi] = snapToGrid
        ? tileSpan(Math.floor(cell.row / geom.tile), Math.floor(cell.row / geom.tile), geom.tile, geom.rows)
        : [cell.row, cell.row + 1];
      const [cLo, cHi] = snapToGrid
        ? tileSpan(Math.floor(cell.col / geom.tile), Math.floor(cell.col / geom.tile), geom.tile, geom.cols)
        : [cell.col, cell.col + 1];
      const label = shape.map((_, ax) => {
        if (ax === rowAxis) return rHi - rLo === 1 ? `${rLo}` : `${rLo}:${rHi}`;
        if (ax === colAxis) return cHi - cLo === 1 ? `${cLo}` : `${cLo}:${cHi}`;
        return `${cfg.sliders[ax] ?? 0}`;
      });
      setHover(`(${label.join(", ")})`);
      if (!drag)
        setPreviewCell(
          tensor.id,
          shape.map((_, ax) => (ax === rowAxis ? rLo : ax === colAxis ? cLo : cfg.sliders[ax] ?? 0))
        );
    } else {
      setHover(null);
      setPreviewCell(null);
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
    setPreviewCell(null);
  };

  const totalBytes = shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[tensor.dtype];
  const axisName = (ax: number) => tensor.axisNames?.[ax] ?? `ax${ax}`;
  const numericShape = `[${shape.join(" × ")}]`;
  const roleTag = tensor.producer ? null : tensor.role === "weight" ? "weight" : "input";
  // Exactness is carried by hatching on the canvas; this repeats it in the
  // header because an over-approximation must never be mistakable for ground
  // truth, and hatching is easy to miss on a small or sparsely covered card.
  const approximation = visibleApproximation(back?.region, fwd?.region);

  return (
    <div
      ref={rootRef}
      className={`tensor-card${isSelected ? " selected" : ""}`}
      data-tensor={tensor.id}
    >
      {/* The tensor plate is deliberately frameless. Its persistent label is the
          name, the resolved numeric shape, and the two facts that change how the
          grid below should be read: where the tensor comes from, and whether its
          highlight is exact. */}
      <div className="tc-header">
        <span className="tc-name-wrap">
          <span className="tc-name" tabIndex={0}>{tensor.name}</span>
          <span className="tc-info" role="tooltip">
            <span>shape</span><b>{numericShape}</b>
            <span>dtype</span><b>{tensor.dtype}</b>
            <span>size</span><b>{formatBytes(totalBytes)}</b>
          </span>
        </span>
        <span className="tc-shape">{numericShape}</span>
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
            title="projection: union over hidden axes; slice: membership at slider index only"
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
  name = ""
): { w: number; h: number } {
  const rank = shape.length;
  const { rowAxis, colAxis } = viewAxes(shape);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  const canvas = cardPx(rows, cols, px);
  const hiddenAxes = Math.max(0, rank - (rowAxis >= 0 ? 1 : 0) - (colAxis >= 0 ? 1 : 0));
  // Transparent label + optional higher-rank controls + canvas. There is no
  // decorative outer-card padding: this is the solid collision footprint.
  const h = 24 + (rank > 2 ? 24 : 0) + hiddenAxes * 20 + canvas.h;
  const shapeLabel = `[${shape.join(" × ")}]`;
  const labelW = name.length * 9 + shapeLabel.length * 6.5 + 12;
  const w = Math.max(canvas.w, labelW, 120);
  return { w, h };
}
