import React, { useEffect, useMemo, useRef, useState } from "react";
import { DTYPE_BYTES, Tensor } from "../core/graph";
import { Region, empty, fromBox, intersect } from "../core/region";
import { cellFromEvent, drawGrid, gridGeometry, Layer, tileSpan } from "./grid";
import { aggregateColors, boxColor, isDarkTheme } from "./palette";
import { useStore, viewAxes, ViewCfg } from "./store";

export const COLORS = {
  overlap: [239, 68, 68] as [number, number, number], // red: upstream ∩ downstream
};

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1 << 20) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1 << 30) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${(n / (1 << 30)).toFixed(2)} GB`;
}

const depthAlpha = (depth: number, base = 0.72) => base / (1 + 0.35 * Math.max(0, depth - 1));

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
  const direction = useStore((s) => s.direction);
  const selectMode = useStore((s) => s.selectMode);
  const setSelection = useStore((s) => s.setSelection);
  const setPreviewCell = useStore((s) => s.setPreviewCell);
  const focusTensor = useStore((s) => s.focusTensor);
  const perBox = useStore((s) => s.perBox);
  const focusedBox = useStore((s) => s.focusedBox);
  const tileScale = useStore((s) => s.tileScale);
  const setDragging = useStore((s) => s.setDragging);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ r0: number; c0: number; r1: number; c1: number } | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  const geom = useMemo(() => gridGeometry(shape, cfg, tileScale), [shape, cfg, tileScale]);
  const { rowAxis, colAxis } = viewAxes(shape);

  const isSelected = selection?.tensorId === tensor.id;
  const back = backwardRes?.tensors.get(tensor.id);
  const fwd = forwardRes?.tensors.get(tensor.id);
  const prev = preview?.tensors.get(tensor.id);

  const dark = isDarkTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const layers: Layer[] = [];
    const agg = aggregateColors(dark);

    // Transient hover preview, drawn faintly under everything else.
    if (prev && !isSelected)
      layers.push({ region: prev.region, color: agg.upstream, alpha: 0.22 * depthAlpha(prev.depth), hatch: false });

    if (perBox) {
      // Hue = which selected box this region came from. Fill = upstream,
      // outline = downstream, so direction survives hue being spoken for.
      perBox.forEach((bp, i) => {
        if (focusedBox !== null && focusedBox !== i) return; // isolate the focused box
        const color = boxColor(i, dark);
        const bTr = bp.backward?.tensors.get(tensor.id);
        const fTr = bp.forward?.tensors.get(tensor.id);
        const showBoth = direction === "both";
        if (bTr && !(isSelected && bTr.depth === 0))
          layers.push({
            region: bTr.region,
            color,
            alpha: depthAlpha(bTr.depth),
            hatch: !bTr.region.exact,
          });
        if (fTr && !(isSelected && fTr.depth === 0))
          layers.push({
            region: fTr.region,
            color,
            alpha: depthAlpha(fTr.depth),
            hatch: !fTr.region.exact,
            strokeOnly: showBoth,
          });
      });
    } else {
      // Too many boxes to attribute: fall back to one hue per direction.
      if (back && !(isSelected && back.depth === 0))
        layers.push({ region: back.region, color: agg.upstream, alpha: depthAlpha(back.depth), hatch: !back.region.exact });
      if (fwd && !(isSelected && fwd.depth === 0))
        layers.push({ region: fwd.region, color: agg.downstream, alpha: depthAlpha(fwd.depth), hatch: !fwd.region.exact });
      if (back && fwd && direction === "both" && !isSelected) {
        const ov = intersect(back.region, fwd.region);
        if (ov.boxes.length)
          layers.push({ region: ov, color: COLORS.overlap, alpha: 0.85, hatch: !ov.exact });
      }
    }

    // The selection itself: each box in its own hue, dimmed when another is focused.
    if (isSelected && selection)
      selection.region.boxes.forEach((b, i) => {
        const isFocused = focusedBox === null || focusedBox === i;
        layers.push({
          region: { boxes: [b], exact: selection.region.exact, reasons: [] },
          color: boxColor(i, dark),
          alpha: isFocused ? 0.9 : 0.15,
          hatch: false,
          outline: isFocused,
        });
      });

    if (drag) {
      const b = dragToBox(drag);
      layers.push({
        region: fromBox(b.map(([lo, hi]) => ({ lo, hi }))),
        color: boxColor(selection && isSelected ? selection.region.boxes.length : 0, dark),
        alpha: 0.4,
        hatch: false,
        outline: true,
      });
    }
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
    const [rLo, rHi] = tileSpan(d.r0, d.r1, geom.tile, geom.rows);
    const [cLo, cHi] = tileSpan(d.c0, d.c1, geom.tile, geom.cols);
    return shape.map((e, ax) => {
      if (ax === rowAxis) return [rLo, rHi];
      if (ax === colAxis) return [cLo, cHi];
      return [cfg.sliders[ax] ?? 0, (cfg.sliders[ax] ?? 0) + 1];
    });
  }

  /** Modifier keys win; otherwise the toolbar's compose mode applies. */
  function composeOf(e: React.MouseEvent): "union" | "subtract" | undefined {
    if (e.altKey) return "subtract";
    if (e.shiftKey) return "union";
    return undefined;
  }

  function commit(box: [number, number][], e: React.MouseEvent) {
    setSelection(tensor.id, fromBox(box.map(([lo, hi]) => ({ lo, hi }))), composeOf(e));
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = cellFromEvent(e, canvasRef.current!, geom);
    if (!cell) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    if (selectMode === "all") {
      commit(shape.map((n) => [0, n]), e);
      return;
    }
    if (selectMode === "row") {
      commit(dragToBox({ r0: cell.row, r1: cell.row, c0: 0, c1: geom.tileCols - 1 }), e);
      return;
    }
    if (selectMode === "col") {
      commit(dragToBox({ r0: 0, r1: geom.tileRows - 1, c0: cell.col, c1: cell.col }), e);
      return;
    }
    setDrag({ r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col });
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = cellFromEvent(e, canvasRef.current!, geom);
    if (drag && cell && selectMode !== "cell") {
      setDrag({ ...drag, r1: cell.row, c1: cell.col });
    }
    if (cell) {
      const [rLo, rHi] = tileSpan(cell.row, cell.row, geom.tile, geom.rows);
      const [cLo, cHi] = tileSpan(cell.col, cell.col, geom.tile, geom.cols);
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
    const d = selectMode === "cell" ? { ...drag, r1: drag.r0, c1: drag.c0 } : drag;
    setDrag(null);
    commit(dragToBox(d), e);
  };

  const onLeave = () => {
    setHover(null);
    setPreviewCell(null);
  };

  const totalBytes = shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[tensor.dtype];
  const axisName = (ax: number) => tensor.axisNames?.[ax] ?? `ax${ax}`;
  const isInput = !tensor.producer;
  const region: Region | undefined = back?.region ?? fwd?.region;
  const symShape = tensor.shape.length ? tensor.shape.join("×") : shape.join("×");

  return (
    <div
      ref={rootRef}
      className={`tensor-card${isSelected ? " selected" : ""}`}
      data-tensor={tensor.id}
    >
      {/* The header carries the name only; the literal shape, dtype and size live
          in the hover panel so the cards stay readable at a glance. */}
      <div className="tc-header">
        <span className="tc-name">{tensor.name}</span>
        <span className="tc-badges">
          {isInput &&
            (tensor.role === "weight" ? (
              <span className="badge weight" title="learned parameter">w</span>
            ) : (
              <span className="badge input" title="graph input">in</span>
            ))}
          {region && !region.exact && (
            <span className="badge approx" title={region.reasons.join(", ")}>
              ≈
            </span>
          )}
          <span className="badge tile">▦ {geom.tile}</span>
        </span>
        <div className="tc-info" role="tooltip">
          <div className="tc-info-name">{tensor.name}</div>
          <dl>
            <dt>shape</dt>
            <dd>[{symShape}]</dd>
            {symShape !== shape.join("×") && (
              <>
                <dt>resolved</dt>
                <dd>{shape.join("×")}</dd>
              </>
            )}
            <dt>dtype</dt>
            <dd>{tensor.dtype}</dd>
            <dt>size</dt>
            <dd>{formatBytes(totalBytes)}</dd>
            <dt>view</dt>
            <dd>
              {rowAxis >= 0 ? `${axisName(rowAxis)} × ${axisName(colAxis)}` : colAxis >= 0 ? axisName(colAxis) : "scalar"}
            </dd>
            <dt>tile</dt>
            <dd>
              {geom.tile}×{geom.tile} ({geom.tileRows}×{geom.tileCols} cells)
            </dd>
          </dl>
        </div>
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
          onPointerLeave={onLeave}
        />
        {hover && <div className="tc-tooltip">{hover}</div>}
      </div>
    </div>
  );
}

export { empty as emptyRegion };

/** Cfg helper exported for GraphView size estimation. */
/** Layout footprint of a card. Independent of the tile: detail never relayouts. */
export function cardSize(shape: number[], cfg: ViewCfg): { w: number; h: number } {
  const geom = gridGeometry(shape, cfg, 0);
  const rank = shape.length;
  const { rowAxis, colAxis } = viewAxes(shape);
  const hiddenAxes = Math.max(0, rank - (rowAxis >= 0 ? 1 : 0) - (colAxis >= 0 ? 1 : 0));
  // header + optional projection toggle + one slider per hidden axis + canvas + padding
  const h = 30 + (rank > 2 ? 24 : 0) + hiddenAxes * 20 + geom.canvasH + 18;
  const w = Math.max(geom.canvasW + 18, 120);
  return { w, h };
}
