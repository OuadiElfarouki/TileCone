import React, { useEffect, useMemo, useRef, useState } from "react";
import { DTYPE_BYTES, Tensor } from "../core/graph";
import { empty, fromBox } from "../core/region";
import { cellFromEvent, drawGrid, gridGeometry, Layer, tileSpan } from "./grid";
import { aggregateColors, boxColor } from "./palette";
import { useStore, viewAxes, ViewCfg } from "./store";

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1 << 20) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1 << 30) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${(n / (1 << 30)).toFixed(2)} GB`;
}

const depthAlpha = (depth: number, base = 0.72) => base / (1 + 0.35 * Math.max(0, depth - 1));

/** How far a non-focused box's cone fades. Fading, never hiding: the peers have
 * to stay legible or there is nothing to compare the focused one against. */
const PEER_FADE = 0.4;
/** Outline weight on the focused box's cone. */
const EMPHASIS_LINE_PX = 2.7;

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

  const isSelected = selection?.tensorId === tensor.id;
  const back = backwardRes?.tensors.get(tensor.id);
  const fwd = forwardRes?.tensors.get(tensor.id);
  const prev = preview?.tensors.get(tensor.id);

  const dark = theme === "dark";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
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
        const bTr = bp.backward?.tensors.get(tensor.id);
        const fTr = bp.forward?.tensors.get(tensor.id);
        const into = emph ? emphasised : plain;
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
            outline: emph,
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
        layers.push({ region: fwd.region, color: agg.downstream, alpha: depthAlpha(fwd.depth), hatch: !fwd.region.exact });
    }

    // The selection itself: each box in its own hue, dimmed when another is
    // focused. A hidden box keeps its rectangle — hiding removes the *cone*, and
    // a probe you cannot see is a probe you cannot move back.
    if (isSelected && selection)
      selection.region.boxes.forEach((b, i) => {
        const isFocused = focusedBox === null || focusedBox === i;
        const hidden = hiddenBoxes.has(i);
        layers.push({
          region: { boxes: [b], exact: selection.region.exact, reasons: [] },
          color: boxColor(i, dark),
          alpha: hidden ? 0.18 : isFocused ? 0.9 : 0.35,
          hatch: false,
          outline: isFocused && !hidden,
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
    const cell = cellFromEvent(e, canvasRef.current!, geom);
    if (!cell) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setDrag({ r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col });
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = cellFromEvent(e, canvasRef.current!, geom);
    if (drag && cell) {
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

  return (
    <div
      ref={rootRef}
      className={`tensor-card${isSelected ? " selected" : ""}`}
      data-tensor={tensor.id}
    >
      {/* The tensor plate is deliberately frameless. Its only persistent label
          is the name plus resolved numeric shape, sitting above the grid. */}
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

export { empty as emptyRegion };

/** Cfg helper exported for GraphView size estimation. */
/** Layout footprint of a card at the graph's scale `px`. Independent of the
 * tile: detail never relayouts. */
export function cardSize(
  shape: number[],
  cfg: ViewCfg,
  px: number,
  name = ""
): { w: number; h: number } {
  const geom = gridGeometry(shape, cfg, 0, px);
  const rank = shape.length;
  const { rowAxis, colAxis } = viewAxes(shape);
  const hiddenAxes = Math.max(0, rank - (rowAxis >= 0 ? 1 : 0) - (colAxis >= 0 ? 1 : 0));
  // Transparent label + optional higher-rank controls + canvas. There is no
  // decorative outer-card padding: this is the solid collision footprint.
  const h = 24 + (rank > 2 ? 24 : 0) + hiddenAxes * 20 + geom.canvasH;
  const shapeLabel = `[${shape.join(" × ")}]`;
  const labelW = name.length * 9 + shapeLabel.length * 6.5 + 12;
  const w = Math.max(geom.canvasW, labelW, 120);
  return { w, h };
}
