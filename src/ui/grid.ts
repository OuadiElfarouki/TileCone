/**
 * Canvas grid rendering for tensor cards. Pure drawing + geometry; no React.
 *
 * One drawn cell = one tile. A cell's fill alpha is the *coverage* of that tile
 * by the region: the fraction of the tile's elements that are in it, including
 * the hidden-axis fraction. So a partially-covered tile reads as partially
 * filled rather than being rounded to all-or-nothing, and the picture stays
 * honest at every zoom level instead of degrading into sub-pixel noise.
 */

import { Box, Region } from "../core/region";
import { CARD_SURFACE } from "./palette";
import { cardPx, planeExtents, tileFor } from "./tiling";
import { viewAxes, ViewCfg } from "./store";

export type GridGeom = {
  rows: number; // element extent
  cols: number;
  tile: number; // elements per cell, both axes
  tileRows: number; // drawn cells
  tileCols: number;
  cellW: number; // CSS px per drawn cell (may be fractional; drawing snaps to px)
  cellH: number;
  canvasW: number; // fixed by the tensor's shape, independent of `tile`
  canvasH: number;
  rowAxis: number;
  colAxis: number;
};

export function gridGeometry(
  shape: number[],
  _cfg: ViewCfg,
  tileScale: number,
  px: number
): GridGeom {
  const { rowAxis, colAxis } = viewAxes(shape);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  // The card is sized by the shape and the graph's scale; the tile only sets the
  // lattice inside it.
  const { w: canvasW, h: canvasH } = cardPx(rows, cols, px);
  const tile = tileFor(rows, cols, tileScale, px);
  const tileRows = Math.ceil(rows / tile);
  const tileCols = Math.ceil(cols / tile);
  return {
    rows,
    cols,
    tile,
    tileRows,
    tileCols,
    cellW: canvasW / tileCols,
    cellH: canvasH / tileRows,
    canvasW,
    canvasH,
    rowAxis,
    colAxis,
  };
}

export type Layer = {
  region: Region;
  color: [number, number, number];
  alpha: number; // base alpha (depth shading already applied by caller)
  hatch: boolean; // over-approximation -> diagonal hatching
  outline?: boolean; // strong border (selection)
  /** Outline weight. Emphasis uses a heavier stroke than the 1.5 default. */
  lineWidth?: number;
  /**
   * Draw only a border, no fill. This is how *direction* stays readable once
   * hue is spoken for by box identity: in "both" mode the upstream cone is
   * filled and the downstream cone is outlined, so a region that is each can be
   * told apart without a second hue.
   */
  strokeOnly?: boolean;
};

/** Fraction of a box's hidden-axis volume that is currently visible. */
function hiddenFraction(box: Box, shape: number[], cfg: ViewCfg, geom: GridGeom): number {
  let frac = 1;
  for (let ax = 0; ax < shape.length; ax++) {
    if (ax === geom.rowAxis || ax === geom.colAxis) continue;
    const I = box[ax];
    if (cfg.projection) {
      frac *= (I.hi - I.lo) / shape[ax];
    } else {
      const v = cfg.sliders[ax] ?? 0;
      if (v < I.lo || v >= I.hi) return 0;
    }
  }
  return frac;
}

/** Never let a thin region vanish. Over-stating extent is the safe direction. */
export const MIN_MARK_PX = 1;

export type RegionRect = { x: number; y: number; w: number; h: number; alpha: number };

/**
 * A region as exact rectangles in canvas pixels.
 *
 * Regions are drawn at *element* precision, not quantised to the tile lattice.
 * The lattice is a reading aid drawn on top; the canvas itself maps elements to
 * pixels linearly, so the true rectangle is always drawable. Quantising instead
 * would show a half-lit cell wherever a region ended mid-tile, which reads as
 * "partly selected" when the truth is "these exact elements".
 *
 * The one genuinely fractional quantity survives as `alpha`: in projection mode
 * a box covering part of a hidden axis really does represent a fraction of what
 * the drawn cell stands for. That is about axes not on screen, so it cannot be
 * expressed geometrically here.
 *
 * Pure and DOM-free so the geometry can be tested directly.
 */
export function regionRects(
  region: Region,
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom
): RegionRect[] {
  const { rowAxis, colAxis, rows, cols, canvasW, canvasH } = geom;
  const rects: RegionRect[] = [];
  for (const box of region.boxes) {
    const alpha = hiddenFraction(box, shape, cfg, geom);
    if (alpha <= 0) continue;
    const rI = rowAxis >= 0 ? box[rowAxis] : { lo: 0, hi: 1 };
    const cI = colAxis >= 0 ? box[colAxis] : { lo: 0, hi: 1 };
    const x = (Math.max(0, cI.lo) / cols) * canvasW;
    const y = (Math.max(0, rI.lo) / rows) * canvasH;
    const x1 = (Math.min(cols, cI.hi) / cols) * canvasW;
    const y1 = (Math.min(rows, rI.hi) / rows) * canvasH;
    if (x1 <= x || y1 <= y) continue;
    rects.push({
      x,
      y,
      w: Math.min(Math.max(MIN_MARK_PX, x1 - x), canvasW - x),
      h: Math.min(Math.max(MIN_MARK_PX, y1 - y), canvasH - y),
      alpha,
    });
  }
  return rects;
}

/** Conservative cross-browser ceiling on a canvas backing-store side. */
const MAX_CANVAS_DIM = 8192;

const hatchCache = new Map<string, CanvasPattern | null>();

function hatchPattern(
  ctx: CanvasRenderingContext2D,
  color: [number, number, number]
): CanvasPattern | null {
  const key = color.join(",");
  if (hatchCache.has(key)) return hatchCache.get(key)!;
  const c = document.createElement("canvas");
  c.width = 8;
  c.height = 8;
  const g = c.getContext("2d")!;
  g.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.9)`;
  g.lineWidth = 1.6;
  g.beginPath();
  g.moveTo(-2, 6);
  g.lineTo(6, -2);
  g.moveTo(2, 10);
  g.lineTo(10, 2);
  g.stroke();
  const p = ctx.createPattern(c, "repeat");
  hatchCache.set(key, p);
  return p;
}

export function drawGrid(
  canvas: HTMLCanvasElement,
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom,
  layers: Layer[],
  dark: boolean,
  renderScale = 1
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  // Supersample by the current graph zoom so cards stay sharp when zoomed in,
  // but never past the backing-store limit: since the card scale became a
  // property of the graph, a very wide tensor can exceed what the old per-card
  // cap used to make impossible, and a canvas over the limit renders blank
  // rather than clipped. Softening is the acceptable failure here.
  const res = Math.min(
    (window.devicePixelRatio || 1) * renderScale,
    MAX_CANVAS_DIM / Math.max(1, geom.canvasW),
    MAX_CANVAS_DIM / Math.max(1, geom.canvasH)
  );
  canvas.width = Math.ceil(geom.canvasW * res);
  canvas.height = Math.ceil(geom.canvasH * res);
  ctx.setTransform(res, 0, 0, res, 0, 0);
  ctx.clearRect(0, 0, geom.canvasW, geom.canvasH);

  ctx.fillStyle = dark ? CARD_SURFACE.dark : CARD_SURFACE.light;
  ctx.fillRect(0, 0, geom.canvasW, geom.canvasH);

  const { tileRows, tileCols } = geom;

  for (const layer of layers) {
    const [r, g, b] = layer.color;
    const rects = regionRects(layer.region, shape, cfg, geom);

    if (layer.strokeOnly) {
      ctx.strokeStyle = `rgba(${r},${g},${b},${Math.min(1, layer.alpha + 0.25)})`;
      ctx.lineWidth = layer.lineWidth ?? 2;
      for (const q of rects) ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.w - 1, q.h - 1);
      continue;
    }

    for (const q of rects) {
      ctx.fillStyle = `rgba(${r},${g},${b},${Math.min(1, layer.alpha * q.alpha)})`;
      ctx.fillRect(q.x, q.y, q.w, q.h);
    }

    if (layer.hatch) {
      const p = hatchPattern(ctx, [r, g, b]);
      if (p) {
        ctx.save();
        ctx.fillStyle = p;
        for (const q of rects) {
          ctx.globalAlpha = Math.min(1, layer.alpha * q.alpha);
          ctx.fillRect(q.x, q.y, q.w, q.h);
        }
        ctx.restore();
      }
    }

    if (layer.outline) {
      ctx.strokeStyle = `rgba(${r},${g},${b},0.95)`;
      ctx.lineWidth = layer.lineWidth ?? 1.5;
      for (const q of rects) ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.w - 1, q.h - 1);
    }
  }

  // tile boundaries — the cell grid *is* the tile grid
  if (Math.min(geom.cellW, geom.cellH) >= 5) {
    ctx.strokeStyle = dark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.10)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let c = 1; c < tileCols; c++) {
      const x = Math.round(c * geom.cellW) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, geom.canvasH);
    }
    for (let r = 1; r < tileRows; r++) {
      const y = Math.round(r * geom.cellH) + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(geom.canvasW, y);
    }
    ctx.stroke();
  }

  ctx.strokeStyle = dark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, geom.canvasW - 1, geom.canvasH - 1);
}

/** The tile size a tensor renders at, without building full geometry. */
export function tileOf(shape: number[], tileScale: number, px: number): number {
  const { rowAxis, colAxis } = viewAxes(shape);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  return tileFor(rows, cols, tileScale, px);
}

/**
 * How far one arrow-key nudge moves the selection.
 *
 * It is whatever unit the pointer works in: a whole tile while snapping, a
 * single element when not. Stepping by a tile with snapping off would let the
 * keyboard place a box at offsets a drag cannot reach.
 */
export function nudgeUnit(
  shape: number[],
  tileScale: number,
  px: number,
  snapToGrid: boolean
): number {
  return snapToGrid ? tileOf(shape, tileScale, px) : 1;
}

/**
 * Pixel position -> element index. The canvas always spans the tensor's full
 * extent, so element resolution is available regardless of the tile lattice
 * drawn on top of it; this is what lets a drag cut an unsnapped range.
 */
export function elementFromEvent(
  e: { clientX: number; clientY: number },
  canvas: HTMLCanvasElement,
  geom: GridGeom
): { row: number; col: number } | null {
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * geom.canvasW;
  const y = ((e.clientY - rect.top) / rect.height) * geom.canvasH;
  if (x < 0 || y < 0 || x >= geom.canvasW || y >= geom.canvasH) return null;
  return {
    row: Math.min(geom.rows - 1, Math.max(0, Math.floor((y / geom.canvasH) * geom.rows))),
    col: Math.min(geom.cols - 1, Math.max(0, Math.floor((x / geom.canvasW) * geom.cols))),
  };
}

/** Element range -> the interval covering it, snapped out to whole tiles. */
export function snapSpan(e0: number, e1: number, tile: number, extent: number): [number, number] {
  const lo = Math.min(e0, e1);
  const hi = Math.max(e0, e1);
  return [
    Math.max(0, Math.floor(lo / tile) * tile),
    Math.min(extent, (Math.floor(hi / tile) + 1) * tile),
  ];
}

/** Tile-cell range -> element interval on that axis, clamped to the extent. */
export function tileSpan(t0: number, t1: number, tile: number, extent: number): [number, number] {
  const lo = Math.max(0, Math.min(t0, t1) * tile);
  const hi = Math.min(extent, (Math.max(t0, t1) + 1) * tile);
  return [lo, hi];
}
