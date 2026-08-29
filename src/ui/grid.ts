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

export function gridGeometry(shape: number[], _cfg: ViewCfg, tileScale: number): GridGeom {
  const { rowAxis, colAxis } = viewAxes(shape);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  // The card is sized by the shape alone; the tile only sets the lattice inside it.
  const { w: canvasW, h: canvasH } = cardPx(rows, cols);
  const tile = tileFor(rows, cols, tileScale);
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

/** Pixel bounds of one tile cell, snapped to whole pixels so cells tile seamlessly. */
export function cellRect(
  geom: GridGeom,
  tr: number,
  tc: number
): { x: number; y: number; w: number; h: number } {
  const x0 = Math.round(tc * geom.cellW);
  const x1 = Math.round((tc + 1) * geom.cellW);
  const y0 = Math.round(tr * geom.cellH);
  const y1 = Math.round((tr + 1) * geom.cellH);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

export type Layer = {
  region: Region;
  color: [number, number, number];
  alpha: number; // base alpha (depth shading already applied by caller)
  hatch: boolean; // over-approximation -> diagonal hatching
  outline?: boolean; // strong border (selection)
  /** Draw only a border, no fill. Marks downstream regions in "both" mode, so
   * direction stays readable when hue is already carrying box identity. */
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

/**
 * Per-cell coverage in [0,1], row-major over the tile grid. Pure and DOM-free so
 * it can be unit-tested. Region boxes are disjoint after canonicalize, so the
 * per-box contributions sum without double counting.
 */
export function tileCoverage(
  region: Region,
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom
): Float32Array {
  const acc = new Float32Array(geom.tileRows * geom.tileCols);
  const { tile, tileRows, tileCols, rowAxis, colAxis, rows, cols } = geom;
  for (const box of region.boxes) {
    const hf = hiddenFraction(box, shape, cfg, geom);
    if (hf <= 0) continue;
    const rI = rowAxis >= 0 ? box[rowAxis] : { lo: 0, hi: 1 };
    const cI = colAxis >= 0 ? box[colAxis] : { lo: 0, hi: 1 };
    const tr0 = Math.max(0, Math.floor(rI.lo / tile));
    const tr1 = Math.min(tileRows - 1, Math.floor((rI.hi - 1) / tile));
    const tc0 = Math.max(0, Math.floor(cI.lo / tile));
    const tc1 = Math.min(tileCols - 1, Math.floor((cI.hi - 1) / tile));
    for (let tr = tr0; tr <= tr1; tr++) {
      const rOverlap =
        Math.min(rI.hi, (tr + 1) * tile) - Math.max(rI.lo, tr * tile);
      if (rOverlap <= 0) continue;
      for (let tc = tc0; tc <= tc1; tc++) {
        const cOverlap =
          Math.min(cI.hi, (tc + 1) * tile) - Math.max(cI.lo, tc * tile);
        if (cOverlap <= 0) continue;
        acc[tr * tileCols + tc] += rOverlap * cOverlap * hf;
      }
    }
  }
  // Normalise by each tile's true element count. Edge tiles are clipped by the
  // tensor bounds, so a fully-selected partial tile must still read as 1.0.
  for (let tr = 0; tr < tileRows; tr++) {
    const rExt = Math.min((tr + 1) * tile, rows) - tr * tile;
    for (let tc = 0; tc < tileCols; tc++) {
      const cExt = Math.min((tc + 1) * tile, cols) - tc * tile;
      const i = tr * tileCols + tc;
      const vol = rExt * cExt;
      acc[i] = vol > 0 ? Math.min(1, acc[i] / vol) : 0;
    }
  }
  return acc;
}

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

/** Outer boundary of the covered cells, so a region reads as one shape. */
function strokeCoveredEdges(
  ctx: CanvasRenderingContext2D,
  cov: Float32Array,
  geom: GridGeom,
  inset: number
): void {
  const { tileRows, tileCols } = geom;
  const on = (r: number, c: number) =>
    r >= 0 && c >= 0 && r < tileRows && c < tileCols && cov[r * tileCols + c] > 0;
  ctx.beginPath();
  for (let r = 0; r < tileRows; r++) {
    for (let c = 0; c < tileCols; c++) {
      if (!on(r, c)) continue;
      const { x, y, w, h } = cellRect(geom, r, c);
      if (!on(r - 1, c)) {
        ctx.moveTo(x, y + inset);
        ctx.lineTo(x + w, y + inset);
      }
      if (!on(r + 1, c)) {
        ctx.moveTo(x, y + h - inset);
        ctx.lineTo(x + w, y + h - inset);
      }
      if (!on(r, c - 1)) {
        ctx.moveTo(x + inset, y);
        ctx.lineTo(x + inset, y + h);
      }
      if (!on(r, c + 1)) {
        ctx.moveTo(x + w - inset, y);
        ctx.lineTo(x + w - inset, y + h);
      }
    }
  }
  ctx.stroke();
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
  // Supersample by the current graph zoom so cards stay sharp when zoomed in.
  const res = (window.devicePixelRatio || 1) * renderScale;
  canvas.width = Math.ceil(geom.canvasW * res);
  canvas.height = Math.ceil(geom.canvasH * res);
  ctx.setTransform(res, 0, 0, res, 0, 0);
  ctx.clearRect(0, 0, geom.canvasW, geom.canvasH);

  ctx.fillStyle = dark ? "#16181d" : "#f3f4f6";
  ctx.fillRect(0, 0, geom.canvasW, geom.canvasH);

  const { tileRows, tileCols } = geom;

  for (const layer of layers) {
    const [r, g, b] = layer.color;
    const cov = tileCoverage(layer.region, shape, cfg, geom);

    if (layer.strokeOnly) {
      ctx.strokeStyle = `rgba(${r},${g},${b},${Math.min(1, layer.alpha + 0.25)})`;
      ctx.lineWidth = 2;
      strokeCoveredEdges(ctx, cov, geom, 1);
      continue;
    }

    for (let tr = 0; tr < tileRows; tr++) {
      for (let tc = 0; tc < tileCols; tc++) {
        const f = cov[tr * tileCols + tc];
        if (f <= 0) continue;
        ctx.fillStyle = `rgba(${r},${g},${b},${Math.min(1, layer.alpha * f)})`;
        const q = cellRect(geom, tr, tc);
        ctx.fillRect(q.x, q.y, q.w, q.h);
      }
    }

    if (layer.hatch) {
      const p = hatchPattern(ctx, [r, g, b]);
      if (p) {
        ctx.save();
        ctx.fillStyle = p;
        for (let tr = 0; tr < tileRows; tr++) {
          for (let tc = 0; tc < tileCols; tc++) {
            const f = cov[tr * tileCols + tc];
            if (f <= 0) continue;
            ctx.globalAlpha = Math.min(1, layer.alpha * f);
            const q = cellRect(geom, tr, tc);
            ctx.fillRect(q.x, q.y, q.w, q.h);
          }
        }
        ctx.restore();
      }
    }

    if (layer.outline) {
      ctx.strokeStyle = `rgba(${r},${g},${b},0.95)`;
      ctx.lineWidth = 1.5;
      strokeCoveredEdges(ctx, cov, geom, 0.75);
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
export function tileOf(shape: number[], tileScale: number): number {
  const { rowAxis, colAxis } = viewAxes(shape);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  return tileFor(rows, cols, tileScale);
}

/** Pixel position -> tile cell. */
export function cellFromEvent(
  e: { clientX: number; clientY: number },
  canvas: HTMLCanvasElement,
  geom: GridGeom
): { row: number; col: number } | null {
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * geom.canvasW;
  const y = ((e.clientY - rect.top) / rect.height) * geom.canvasH;
  const col = Math.floor(x / geom.cellW);
  const row = Math.floor(y / geom.cellH);
  if (col < 0 || col >= geom.tileCols || row < 0 || row >= geom.tileRows) return null;
  return { row, col };
}

/** Tile-cell range -> element interval on that axis, clamped to the extent. */
export function tileSpan(t0: number, t1: number, tile: number, extent: number): [number, number] {
  const lo = Math.max(0, Math.min(t0, t1) * tile);
  const hi = Math.min(extent, (Math.max(t0, t1) + 1) * tile);
  return [lo, hi];
}
