/**
 * Tensor-card grid geometry: which elements a cell stands for, where a region
 * lands in canvas pixels, and how gestures snap and step. Pure and DOM-free;
 * the canvas drawing that consumes it is `components/card/draw-grid.ts`.
 *
 * One drawn cell = one tile. A cell's fill alpha is the *coverage* of that tile
 * by the region: the fraction of the tile's elements that are in it, including
 * the hidden-axis fraction. So a partially-covered tile reads as partially
 * filled rather than being rounded to all-or-nothing, and the picture stays
 * honest at every zoom level instead of degrading into sub-pixel noise.
 */

import { Box, Interval, Region, disjointify } from "../../core/region";
import { cardPx, planeExtents, tileFor } from "./tiling";
import { viewAxes, type ViewCfg } from "./tensor-view";
import { gestureTile } from "./tile-spec";
import { cardScaleFor } from "./card-size";

export type GridGeom = {
  rows: number; // element extent
  cols: number;
  /** Elements per cell on the row and column axes. Equal unless the tensor has
   * a tile of its own (`ViewCfg.tile`), which need not be square. */
  rowTile: number;
  colTile: number;
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
  cfg: ViewCfg | undefined,
  tileScale: number,
  px: number
): GridGeom {
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  // The card is sized by the shape and the graph's scale; the tile only sets the
  // lattice inside it. A chosen pair of axes may take a smaller scale of its own.
  const scale = cardScaleFor(shape, cfg, px);
  const { w: canvasW, h: canvasH } = cardPx(rows, cols, scale);
  // A tile of the tensor's own is a quantity and is drawn as given, however
  // dense; `latticeStride` keeps the drawn boundaries apart on screen.
  const tile = gestureTile(shape, cfg, tileFor(rows, cols, tileScale, scale));
  const rowTile = rowAxis >= 0 ? tile[rowAxis] : 1;
  const colTile = colAxis >= 0 ? tile[colAxis] : 1;
  const tileRows = Math.ceil(rows / rowTile);
  const tileCols = Math.ceil(cols / colTile);
  return {
    rows,
    cols,
    rowTile,
    colTile,
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
  /**
   * Draw this layer's pattern in the card surface colour rather than its hue.
   *
   * For a mark that lands on a solid fill of its own hue, where drawing in that
   * hue would be drawing nothing. The approximation hatch has always done this;
   * the stipple needs it for the same reason and more often, because what a
   * tile is combined with routinely lands inside what it reads.
   */
  knockout?: boolean;
  seed?: boolean; // external corner marks identify the region the user placed
  outline?: boolean; // strong border (selection)
  /** Outline weight. Emphasis uses a heavier stroke than the 1.5 default. */
  lineWidth?: number;
  /**
   * Direction is fill geometry, not a perimeter: hue belongs to tile identity,
   * solid fill means "needs", and a diagonal ruling means "feeds". `density`
   * sets the spacing between rulings : further apart for a smaller share : and
   * stays explicit so supplied-share can own it later without changing the
   * layer contract. `angle` separates one box's ruling from another's, so two
   * cones that reach the same elements cross there instead of hiding.
   */
  pattern?:
    | { kind: "stripe"; density: number; angle: number }
    /**
     * Entanglement: what the tile is *combined with*, as opposed to what it
     * reads or feeds. A third relation needs a third texture, and it has to be
     * one neither of the others can be mistaken for. A ruling at a new angle
     * would read as another cone, and a denser ruling as another share; a
     * stipple is the one mark here that is not a line, so it cannot be confused
     * with the downstream ruling or the approximation hatch.
     */
    | { kind: "stipple"; density: number };
};

/**
 * What the Plan view draws on a card besides its layers.
 *
 * `lattice` replaces the display lattice: in the Plan view the lines on a card
 * are the plan's tile boundaries, and a tensor the plan does not tile has none.
 * `tiles` are the producer tiles the inspected task needs, outlined in neutral
 * ink. The task's demand is a layer filled inside them, so the unfilled part
 * of an outlined tile is the part the task does not read.
 */
export type PlanPaint = {
  /**
   * The tile extents on the visible row and column axes, in elements.
   * `proposed` draws them fainter: the tensor is not tiled yet and this is the
   * cover a click would create, not part of the plan.
   */
  lattice: { rows: number; cols: number; proposed?: boolean } | null;
  /** Needed producer tiles; `definite` false draws the outline dashed. */
  tiles: { box: Box; definite: boolean }[];
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
/** @internal Exported with `regionRects` for renderer invariant tests. */
export const MIN_MARK_PX = 1;

/**
 * Screen-space stride threshold for the drawn lattice, in CSS px.
 *
 * Distinct from `tiling.MIN_CELL_PX`, which is a *canvas*-space budget used to
 * choose the tile: one constant compared in two coordinate systems is what let
 * the lattice disappear below 100% zoom while snapping still bound to it.
 */
export const MIN_LATTICE_PX = 5;

/**
 * How many tile boundaries to skip so the drawn ones stay `MIN_LATTICE_PX`
 * apart on screen. Always a power of two, so every drawn line is also a
 * snapping boundary; the lattice reads coarser as the view zooms out instead of
 * collapsing into a wash or vanishing.
 *
 * Note what this does not give: the drawn lines are a subset of the snapping
 * boundaries, not all of them, so a snapped edge can still land between two
 * drawn lines. Closing that gap would mean deriving the snap unit from the
 * viewport, which would make the same drag select a different range at a
 * different zoom and break shared links.
 */
export function latticeStride(cell: number, count: number, viewScale: number): number {
  let stride = 1;
  while (stride < count && cell * stride * viewScale < MIN_LATTICE_PX) stride *= 2;
  return stride;
}

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
/** @internal Pure geometry seam used by drawing and direct renderer tests. */
export function regionRects(
  region: Region,
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom,
  viewScale = 1
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
    const w = Math.min(Math.max(MIN_MARK_PX / viewScale, x1 - x), canvasW);
    const h = Math.min(Math.max(MIN_MARK_PX / viewScale, y1 - y), canvasH);
    rects.push({
      x: Math.min(x, canvasW - w),
      y: Math.min(y, canvasH - h),
      w,
      h,
      alpha,
    });
  }
  return rects;
}

/** Project a set to coverage before alpha compositing. Disjoint N-D boxes can
 * overlap in the visible plane: their hidden volumes add, their opacities do not.
 * Sweep rectangle boundaries rather than enumerating tensor elements. */
export function regionFillRects(
  region: Region, shape: number[], cfg: ViewCfg, geom: GridGeom, viewScale = 1
): RegionRect[] {
  const rects = regionRects(disjointify(region), shape, cfg, geom, viewScale);
  if (!cfg.projection || shape.length <= 2 || rects.length < 2) return rects;
  const ys = [...new Set(rects.flatMap((q) => [q.y, q.y + q.h]))].sort((a, b) => a - b);
  const out: RegionRect[] = [];
  let previous = new Map<string, RegionRect>();
  for (let row = 0; row + 1 < ys.length; row++) {
    const y = ys[row], h = ys[row + 1] - y;
    const events = new Map<number, number>();
    for (const q of rects) {
      if (q.y > y || q.y + q.h <= y) continue;
      events.set(q.x, (events.get(q.x) ?? 0) + q.alpha);
      events.set(q.x + q.w, (events.get(q.x + q.w) ?? 0) - q.alpha);
    }
    const xs = [...events.keys()].sort((a, b) => a - b);
    const spans: RegionRect[] = [];
    let coverage = 0;
    for (let col = 0; col + 1 < xs.length; col++) {
      coverage += events.get(xs[col])!;
      if (coverage <= 1e-12) continue;
      const x = xs[col], w = xs[col + 1] - x;
      // Minimum-width screen marks can overlap beyond their true geometry.
      const alpha = Math.min(1, coverage);
      const last = spans[spans.length - 1];
      if (last && last.x + last.w === x && Math.abs(last.alpha - alpha) < 1e-12)
        last.w += w;
      else spans.push({ x, y, w, h, alpha });
    }
    const next = new Map<string, RegionRect>();
    for (const q of spans) {
      const key = `${q.x}:${q.w}:${q.alpha}`;
      const above = previous.get(key);
      if (above && above.y + above.h === y) {
        above.h += h;
        next.set(key, above);
      } else {
        out.push(q);
        next.set(key, q);
      }
    }
    previous = next;
  }
  return out;
}

/** The square display tile a tensor renders at without a tile of its own. */
export function tileOf(
  shape: number[],
  tileScale: number,
  px: number,
  cfg?: Pick<ViewCfg, "axes">
): number {
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  return tileFor(rows, cols, tileScale, cardScaleFor(shape, cfg, px));
}

/**
 * How far one arrow-key nudge moves the selection along `axis`.
 *
 * It is whatever unit the pointer works in: the tensor's tile on that axis
 * while snapping, a single element when not. Stepping by a tile with snapping
 * off would let the keyboard place a box at offsets a drag cannot reach.
 */
export function nudgeUnit(
  shape: number[],
  cfg: ViewCfg | undefined,
  tileScale: number,
  px: number,
  snapToGrid: boolean,
  axis: number
): number {
  if (!snapToGrid) return 1;
  return gestureTile(shape, cfg, tileOf(shape, tileScale, px, cfg))[axis] ?? 1;
}

/** Delta for one arrow press. An off-lattice selection first lands an edge on
 * the current lattice in the requested direction; once aligned, arrows advance
 * by whole tiles. This preserves the box's exact extent while making a region
 * drawn under an older/finer grid recoverable with the keyboard. */
export function nudgeDelta(
  interval: Interval,
  sign: -1 | 1,
  unit: number,
  snapToGrid: boolean,
  multiplier = 1
): number {
  if (!snapToGrid || unit <= 1) return sign * unit * multiplier;
  const remainder = ((interval.lo % unit) + unit) % unit;
  if (remainder !== 0)
    return sign > 0 ? unit - remainder : -remainder;
  return sign * unit * multiplier;
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

/** Fine paint buckets limit redraws; rounding down preserves screen-space visibility floors. */
export function paintScale(scale: number): number {
  return 2 ** (Math.floor(Math.log2(scale) * 32) / 32);
}
