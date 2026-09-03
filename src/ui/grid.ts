/**
 * Canvas grid rendering for tensor cards. Pure drawing + geometry; no React.
 *
 * One drawn cell = one tile. A cell's fill alpha is the *coverage* of that tile
 * by the region: the fraction of the tile's elements that are in it, including
 * the hidden-axis fraction. So a partially-covered tile reads as partially
 * filled rather than being rounded to all-or-nothing, and the picture stays
 * honest at every zoom level instead of degrading into sub-pixel noise.
 */

import { Box, Interval, Region } from "../core/region";
import { CARD_SURFACE } from "./palette";
import { cardPx, MIN_CELL_PX, planeExtents, tileFor } from "./tiling";
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
   * Direction is fill geometry, not a perimeter: hue belongs to tile identity,
   * solid fill means "needs", and a diagonal ruling means "feeds". `density`
   * sets the spacing between rulings — further apart for a smaller share — and
   * stays explicit so supplied-share can own it later without changing the
   * layer contract. `angle` separates one box's ruling from another's, so two
   * cones that reach the same elements cross there instead of hiding.
   */
  pattern?: { kind: "stripe"; density: number; angle: number };
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

type RegionRect = { x: number; y: number; w: number; h: number; alpha: number };

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

/** Weight of the hairline that delimits a ruled region, in screen CSS px. */
const PATTERN_EDGE_PX = 0.75;

/** Ink coverage at the ends of the density scale. The floor stays clearly a set
 * of separate lines; the ceiling stays clearly ruled rather than solid, so a
 * complete contribution is representable without spending the top of the scale
 * to keep the two directions apart. */
const STRIPE_COVERAGE_MIN = 0.11;
const STRIPE_COVERAGE_MAX = 0.34;
/** Ruling weight in screen CSS px, held constant: spacing carries the quantity. */
const STRIPE_WIDTH_PX = 1;

/**
 * Perpendicular spacing between downstream rulings, in screen CSS px.
 *
 * Density is read as *ink*, so coverage is what moves linearly and the spacing
 * is derived from it: a larger share closes the gap, a smaller one opens it.
 * The quantity therefore lives in a geometric property that survives greyscale,
 * thumbnails, and the hue already spent on tile identity.
 */
/** @internal Pure encoding rule for renderer tests. */
export function stripePitchPx(density: number): number {
  const bounded = Math.min(1, Math.max(0, density));
  const coverage = STRIPE_COVERAGE_MIN + bounded * (STRIPE_COVERAGE_MAX - STRIPE_COVERAGE_MIN);
  return STRIPE_WIDTH_PX / coverage;
}

/** A perimeter wider than its rectangle invents area. Drop it until both axes
 * have enough screen extent to contain the full stroke plus a visible centre. */
/** @internal Pure renderer rule for thin-region tests. */
export function outlineFitsRect(
  rect: Pick<RegionRect, "w" | "h">,
  lineWidth: number,
  paintScale = 1
): boolean {
  const minimum = Math.max(3, lineWidth * 2 + 1);
  return Math.min(rect.w, rect.h) * paintScale >= minimum;
}

/** Below this paint scale a ruling falls under a pixel and greys into a wash,
 * which reads as a smaller share rather than a smaller region. The caller
 * buckets zoom to powers of two, so this is the "anything under 1:1" bucket
 * rather than a tunable fraction; moving it has no effect until it crosses 0.5. */
export const MIN_PATTERN_SCALE = 1;
/** A pattern cannot represent a region narrower than one useful pattern mark. */
export const MIN_PATTERN_EXTENT_PX = 3;

/** @internal Pure fallback rule for renderer tests. */
export function patternFitsRect(
  rect: Pick<RegionRect, "w" | "h">,
  paintScale = 1
): boolean {
  return paintScale >= MIN_PATTERN_SCALE &&
    Math.min(rect.w, rect.h) * paintScale >= MIN_PATTERN_EXTENT_PX;
}

/**
 * Ruling angle per selection box, in degrees counter-clockwise from horizontal.
 *
 * Two boxes whose downstream cones overlap used to draw the same ruling at the
 * same phase, so the later one landed exactly on the earlier and the shared
 * area read as a single region. Hue cannot resolve that — the two hues are
 * painted over each other — so the angle has to. Given its own slope, an
 * overlap crosses itself and says "both of these reach here", which is a fact
 * the panel otherwise only states as two separate rows.
 *
 * Angles run 30 degrees apart, skipping the 45 the approximation hatch owns and
 * staying clear of 0/90, where a ruling would run parallel to the tile lattice.
 * There are more slots than the three categorical hues on purpose: every box
 * past the third shares one neutral colour, so past that point the angle is the
 * only thing still telling them apart.
 */
const STRIPE_ANGLES_DEG = [135, 165, 15, 75];

/** @internal Pure encoding rule for renderer tests. */
export function stripeAngleDeg(boxIndex: number): number {
  const n = STRIPE_ANGLES_DEG.length;
  return STRIPE_ANGLES_DEG[((Math.trunc(boxIndex) % n) + n) % n];
}

/** The slope reserved for over-approximation, kept out of the box rotation. */
export const HATCH_ANGLE_DEG = 45;
/** Perpendicular spacing of the approximation hatch, in screen CSS px. */
const HATCH_PITCH_PX = 8 / Math.SQRT2;
/** Weight of a hatch line, in screen CSS px. */
const HATCH_WIDTH_PX = 1.6;

/** Beyond this many lines a ruling is denser than the region is wide in pixels;
 * the caller paints solid instead of spending the frame on invisible strokes. */
const MAX_RULING_LINES = 4096;

type Segment = { x1: number; y1: number; x2: number; y2: number };

/**
 * A ruled fill as line segments in canvas pixels.
 *
 * The lines are stroked rather than tiled as a repeating bitmap. A bitmap tile
 * only repeats seamlessly at angles commensurate with its own edges, which is
 * what limited the fill to 45 degrees; stroking accepts any angle, is rasterised
 * at full resolution instead of resampled, and drops the pattern cache with it.
 *
 * Phase is anchored to the canvas origin, not to the rect, so every region of a
 * card lies on one continuous ruling: two regions at the same angle line up
 * instead of stepping, and two at different angles cross the same way wherever
 * they meet.
 *
 * Pure and DOM-free so spacing, angle and phase can be asserted directly.
 */
/** @internal Pure geometry seam for ruled fills. */
export function rulingSegments(
  rect: Pick<RegionRect, "x" | "y" | "w" | "h">,
  angleDeg: number,
  pitch: number
): Segment[] {
  if (!(pitch > 0)) return [];
  const theta = (angleDeg * Math.PI) / 180;
  // Canvas y grows downward, so a counter-clockwise visual angle negates it.
  const dx = Math.cos(theta);
  const dy = -Math.sin(theta);
  // Unit normal: the axis the lines are spaced along.
  const nx = -dy;
  const ny = dx;
  const { x, y, w, h } = rect;
  const corners = [
    x * nx + y * ny,
    (x + w) * nx + y * ny,
    x * nx + (y + h) * ny,
    (x + w) * nx + (y + h) * ny,
  ];
  const lo = Math.min(...corners);
  const hi = Math.max(...corners);
  if ((hi - lo) / pitch > MAX_RULING_LINES) return [];
  const cx = x + w / 2;
  const cy = y + h / 2;
  const centre = cx * nx + cy * ny;
  // Long enough to cross the rect from any offset; the caller clips.
  const reach = Math.hypot(w, h) / 2 + pitch;
  const out: Segment[] = [];
  for (let s = Math.ceil(lo / pitch) * pitch; s <= hi; s += pitch) {
    const ox = cx + (s - centre) * nx;
    const oy = cy + (s - centre) * ny;
    out.push({
      x1: ox - dx * reach,
      y1: oy - dy * reach,
      x2: ox + dx * reach,
      y2: oy + dy * reach,
    });
  }
  return out;
}

/** Stroke one ruled rect. Screen-CSS sizes are divided by the zoom bucket, so
 * spacing and weight stay constant on screen as the graph scales. */
function strokeRuling(
  ctx: CanvasRenderingContext2D,
  rect: RegionRect,
  color: [number, number, number],
  spec: { angle: number; pitch: number; width: number; alpha: number },
  paintScale: number
): boolean {
  const segments = rulingSegments(rect, spec.angle, spec.pitch / paintScale);
  if (!segments.length) return false;
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.globalAlpha = spec.alpha;
  ctx.strokeStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
  ctx.lineWidth = spec.width / paintScale;
  ctx.beginPath();
  for (const s of segments) {
    ctx.moveTo(s.x1, s.y1);
    ctx.lineTo(s.x2, s.y2);
  }
  ctx.stroke();
  ctx.restore();
  return true;
}

export function drawGrid(
  canvas: HTMLCanvasElement,
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom,
  layers: Layer[],
  dark: boolean,
  renderScale = 1,
  paintScale = 1
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

    for (const q of rects) {
      const alpha = Math.min(1, layer.alpha * q.alpha);
      const ruled =
        layer.pattern?.kind === "stripe" &&
        patternFitsRect(q, paintScale) &&
        strokeRuling(
          ctx,
          q,
          [r, g, b],
          {
            angle: layer.pattern.angle,
            pitch: stripePitchPx(layer.pattern.density),
            width: STRIPE_WIDTH_PX,
            alpha,
          },
          paintScale
        );
      if (ruled) {
        // A ruled fill has no edge of its own: the eye stops at the last line
        // inside the region, not at its bound, so the extent reads as ragged.
        // A hairline restores the shape at a fraction of the weight of the
        // perimeter that used to carry direction — thin enough that it stays a
        // delimiter and cannot be read as an encoding of its own. Solid fills
        // are already crisp, so only ruled rects get one.
        if (outlineFitsRect(q, PATTERN_EDGE_PX, paintScale)) {
          ctx.save();
          ctx.globalAlpha = Math.min(1, alpha + 0.12);
          ctx.strokeStyle = `rgb(${r},${g},${b})`;
          ctx.lineWidth = PATTERN_EDGE_PX / paintScale;
          const inset = ctx.lineWidth / 2;
          ctx.strokeRect(q.x + inset, q.y + inset, q.w - ctx.lineWidth, q.h - ctx.lineWidth);
          ctx.restore();
        }
        continue;
      }
      // Thin marks stay present, and fitted-out views avoid a ruling that would
      // collapse into a wash. Lower alpha keeps the low-zoom fallback distinct
      // from solid upstream.
      const fallbackAlpha = layer.pattern && paintScale < MIN_PATTERN_SCALE ? alpha * 0.55 : alpha;
      ctx.fillStyle = `rgba(${r},${g},${b},${fallbackAlpha})`;
      ctx.fillRect(q.x, q.y, q.w, q.h);
    }

    // Over-approximation rides the same routine at its own reserved slope, so
    // it composes with a downstream ruling as a crossing rather than as a
    // second texture that has to be told apart from the first.
    if (layer.hatch)
      for (const q of rects)
        strokeRuling(
          ctx,
          q,
          [r, g, b],
          {
            angle: HATCH_ANGLE_DEG,
            pitch: HATCH_PITCH_PX,
            width: HATCH_WIDTH_PX,
            alpha: Math.min(1, layer.alpha * q.alpha) * 0.9,
          },
          paintScale
        );

    if (layer.outline) {
      ctx.strokeStyle = `rgba(${r},${g},${b},0.95)`;
      ctx.lineWidth = layer.lineWidth ?? 1.5;
      for (const q of rects)
        if (outlineFitsRect(q, ctx.lineWidth, paintScale))
          ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.w - 1, q.h - 1);
    }
  }

  // tile boundaries — the cell grid *is* the tile grid
  if (Math.min(geom.cellW, geom.cellH) >= MIN_CELL_PX) {
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

/** Tile-cell range -> element interval on that axis, clamped to the extent. */
export function tileSpan(t0: number, t1: number, tile: number, extent: number): [number, number] {
  const lo = Math.max(0, Math.min(t0, t1) * tile);
  const hi = Math.min(extent, (Math.max(t0, t1) + 1) * tile);
  return [lo, hi];
}
