/**
 * Canvas grid rendering for tensor cards. Pure drawing + geometry; no React.
 *
 * One drawn cell = one tile. A cell's fill alpha is the *coverage* of that tile
 * by the region: the fraction of the tile's elements that are in it, including
 * the hidden-axis fraction. So a partially-covered tile reads as partially
 * filled rather than being rounded to all-or-nothing, and the picture stays
 * honest at every zoom level instead of degrading into sub-pixel noise.
 */

import { Box, Interval, Region, disjointify } from "../core/region";
import { CARD_SURFACE } from "./palette";
import { cardPx, planeExtents, tileFor } from "./tiling";
import { viewAxes, type ViewCfg } from "./tensor-view";

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

/** @internal Corner marks sit outside the fill, at fixed screen size. Canvas
 * clipping keeps edge markers out of neighbouring cards and connectors. */
export function seedCornerSegments(rect: Pick<RegionRect, "x" | "y" | "w" | "h">, scale: number): Segment[] {
  const gap = 2 / scale, length = 3 / scale;
  return [-1, 1].flatMap((sx) => [-1, 1].flatMap((sy) => {
    const x = rect.x + (sx > 0 ? rect.w : 0) + sx * gap;
    const y = rect.y + (sy > 0 ? rect.h : 0) + sy * gap;
    return [
      { x1: x, y1: y, x2: x - sx * length, y2: y },
      { x1: x, y1: y, x2: x, y2: y - sy * length },
    ];
  }));
}

/** Conservative cross-browser ceiling on a canvas backing-store side. */
const MAX_CANVAS_DIM = 8192;

/** Weight of the hairline that delimits a ruled region, in screen CSS px. */
const PATTERN_EDGE_PX = 0.75;
/** Dash for the degraded stipple's delimiter: dots, like the fill it replaces. */
const STIPPLE_EDGE_DASH_PX = [1, 1.6] as const;

/** Ink coverage at the ends of the density scale. The floor stays clearly a set
 * of separate lines; the ceiling stays clearly ruled rather than solid, so a
 * complete contribution is representable without spending the top of the scale
 * to keep the two directions apart. */
const STRIPE_COVERAGE_MIN = 0.11;
const STRIPE_COVERAGE_MAX = 0.34;
/** Ruling weight in screen CSS px, held constant: spacing carries the quantity. */
const STRIPE_WIDTH_PX = 1;
/** Dot radius for the entanglement stipple, in screen pixels. */
const STIPPLE_RADIUS_PX = 0.9;

/** Spacing between stipple dots. Sparser than a ruling of the same density:
 * dots cover far less of a rect than lines at equal pitch, so matching the
 * pitch would read as a much lighter mark rather than a different one. */
export function stipplePitchPx(density: number): number {
  const coverage = Math.max(0.05, Math.min(1, density));
  return (STIPPLE_RADIUS_PX * 2) / coverage;
}

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
  viewScale = 1
): boolean {
  const minimum = Math.max(3, lineWidth * 2 + 1);
  return Math.min(rect.w, rect.h) * viewScale >= minimum;
}

/** A pattern cannot represent a region narrower than one useful pattern mark. */
export const MIN_PATTERN_EXTENT_PX = 3;

/** @internal Pure fallback rule for renderer tests. */
/**
 * Screen-px extent a mark needs on its short axis before it is legible.
 *
 * Every pattern kind answers here, so the gate below and the renderer that
 * draws the mark cannot disagree about what fits. They did: the gate asked for
 * 3px while `strokeStipple` refused anything under a full pitch of 3.6, so the
 * renderer regularly passed a rect the stipple then declined - and the fall
 * through was a solid fill, which is the *needs* mark. Three textures collapsed
 * to two, silently, in the same hue.
 *
 * `undefined` is the approximation hatch, which rides a ruling's geometry.
 */
export function patternFloorFor(pattern: Layer["pattern"] | undefined): number {
  // A stipple must fit one whole pitch, or its lattice has no row to sit on.
  if (pattern?.kind === "stipple") return stipplePitchPx(pattern.density);
  return MIN_PATTERN_EXTENT_PX;
}

export function patternFitsRect(
  rect: Pick<RegionRect, "w" | "h">,
  viewScale = 1,
  pattern?: Layer["pattern"]
): boolean {
  // Pitch and stroke weight are counter-scaled below, so zoom alone cannot
  // destroy direction. Only the region's resulting screen extent may force a
  // fallback: density degrades before direction does.
  return Math.min(rect.w, rect.h) * viewScale >= patternFloorFor(pattern);
}

/**
 * Ruling angle per selection box, in degrees counter-clockwise from horizontal.
 *
 * Two boxes whose downstream cones overlap used to draw the same ruling at the
 * same phase, so the later one landed exactly on the earlier and the shared
 * area read as a single region. Hue cannot resolve that : the two hues are
 * painted over each other : so the angle has to. Given its own slope, an
 * overlap crosses itself and says "both of these reach here", which is a fact
 * the panel otherwise only states as two separate rows.
 *
 * The twelve slots match the per-box attribution cap. Every angle stays clear
 * of 0/90 and the 45-degree approximation hatch. The order maximises separation
 * for the common first few boxes; later neutral boxes use the remaining safe
 * slopes, so no two attributable boxes can erase each other exactly.
 */
const STRIPE_ANGLES_DEG = [
  135, 165, 15, 75, 105, 111, 117, 123, 129, 147, 153, 159,
];

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
/** A broken stroke makes approximation a different texture kind from the solid
 * downstream rulings it may cross. Values are screen CSS px. */
const HATCH_DASH_PX = [2.4, 2.4] as const;

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

/** Stroke one ruled rect. Sizes are divided by the fine paint scale, so CSS
 * transforms do not change the screen-space meaning of spacing or weight. */
/**
 * A grid of dots inside the rect, used for entanglement.
 *
 * Returns false when the rect cannot hold a legible pattern, so the caller
 * degrades exactly as it does for a ruling that will not fit rather than
 * drawing something that reads as a different encoding.
 */
function strokeStipple(
  ctx: CanvasRenderingContext2D,
  rect: RegionRect,
  color: [number, number, number] | string,
  spec: { pitch: number; radius: number; alpha: number },
  viewScale: number
): boolean {
  const pitch = spec.pitch / viewScale;
  const radius = spec.radius / viewScale;
  if (pitch <= 0 || rect.w < pitch || rect.h < pitch) return false;
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.globalAlpha = spec.alpha;
  ctx.fillStyle =
    typeof color === "string" ? color : `rgb(${color[0]},${color[1]},${color[2]})`;
  ctx.beginPath();
  // Offset by half a pitch so the lattice sits inside the rect rather than
  // clipping a row of half-dots along its top and left edges.
  for (let y = rect.y + pitch / 2; y < rect.y + rect.h; y += pitch)
    for (let x = rect.x + pitch / 2; x < rect.x + rect.w; x += pitch) {
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
    }
  ctx.fill();
  ctx.restore();
  return true;
}

function strokeRuling(
  ctx: CanvasRenderingContext2D,
  rect: RegionRect,
  color: [number, number, number] | string,
  spec: {
    angle: number;
    pitch: number;
    width: number;
    alpha: number;
    dash?: readonly number[];
  },
  viewScale: number
): boolean {
  const segments = rulingSegments(rect, spec.angle, spec.pitch / viewScale);
  if (!segments.length) return false;
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.globalAlpha = spec.alpha;
  ctx.strokeStyle = typeof color === "string" ? color : `rgb(${color[0]},${color[1]},${color[2]})`;
  ctx.lineWidth = spec.width / viewScale;
  ctx.setLineDash(spec.dash?.map((length) => length / viewScale) ?? []);
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
  viewScale = 1
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
  const width = Math.ceil(geom.canvasW * res);
  const height = Math.ceil(geom.canvasH * res);
  // Reuse the backing store when only the paint changes.
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.setTransform(res, 0, 0, res, 0, 0);
  ctx.clearRect(0, 0, geom.canvasW, geom.canvasH);

  ctx.fillStyle = dark ? CARD_SURFACE.dark : CARD_SURFACE.light;
  ctx.fillRect(0, 0, geom.canvasW, geom.canvasH);

  const { tileRows, tileCols } = geom;

  for (const layer of layers) {
    const [r, g, b] = layer.color;
    // Fill is painted from the disjoint form. A region's boxes may overlap, and
    // each rect is composited at the layer's alpha, so a shared element would
    // otherwise be painted twice and read darker than its neighbours - which
    // would make alpha mean multiplicity by accident, on the one channel this
    // renderer deliberately holds constant. Splitting here is invisible: the
    // painted set is identical, and no user-facing box count comes from it.
    const rects = regionRects(disjointify(layer.region), shape, cfg, geom, viewScale);
    // Perimeter marks take the region's own boxes instead. A border says "this
    // is the region", and the region is the row band and the column band - two
    // rectangles that cross. Tracing the split form instead outlines three
    // shapes, one of which nothing reads, which is the artifact the stored form
    // exists to avoid. Overlapping borders are the intended reading: they cross,
    // one over the other. Fills cannot do this, because they composite.
    const boundRects = regionRects(layer.region, shape, cfg, geom, viewScale);

    let anyRuled = false;
    let stippleDegraded = false;
    for (const q of rects) {
      const alpha = Math.min(1, layer.alpha * q.alpha);
      const stippled =
        layer.pattern?.kind === "stipple" &&
        patternFitsRect(q, viewScale, layer.pattern) &&
        strokeStipple(
          ctx,
          q,
          layer.knockout ? (dark ? CARD_SURFACE.dark : CARD_SURFACE.light) : [r, g, b],
          {
            pitch: stipplePitchPx(layer.pattern.density),
            radius: STIPPLE_RADIUS_PX,
            alpha,
          },
          viewScale
        );
      if (stippled) continue;
      if (layer.pattern?.kind === "stipple") {
        // A stipple that will not fit degrades to a delimiter, never to a
        // solid. A solid in this hue *is* the needs mark, so borrowing it
        // would answer "combined with" in the encoding for "reads" - and
        // entanglement's answer is characteristically a thin band, so this is
        // the common case at the fitted overview rather than an edge one.
        stippleDegraded = true;
        continue;
      }
      const ruled =
        layer.pattern?.kind === "stripe" &&
        patternFitsRect(q, viewScale, layer.pattern) &&
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
          viewScale
        );
      if (ruled) {
        anyRuled = true;
        continue;
      }
      // Any failed requested pattern is the same degraded encoding, whether its
      // cause is a thin region or the renderer's line-count safety cap. Alpha is
      // otherwise reserved for hidden-axis coverage; the fallback borrows it
      // because no geometric direction channel survives in a sub-3px mark.
      const fallbackAlpha = layer.pattern ? alpha * 0.55 : alpha;
      ctx.fillStyle = `rgba(${r},${g},${b},${fallbackAlpha})`;
      ctx.fillRect(q.x, q.y, q.w, q.h);
    }

    // The degraded stipple's own mark: a dotted hairline around the box. Dotted
    // because the relation is a stipple, so the edge is made of the same dots
    // the fill would have been; a hairline because it is a delimiter and must
    // not read as the selection's perimeter.
    if (stippleDegraded)
      for (const q of boundRects) {
        if (!outlineFitsRect(q, PATTERN_EDGE_PX, viewScale)) continue;
        ctx.save();
        ctx.globalAlpha = Math.min(1, layer.alpha * q.alpha);
        ctx.strokeStyle = layer.knockout
          ? dark
            ? CARD_SURFACE.dark
            : CARD_SURFACE.light
          : `rgb(${r},${g},${b})`;
        ctx.lineWidth = PATTERN_EDGE_PX / viewScale;
        ctx.setLineDash(STIPPLE_EDGE_DASH_PX.map((d) => d / viewScale));
        const inset = ctx.lineWidth / 2;
        ctx.strokeRect(q.x + inset, q.y + inset, q.w - ctx.lineWidth, q.h - ctx.lineWidth);
        ctx.restore();
      }

    // A ruled fill has no edge of its own: the eye stops at the last line inside
    // the region, not at its bound, so the extent reads as ragged. A hairline
    // restores the shape at a fraction of the weight of the perimeter that used
    // to carry direction : thin enough that it stays a delimiter and cannot be
    // read as an encoding of its own. Solid fills are already crisp and get
    // none, and neither does a requested ruling that degraded to one - so this
    // waits on a ruling having actually been stroked, not merely asked for. It
    // bounds each stored box, so a region of two crossing bands is delimited as
    // two bands.
    if (anyRuled)
      for (const q of boundRects) {
        if (!patternFitsRect(q, viewScale)) continue;
        if (!outlineFitsRect(q, PATTERN_EDGE_PX, viewScale)) continue;
        ctx.save();
        ctx.globalAlpha = Math.min(1, Math.min(1, layer.alpha * q.alpha) + 0.12);
        ctx.strokeStyle = `rgb(${r},${g},${b})`;
        ctx.lineWidth = PATTERN_EDGE_PX / viewScale;
        const inset = ctx.lineWidth / 2;
        ctx.strokeRect(q.x + inset, q.y + inset, q.w - ctx.lineWidth, q.h - ctx.lineWidth);
        ctx.restore();
      }

    // Over-approximation rides the same routine at its own reserved slope, so
    // it composes with a downstream ruling as a crossing rather than as a
    // second texture that has to be told apart from the first. On solid fills,
    // surface-coloured dashes give contrast without erasing underlying layers.
    if (layer.hatch)
      for (const q of rects)
        if (patternFitsRect(q, viewScale))
          strokeRuling(
            ctx,
            q,
            layer.pattern ? [r, g, b] : (dark ? CARD_SURFACE.dark : CARD_SURFACE.light),
            {
              angle: HATCH_ANGLE_DEG,
              pitch: HATCH_PITCH_PX,
              width: HATCH_WIDTH_PX,
              alpha: Math.min(1, layer.alpha * q.alpha) * 0.9,
              dash: HATCH_DASH_PX,
            },
            viewScale
          );

    if (layer.outline) {
      ctx.strokeStyle = `rgba(${r},${g},${b},0.95)`;
      const screenLineWidth = layer.lineWidth ?? 1.5;
      ctx.lineWidth = screenLineWidth / viewScale;
      for (const q of boundRects) {
        if (!outlineFitsRect(q, screenLineWidth, viewScale)) continue;
        const inset = ctx.lineWidth / 2;
        ctx.strokeRect(q.x + inset, q.y + inset, q.w - ctx.lineWidth, q.h - ctx.lineWidth);
      }
    }
  }

  // tile boundaries : the cell grid *is* the tile grid, drawn at a stride that
  // keeps the lines apart on screen. Each axis strides on its own cell size, so
  // a card with wide cells and short ones keeps the boundaries it can show.
  const strideC = latticeStride(geom.cellW, tileCols, viewScale);
  const strideR = latticeStride(geom.cellH, tileRows, viewScale);
  if (strideC < tileCols || strideR < tileRows) {
    ctx.strokeStyle = dark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.10)";
    ctx.lineWidth = 1 / viewScale;
    ctx.beginPath();
    for (let c = strideC; c < tileCols; c += strideC) {
      const x = c * geom.cellW;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, geom.canvasH);
    }
    for (let r = strideR; r < tileRows; r += strideR) {
      const y = r * geom.cellH;
      ctx.moveTo(0, y);
      ctx.lineTo(geom.canvasW, y);
    }
    ctx.stroke();
  }

  ctx.strokeStyle = dark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.18)";
  ctx.lineWidth = Math.min(1 / viewScale, geom.canvasW, geom.canvasH);
  const inset = ctx.lineWidth / 2;
  ctx.strokeRect(inset, inset, geom.canvasW - ctx.lineWidth, geom.canvasH - ctx.lineWidth);

  // Seed marks are the final annotation, so later cone fills and the lattice
  // cannot cover them. They spend geometry, not a second fill treatment.
  for (const layer of layers) {
    if (!layer.seed) continue;
    // Seeds keep their own boxes: a corner mark identifies a part the user drew,
    // so it must sit at that part's bounds and not at a fragment of it.
    for (const rect of regionRects(layer.region, shape, cfg, geom, viewScale)) {
      ctx.save();
      ctx.globalAlpha = layer.alpha * rect.alpha;
      ctx.beginPath();
      for (const segment of seedCornerSegments(rect, viewScale)) {
        ctx.moveTo(segment.x1, segment.y1);
        ctx.lineTo(segment.x2, segment.y2);
      }
      ctx.strokeStyle = dark ? CARD_SURFACE.dark : CARD_SURFACE.light;
      ctx.lineWidth = 3 / viewScale;
      ctx.stroke();
      ctx.strokeStyle = `rgb(${layer.color.join(",")})`;
      ctx.lineWidth = 1 / viewScale;
      ctx.stroke();
      ctx.restore();
    }
  }
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

/** Fine paint buckets limit redraws; rounding down preserves screen-space visibility floors. */
export function paintScale(scale: number): number {
  return 2 ** (Math.floor(Math.log2(scale) * 32) / 32);
}
