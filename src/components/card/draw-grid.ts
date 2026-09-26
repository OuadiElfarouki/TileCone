/**
 * Canvas drawing for tensor cards: fills, rulings, stipples, hatches, the
 * lattice, outlines and seed marks. Consumes the pure geometry in
 * `view/tensor/grid.ts` and decides nothing about which elements are shown.
 */

import { CARD_SURFACE } from "../../view/palette";
import {
  latticeStride,
  regionFillRects,
  regionRects,
  type GridGeom,
  type Layer,
  type PlanPaint,
  type RegionRect,
} from "../../view/tensor/grid";
import type { ViewCfg } from "../../view/tensor/tensor-view";

/** Width of a needed producer tile's outline, in screen px. */
const PLAN_TILE_LINE_PX = 1.5;
/** Dash pattern for a producer the task only possibly needs, in screen px. */
const PLAN_TILE_DASH_PX = [3, 2.5] as const;

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
  viewScale = 1,
  plan?: PlanPaint
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
    const rects = regionFillRects(layer.region, shape, cfg, geom, viewScale);
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
  // a card with wide cells and short ones keeps the boundaries it can show. In
  // the Plan view the grid is the plan's, whose cells need not be square.
  const lattice = plan
    ? plan.lattice && {
        cellW: (plan.lattice.cols / geom.cols) * geom.canvasW,
        cellH: (plan.lattice.rows / geom.rows) * geom.canvasH,
        count: {
          cols: Math.ceil(geom.cols / plan.lattice.cols),
          rows: Math.ceil(geom.rows / plan.lattice.rows),
        },
      }
    : { cellW: geom.cellW, cellH: geom.cellH, count: { cols: tileCols, rows: tileRows } };
  if (lattice) {
    const strideC = latticeStride(lattice.cellW, lattice.count.cols, viewScale);
    const strideR = latticeStride(lattice.cellH, lattice.count.rows, viewScale);
    if (strideC < lattice.count.cols || strideR < lattice.count.rows) {
      ctx.strokeStyle = plan?.lattice?.proposed
        ? dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"
        : dark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.10)";
      ctx.lineWidth = 1 / viewScale;
      ctx.beginPath();
      for (let c = strideC; c < lattice.count.cols; c += strideC) {
        const x = c * lattice.cellW;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, geom.canvasH);
      }
      for (let r = strideR; r < lattice.count.rows; r += strideR) {
        const y = r * lattice.cellH;
        ctx.moveTo(0, y);
        ctx.lineTo(geom.canvasW, y);
      }
      ctx.stroke();
    }
  }

  // Needed producer tiles: neutral ink, because hue on a card identifies a
  // selected tile and these are not selected. Dashed when the need is only
  // possible; the demand filled inside them carries the approximation hatch.
  if (plan)
    for (const { box, definite } of plan.tiles)
      for (const q of regionRects({ boxes: [box], exact: true, reasons: [] }, shape, cfg, geom, viewScale)) {
        if (!outlineFitsRect(q, PLAN_TILE_LINE_PX, viewScale)) continue;
        ctx.save();
        ctx.strokeStyle = dark ? "rgba(255,255,255,0.78)" : "rgba(0,0,0,0.72)";
        ctx.lineWidth = PLAN_TILE_LINE_PX / viewScale;
        ctx.setLineDash(definite ? [] : PLAN_TILE_DASH_PX.map((d) => d / viewScale));
        const inset = ctx.lineWidth / 2;
        ctx.strokeRect(q.x + inset, q.y + inset, q.w - ctx.lineWidth, q.h - ctx.lineWidth);
        ctx.restore();
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
