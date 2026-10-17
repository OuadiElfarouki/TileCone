/**
 * Tile sizing. The display tile is the lattice a card is ruled with and the
 * unit a snapped gesture and an arrow step take; analysis stays
 * element-precise whatever it is.
 *
 * One rule, per axis: about `20 / 2^tileScale` tiles across the axis, the tile
 * rounded to a power of two and capped at half the axis, so even the coarsest
 * setting leaves two tiles. It depends on the axis's extent and the setting
 * and nothing else. Two consequences are the point of it:
 *
 *   - an axis of one length is gridded alike on every card. `seq = 128` takes
 *     the same tile on Q and on the scores, so a snapped click selects the same
 *     rows on both. A square tile from each tensor's smaller axis gave `seq` 8
 *     on one card and 2 on the next.
 *   - nothing else in the graph moves it. The earlier rule coarsened tiles
 *     until a cell filled 5 canvas px, which tied every tile to the graph's
 *     scale and so to its largest tensor: editing `K` re-gridded tensors that
 *     have no `K`.
 *
 * Density is the renderer's concern instead: `grid.latticeStride` draws every
 * 2nd, 4th... boundary when cells are too small on screen, and the hover
 * outline shows the tile a click takes.
 *
 * Card size is a separate axis from tile size, and it is a property of the
 * *graph*, not of the tensor: `graphScale` picks one px-per-element for every
 * card, so a dimension two tensors share is drawn at the same physical length in
 * both. That is what makes a matmul's contraction dimension readable as one
 * axis. Sizing each card to its own budget instead (the obvious thing) renders
 * A[M,K] and B[K,N] with two different lengths for K, and the user reads
 * "different granularity" where the truth is "same axis".
 */

/** @internal Policy details are exported so tiling invariants can be tested directly. */
/** The default tile as a fraction of its axis: about 20 tiles across. */
export const AUTO_TILE_FRACTION = 0.05;
export const TILE_SCALE_MIN = -5;
export const TILE_SCALE_MAX = 5;
/** The sentinel meaning no aggregation: one element per logical tile. */
export const TILE_SCALE_NONE = TILE_SCALE_MIN;

/** Px budget for the widest / tallest tensor in the graph. Soft: the floors
 * below may override them, because a tensor nobody can see is worse than a
 * tensor that needs a pan. */
export const MAX_GRAPH_W = 300;
export const MAX_GRAPH_H = 430;
/** The smallest tensor side must stay at least this tall/wide. */
export const MIN_SIDE_PX = 14;
/** Cap on pixels per element, so a tiny graph does not blow up to fill the view. */
export const MAX_ELEM_PX = 15;
/** Absolute floor on px per element, so a huge graph still draws something. */
const MIN_ELEM_PX = 0.18;
/** A degenerate axis (extent 1) is drawn at a fixed size: it has no length to
 * scale, and letting it vote on the global scale would peg `minD` at 1. */
export const DEGENERATE_SIDE_PX = 14;

/** Nearest power of two (in log space), at least 1. */
export function pow2Round(x: number): number {
  if (!(x > 0)) return 1;
  return Math.max(1, 2 ** Math.round(Math.log2(x)));
}

/** Largest power of two <= x, at least 1. */
export function pow2Floor(x: number): number {
  if (!(x >= 1)) return 1;
  return 2 ** Math.floor(Math.log2(x));
}

/** The extents actually drawn: rows and columns of the row-major 2-D view. */
export function planeExtents(
  shape: number[],
  rowAxis: number,
  colAxis: number
): { rows: number; cols: number } {
  return {
    rows: rowAxis >= 0 ? shape[rowAxis] : 1,
    cols: colAxis >= 0 ? shape[colAxis] : 1,
  };
}

/**
 * The display tile on an axis of `extent`: about `20 / 2^tileScale` tiles
 * across it, a power of two, at most half the axis. One element under None,
 * and on an axis of one element.
 */
export function axisTile(extent: number, tileScale: number): number {
  if (tileScale === TILE_SCALE_NONE || extent <= 1) return 1;
  const wanted = pow2Round(AUTO_TILE_FRACTION * extent * 2 ** tileScale);
  return Math.max(1, Math.min(wanted, pow2Floor(extent / 2)));
}

/** The display tile on every axis of `shape`; consumers take the axes they draw. */
export function displayTile(shape: readonly number[], tileScale: number): number[] {
  return shape.map((extent) => axisTile(extent, tileScale));
}

/**
 * Px per element for the ENTIRE graph, so equal dimensions render at equal
 * lengths wherever they appear. O(#tensors), computed once per resolved graph.
 *
 * The largest tensor sets the budget; the smallest *non-degenerate* side then
 * raises the scale if the budget would make it invisible. Degenerate axes are
 * excluded deliberately: they are drawn at a fixed `DEGENERATE_SIDE_PX` and have
 * no length to preserve, so counting them would peg `minD` at 1 and force the
 * whole graph to the `MAX_ELEM_PX` cap, one bias vector would blow every card
 * up by an order of magnitude.
 */
export function graphScale(planes: { rows: number; cols: number }[]): number {
  let maxR = 1;
  let maxC = 1;
  let minD = Infinity;
  for (const { rows, cols } of planes) {
    maxR = Math.max(maxR, rows);
    maxC = Math.max(maxC, cols);
    if (rows > 1) minD = Math.min(minD, rows);
    if (cols > 1) minD = Math.min(minD, cols);
  }
  const fit = Math.min(MAX_GRAPH_W / maxC, MAX_GRAPH_H / maxR);
  // no non-degenerate axis anywhere: nothing to keep visible, so the budget wins
  const need = Number.isFinite(minD) ? MIN_SIDE_PX / minD : 0;
  return Math.min(MAX_ELEM_PX, Math.max(fit, need, MIN_ELEM_PX));
}

/**
 * On-screen size of a tensor's grid, in CSS pixels, at the graph's scale `px`.
 * Depends **only on the tensor's shape and `px`** (never on the tile size) so
 * changing detail re-lattices the card in place instead of resizing it. Aspect
 * ratio follows the tensor.
 */
export function cardPx(rows: number, cols: number, px: number): { w: number; h: number } {
  return {
    w: cols === 1 ? DEGENERATE_SIDE_PX : Math.max(3, Math.round(cols * px)),
    h: rows === 1 ? DEGENERATE_SIDE_PX : Math.max(3, Math.round(rows * px)),
  };
}

/** Every drawn axis length in the graph, once: what a tiling is decided over. */
function drawnExtents(planes: { rows: number; cols: number }[]): number[] {
  return [...new Set(planes.flatMap(({ rows, cols }) => [rows, cols]).filter((e) => e > 1))].sort((a, b) => a - b);
}

function tileState(extents: number[], scale: number): number[] {
  return extents.map((extent) => axisTile(extent, scale));
}

function sameTileState(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((tile, index) => tile === b[index]);
}

/**
 * Slider stops that produce distinct tilings of this graph's axes, plus the
 * semantically distinct None sentinel.
 *
 * `tileScale` remains the serialized power-of-two shift; this only removes UI
 * positions whose tiles are identical to a neighbour's on every axis the graph
 * draws - a small graph clamps the finest and coarsest settings together. For
 * a plateau, the representative closest to the default (zero) is kept, so old
 * share links display at the equivalent live stop.
 */
export function effectiveTileScaleStops(planes: { rows: number; cols: number }[]): number[] {
  const extents = drawnExtents(planes);
  const runs: { lo: number; hi: number }[] = [];
  let runLo = TILE_SCALE_MIN;
  let previous = tileState(extents, TILE_SCALE_MIN);
  for (let scale = TILE_SCALE_MIN + 1; scale <= TILE_SCALE_MAX; scale++) {
    const current = tileState(extents, scale);
    if (!sameTileState(previous, current)) {
      runs.push({ lo: runLo, hi: scale - 1 });
      runLo = scale;
    }
    previous = current;
  }
  runs.push({ lo: runLo, hi: TILE_SCALE_MAX });
  const stops = runs.map(({ lo, hi }) => Math.max(lo, Math.min(hi, 0)));

  // "none" (one logical tile per element) is always the leftmost stop and is
  // distinct from fitted scale values even when its boundaries are too dense
  // to draw individually.
  return stops[0] === TILE_SCALE_MIN ? stops : [TILE_SCALE_MIN, ...stops];
}

/**
 * What a setting asks for, the same on every tensor and in every graph: about
 * how many tiles cross each axis. The tile is rounded to a power of two, so an
 * axis lands near the number rather than on it, and never below two.
 */
export function tileScaleLabel(scale: number): string {
  if (scale === TILE_SCALE_NONE) return "elements";
  // Whole numbers down to 5; below that the half-axis cap holds an axis at
  // two tiles, so 2.5 would overstate it.
  return `≈${Math.max(2, Math.floor(1 / (AUTO_TILE_FRACTION * 2 ** scale)))} per axis`;
}

/** Find which effective stop tiles this graph's axes as `scale` does. */
export function effectiveTileScaleIndex(
  planes: { rows: number; cols: number }[],
  stops: number[],
  scale: number
): number {
  // Intent wins over coincident geometry. On a small graph None and the
  // finest settings can both tile by one element, but the control must still
  // show the setting the user chose.
  const exact = stops.indexOf(scale);
  if (exact >= 0) return exact;
  const extents = drawnExtents(planes);
  const wanted = tileState(extents, scale);
  const found = stops.findIndex((stop) => sameTileState(tileState(extents, stop), wanted));
  if (found >= 0) return found;
  // Defensive fallback for a future non-monotone tiling policy.
  let best = 0;
  for (let i = 1; i < stops.length; i++)
    if (Math.abs(stops[i] - scale) < Math.abs(stops[best] - scale)) best = i;
  return best;
}
