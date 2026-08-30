/**
 * Tile sizing. Tiles are the *rendering unit*: a card draws one cell per tile,
 * shaded by how much of the region that tile actually contains. So the tile size
 * sets both the analysis granularity and the display resolution, and there is one
 * global control for it rather than a per-card setting.
 *
 * Sizing rule, in order:
 *   1. auto  — 5% of the smallest non-degenerate visible axis, snapped to a power
 *              of two: (100×200) -> 4, (200×214) -> 8.
 *   2. scale — the global slider shifts that by powers of two (2^tileScale).
 *   3. clamp — to [1, largest power of two <= minAxis/2], so the coarsest setting
 *              still leaves at least a 2x2 tile grid on the smallest axis.
 *   4. fit   — raise the tile until the grid fits the card at >= MIN_CELL_PX per
 *              cell. This is the "minimum elementary tile" that keeps very large
 *              tensors crisp instead of collapsing them into sub-pixel mush.
 *
 * Card size is a separate axis from tile size, and it is a property of the
 * *graph*, not of the tensor: `graphScale` picks one px-per-element for every
 * card, so a dimension two tensors share is drawn at the same physical length in
 * both. That is what makes a matmul's contraction dimension readable as one
 * axis. Sizing each card to its own budget instead — the obvious thing — renders
 * A[M,K] and B[K,N] with two different lengths for K, and the user reads
 * "different granularity" where the truth is "same axis".
 */

export const AUTO_TILE_FRACTION = 0.05;
export const TILE_SCALE_MIN = -5;
export const TILE_SCALE_MAX = 5;

/** Rendering budget. */
export const MIN_CELL_PX = 3;
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
export const MIN_ELEM_PX = 0.18;
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

/** Smallest axis that actually has extent; a 1-D tensor uses its only axis. */
export function governingAxis(rows: number, cols: number): number {
  const dims = [rows, cols].filter((d) => d > 1);
  return dims.length ? Math.min(...dims) : 1;
}

export function autoTile(rows: number, cols: number): number {
  const minAxis = governingAxis(rows, cols);
  return clampTile(pow2Round(AUTO_TILE_FRACTION * minAxis), rows, cols);
}

export function maxTile(rows: number, cols: number): number {
  const minAxis = governingAxis(rows, cols);
  return Math.max(1, pow2Floor(minAxis / 2));
}

function clampTile(tile: number, rows: number, cols: number): number {
  return Math.max(1, Math.min(tile, maxTile(rows, cols)));
}

/**
 * Px per element for the ENTIRE graph, so equal dimensions render at equal
 * lengths wherever they appear. O(#tensors), computed once per resolved graph.
 *
 * The largest tensor sets the budget; the smallest *non-degenerate* side then
 * raises the scale if the budget would make it invisible. Degenerate axes are
 * excluded deliberately: they are drawn at a fixed `DEGENERATE_SIDE_PX` and have
 * no length to preserve, so counting them would peg `minD` at 1 and force the
 * whole graph to the `MAX_ELEM_PX` cap — one bias vector would blow every card
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
 * Depends **only on the tensor's shape and `px`** — never on the tile size — so
 * changing detail re-lattices the card in place instead of resizing it. Aspect
 * ratio follows the tensor.
 */
export function cardPx(rows: number, cols: number, px: number): { w: number; h: number } {
  return {
    w: cols === 1 ? DEGENERATE_SIDE_PX : Math.max(3, Math.round(cols * px)),
    h: rows === 1 ? DEGENERATE_SIDE_PX : Math.max(3, Math.round(rows * px)),
  };
}

/** The tile actually used for this tensor, given the global detail setting. */
export function tileFor(rows: number, cols: number, tileScale: number, px: number): number {
  const base = pow2Round(AUTO_TILE_FRACTION * governingAxis(rows, cols));
  let tile = clampTile(base * 2 ** tileScale, rows, cols);
  // fit: the card is a fixed size, so refuse tiles that would draw cells below
  // MIN_CELL_PX inside it. This is the minimum elementary tile.
  //
  // NOTE: clamping here is the common case, not the exception — a typical tensor
  // has ~5 distinct tiles across the slider's 11 stops, so roughly half the
  // travel is inert. That predates the graph scale (measured: mean 4.69 distinct
  // stops before, 5.02 after) and is a property of snapping tiles to powers of
  // two inside a bounded card. `isTileClamped` reports it but nothing in the UI
  // consumes that yet; see docs/UI_REFACTOR.md §1.10 before wiring it to a
  // warning, because a warning that is always on says nothing.
  const { w, h } = cardPx(rows, cols, px);
  const maxCols = Math.max(1, Math.floor(w / MIN_CELL_PX));
  const maxRows = Math.max(1, Math.floor(h / MIN_CELL_PX));
  while (Math.ceil(cols / tile) > maxCols || Math.ceil(rows / tile) > maxRows) tile *= 2;
  return tile;
}

/** True when `tileFor` had to override the requested scale to keep cells legible. */
export function isTileClamped(
  rows: number,
  cols: number,
  tileScale: number,
  px: number
): boolean {
  const base = pow2Round(AUTO_TILE_FRACTION * governingAxis(rows, cols));
  return tileFor(rows, cols, tileScale, px) !== base * 2 ** tileScale;
}

function tileState(planes: { rows: number; cols: number }[], scale: number, px: number): number[] {
  return planes.map(({ rows, cols }) => tileFor(rows, cols, scale, px));
}

function sameTileState(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((tile, index) => tile === b[index]);
}

/**
 * Slider stops that produce distinct graph-wide tile lattices.
 *
 * `tileScale` remains the serialized power-of-two shift; this only removes UI
 * positions whose complete per-tensor tile vector is identical to a neighbour.
 * For a plateau, the representative closest to auto (zero) is retained, so old
 * share links preserve their meaning and display at the equivalent live stop.
 */
export function effectiveTileScaleStops(
  planes: { rows: number; cols: number }[],
  px: number
): number[] {
  const runs: { lo: number; hi: number }[] = [];
  let runLo = TILE_SCALE_MIN;
  let previous = tileState(planes, TILE_SCALE_MIN, px);
  for (let scale = TILE_SCALE_MIN + 1; scale <= TILE_SCALE_MAX; scale++) {
    const current = tileState(planes, scale, px);
    if (!sameTileState(previous, current)) {
      runs.push({ lo: runLo, hi: scale - 1 });
      runLo = scale;
    }
    previous = current;
  }
  runs.push({ lo: runLo, hi: TILE_SCALE_MAX });
  const stops = runs.map(({ lo, hi }) => Math.max(lo, Math.min(hi, 0)));

  // "none" — one cell per element — is always offered as the leftmost stop,
  // because asking for no tiling is a meaningful intent even where the fit rule
  // cannot honour it. On a graph whose widest tensor cannot draw a cell per
  // element it settles to the same lattice as its neighbour; the settled size
  // shown beside the slider is what makes that visible rather than silent.
  return stops[0] === TILE_SCALE_MIN ? stops : [TILE_SCALE_MIN, ...stops];
}

/** The request meaning "no tiling": one drawn cell per element, where it fits. */
export const TILE_SCALE_NONE = TILE_SCALE_MIN;

/**
 * The tile every tensor actually settles on at this scale. A single number when
 * the graph agrees, a spread when the fit rule coarsens some tensors and not
 * others — the control is global but the outcome is per tensor, and saying one
 * number for a graph that has two would be a lie.
 */
export function settledTiles(
  planes: { rows: number; cols: number }[],
  scale: number,
  px: number
): { min: number; max: number } {
  const tiles = tileState(planes, scale, px);
  if (!tiles.length) return { min: 1, max: 1 };
  return { min: Math.min(...tiles), max: Math.max(...tiles) };
}

/** Find which effective stop renders the same graph-wide lattice as `scale`. */
export function effectiveTileScaleIndex(
  planes: { rows: number; cols: number }[],
  px: number,
  stops: number[],
  scale: number
): number {
  const wanted = tileState(planes, scale, px);
  const found = stops.findIndex((stop) => sameTileState(tileState(planes, stop, px), wanted));
  if (found >= 0) return found;
  // Defensive fallback for a future non-monotone tiling policy.
  let best = 0;
  for (let i = 1; i < stops.length; i++)
    if (Math.abs(stops[i] - scale) < Math.abs(stops[best] - scale)) best = i;
  return best;
}
