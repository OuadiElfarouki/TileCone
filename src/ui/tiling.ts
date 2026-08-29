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
 */

export const AUTO_TILE_FRACTION = 0.05;
export const TILE_SCALE_MIN = -5;
export const TILE_SCALE_MAX = 5;

/** Rendering budget. */
export const MIN_CELL_PX = 3;
export const MAX_CARD_W = 360;
export const MAX_CARD_H = 300;
/** Cap on pixels per element, so a tiny tensor does not fill the whole card. */
export const MAX_ELEM_PX = 22;

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
 * On-screen size of a tensor's grid, in CSS pixels. Depends **only on the
 * tensor's shape** — never on the tile size — so changing detail re-lattices the
 * card in place instead of resizing it. Aspect ratio follows the tensor.
 */
export function cardPx(rows: number, cols: number): { w: number; h: number } {
  const scale = Math.min(MAX_CARD_W / cols, MAX_CARD_H / rows, MAX_ELEM_PX);
  return { w: Math.max(8, Math.round(cols * scale)), h: Math.max(8, Math.round(rows * scale)) };
}

/** The tile actually used for this tensor, given the global detail setting. */
export function tileFor(rows: number, cols: number, tileScale: number): number {
  const base = pow2Round(AUTO_TILE_FRACTION * governingAxis(rows, cols));
  let tile = clampTile(base * 2 ** tileScale, rows, cols);
  // fit: the card is a fixed size, so refuse tiles that would draw cells below
  // MIN_CELL_PX inside it. This is the minimum elementary tile.
  const { w, h } = cardPx(rows, cols);
  const maxCols = Math.max(1, Math.floor(w / MIN_CELL_PX));
  const maxRows = Math.max(1, Math.floor(h / MIN_CELL_PX));
  while (Math.ceil(cols / tile) > maxCols || Math.ceil(rows / tile) > maxRows) tile *= 2;
  return tile;
}

/** True when `tileFor` had to override the requested scale to keep cells legible. */
export function isTileClamped(rows: number, cols: number, tileScale: number): boolean {
  const base = pow2Round(AUTO_TILE_FRACTION * governingAxis(rows, cols));
  return tileFor(rows, cols, tileScale) !== base * 2 ** tileScale;
}

