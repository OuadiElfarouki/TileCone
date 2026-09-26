/**
 * The tile a tensor is worked in: one extent per axis.
 *
 * Before this existed the app had several tile notions that only agreed by
 * accident: the square display lattice, the extents a drag happened to cover,
 * the extents a plan was seeded with, and the anchor a reuse sweep walked. A
 * tile on `[B, H, S, D]` such as `[1, 2, 64, 128]` could not be expressed by a
 * gesture at all, because the lattice was square and the hidden axes were
 * either whole (projection) or one index (slice).
 *
 * A tensor's tile is stored on its view config (`ViewCfg.tile`) and, when set,
 * governs the lattice drawn on the visible axes, the extent a snapped gesture
 * selects on every axis, the arrow-key step, the starter tile, and the extents
 * a plan is first divided at. When it is absent every one of those falls back
 * to the behaviour the canvas had before: the square display tile on the
 * visible axes, and on the hidden axes the view mode's reading. The detail
 * slider therefore only governs tensors with no tile of their own.
 *
 * Nothing here changes a stored selection. Like snapping and detail, the tile
 * is a property of future gestures, not of boxes already drawn.
 */

import type { Box, Interval } from "../../core/region";
import { viewAxes, type ViewCfg } from "./tensor-view";

/**
 * The extent a snapped gesture selects on each axis.
 *
 * `displayTile` is the square tile the canvas would draw without a tile of the
 * tensor's own (`grid.tileOf`). On a hidden axis the fallback follows the view:
 * projection draws the union over that axis, so a gesture takes all of it, and
 * slice mode shows one index, so a gesture takes that index.
 */
export function gestureTile(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "projection" | "tile" | "axes"> | undefined,
  displayTile: number
): number[] {
  if (cfg?.tile && cfg.tile.length === shape.length) return cfg.tile.slice();
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  return shape.map((extent, axis) =>
    axis === rowAxis || axis === colAxis
      ? Math.max(1, Math.min(extent, displayTile))
      : cfg?.projection === false ? 1 : Math.max(1, extent)
  );
}

/**
 * The extents a starter tile and a first plan division use.
 *
 * The tensor's own tile when it has one. Otherwise the display tile on the
 * visible axes and one element on the others, which is how a kernel grid
 * usually assigns batch and head.
 */
export function seedTile(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "tile" | "axes"> | undefined,
  displayTile: number
): number[] {
  if (cfg?.tile && cfg.tile.length === shape.length) return cfg.tile.slice();
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  return shape.map((extent, axis) =>
    axis === rowAxis || axis === colAxis ? Math.max(1, Math.min(extent, displayTile)) : 1
  );
}

/** The tile of extent `tile` that contains index `i`, clipped to the axis. */
export function tileSpanAt(i: number, tile: number, extent: number): Interval {
  const lo = Math.floor(i / tile) * tile;
  return { lo, hi: Math.min(extent, lo + tile) };
}

/** The box of the tile at `coord` in a tiling of `shape` by `tile`. */
export function tileBoxAt(
  shape: readonly number[],
  tile: readonly number[],
  coord: readonly number[]
): Box {
  return shape.map((extent, axis) => tileSpanAt(coord[axis] * tile[axis], tile[axis], extent));
}

/** How many tiles of `tile` cover an axis of `extent`. */
export const tileCount = (extent: number, tile: number): number =>
  Math.max(1, Math.ceil(extent / tile));

/** The extent of the last tile on an axis: shorter than `tile` when it does not divide. */
export const lastTileExtent = (extent: number, tile: number): number =>
  extent - (tileCount(extent, tile) - 1) * tile;

/**
 * Where an interval sits in a tiling of its axis.
 *
 * `coord` is the tile its lower edge falls in. `aligned` is true when the
 * interval is exactly that tile, including a shortened last one. `last` is the
 * final tile of a run when the interval covers several whole tiles, and equals
 * `coord` otherwise; `whole` says the interval's edges both lie on the lattice.
 * An interval drawn at another offset is described by its range alone, since
 * it is not made of tiles of this tiling.
 */
export function tilePosition(
  interval: Interval,
  tile: number,
  extent: number
): { coord: number; last: number; count: number; aligned: boolean; whole: boolean } {
  const coord = Math.floor(interval.lo / tile);
  const span = tileSpanAt(interval.lo, tile, extent);
  const whole = interval.lo % tile === 0 && (interval.hi % tile === 0 || interval.hi === extent);
  return {
    coord,
    last: whole ? Math.ceil(interval.hi / tile) - 1 : coord,
    count: tileCount(extent, tile),
    aligned: span.lo === interval.lo && span.hi === interval.hi,
    whole,
  };
}

/** A tile is valid for a shape when it names every axis with an extent in [1, axis]. */
export function tileFits(shape: readonly number[], tile: readonly number[]): boolean {
  return tile.length === shape.length &&
    tile.every((t, axis) => Number.isSafeInteger(t) && t >= 1 && t <= Math.max(1, shape[axis]));
}
