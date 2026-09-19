/**
 * Regular tilings of a tensor: the unit an execution plan assigns work by.
 *
 * A tile family covers a whole tensor with boxes of one extent per axis. On an
 * axis the extent does not divide, the last tile is shorter. Every element
 * lies in exactly one tile.
 *
 * This is separate from the canvas lattice in `ui/tiling.ts`, which is square,
 * follows zoom and display detail, and exists for drawing. A family belongs to
 * a plan and changes only when the plan does.
 *
 * Coordinates are enumerated lazily in row-major order. That order is how
 * tiles are listed, not an execution order.
 */

import type { Box } from "../region";

export type TileFamily = {
  readonly tensorId: string;
  /** The tensor's extent on each axis. */
  readonly shape: readonly number[];
  /** A full tile's extent on each axis. */
  readonly tile: readonly number[];
  /** Tiles along each axis: `ceil(shape / tile)`. */
  readonly grid: readonly number[];
  /** Total tiles, checked to be a safe integer. */
  readonly count: number;
};

export type PlanErrorCode =
  | "PLAN_UNKNOWN_TENSOR"
  | "PLAN_GRAPH_INPUT"
  | "PLAN_TILE"
  | "PLAN_SIZE"
  | "PLAN_UNPLANNED"
  | "PLAN_TASK";

export class PlanError extends Error {
  constructor(
    public code: PlanErrorCode,
    message: string
  ) {
    super(message);
    this.name = "PlanError";
  }
}

/**
 * Tile `shape` with boxes of extent `tile`. A tile extent larger than its axis
 * gives one tile covering the whole axis.
 */
export function tileFamily(
  tensorId: string,
  shape: readonly number[],
  tile: readonly number[]
): TileFamily {
  if (tile.length !== shape.length)
    throw new PlanError(
      "PLAN_TILE",
      `"${tensorId}" has rank ${shape.length}, but its tile has rank ${tile.length}`
    );
  tile.forEach((extent, axis) => {
    if (!Number.isSafeInteger(extent) || extent < 1)
      throw new PlanError(
        "PLAN_TILE",
        `"${tensorId}" axis ${axis}: a tile extent must be a positive integer, not ${String(extent)}`
      );
  });
  const grid = shape.map((extent, axis) => Math.ceil(extent / tile[axis]));
  const count = grid.reduce((n, g) => n * g, 1);
  if (!Number.isSafeInteger(count))
    throw new PlanError("PLAN_SIZE", `"${tensorId}": ${count} tiles is past the safe integer range`);
  return { tensorId, shape: [...shape], tile: [...tile], grid, count };
}

/** The box a tile covers, shortened at the tensor's far edge. */
export function tileBox(f: TileFamily, coord: readonly number[]): Box {
  return coord.map((c, axis) => ({
    lo: c * f.tile[axis],
    hi: Math.min((c + 1) * f.tile[axis], f.shape[axis]),
  }));
}

/** Elements in one tile. */
export function tileVolume(f: TileFamily, coord: readonly number[]): number {
  return tileBox(f, coord).reduce((n, { lo, hi }) => n * (hi - lo), 1);
}

/** A tile's row-major position in its family: a stable key for one task. */
export function tileOrdinal(f: TileFamily, coord: readonly number[]): number {
  let ordinal = 0;
  for (let axis = 0; axis < f.grid.length; axis++) ordinal = ordinal * f.grid[axis] + coord[axis];
  return ordinal;
}

/** The coordinate of the tile at a row-major position. */
export function tileCoord(f: TileFamily, ordinal: number): number[] {
  const coord = new Array<number>(f.grid.length);
  for (let axis = f.grid.length - 1; axis >= 0; axis--) {
    coord[axis] = ordinal % f.grid[axis];
    ordinal = Math.floor(ordinal / f.grid[axis]);
  }
  return coord;
}

/** True when `coord` names a tile of the family. */
export function isTile(f: TileFamily, coord: readonly number[]): boolean {
  return (
    coord.length === f.grid.length &&
    coord.every((c, axis) => Number.isSafeInteger(c) && c >= 0 && c < f.grid[axis])
  );
}

/** Every tile, in row-major order. */
export function tiles(f: TileFamily): Generator<number[]> {
  return range(
    f.grid.map(() => 0),
    f.grid
  );
}

/**
 * The tiles sharing at least one element with `box`, in row-major order.
 *
 * On a regular cover these are a range of coordinates on each axis, so the
 * cost is the number of tiles met rather than the size of the family.
 */
export function tilesMeeting(f: TileFamily, box: Box): Generator<number[]> {
  if (box.length !== f.shape.length)
    throw new Error(`"${f.tensorId}" has rank ${f.shape.length}, but the box has rank ${box.length}`);
  const lo: number[] = [];
  const hi: number[] = [];
  box.forEach(({ lo: a, hi: b }, axis) => {
    if (a < 0 || b > f.shape[axis])
      throw new Error(`"${f.tensorId}" axis ${axis}: [${a}, ${b}) is outside [0, ${f.shape[axis]})`);
    lo.push(Math.floor(a / f.tile[axis]));
    hi.push(b > a ? Math.ceil(b / f.tile[axis]) : lo[axis]);
  });
  return range(lo, hi);
}

/** Every coordinate with `lo <= c < hi` on each axis, row-major; one empty coordinate at rank 0. */
function* range(lo: readonly number[], hi: readonly number[]): Generator<number[]> {
  if (lo.some((l, axis) => l >= hi[axis])) return;
  const coord = [...lo];
  for (;;) {
    yield [...coord];
    let axis = coord.length - 1;
    while (axis >= 0) {
      if (++coord[axis] < hi[axis]) break;
      coord[axis] = lo[axis];
      axis--;
    }
    if (axis < 0) return;
  }
}
