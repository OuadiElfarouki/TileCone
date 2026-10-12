import type { UnfoldDir } from "./tensor-view";

/** Each outer nesting level adds another innermost gap. */
export const UNFOLD_GAP_PX = 3;
/** An unfolded card has a larger budget than a single plane. */
export const MAX_UNFOLD_W = 1200;
export const MAX_UNFOLD_H = 860;

/** Split hidden axes without depending on the view's validation or geometry. */
export function splitHiddenAxes(hidden: number[], dir?: UnfoldDir): { across: number[]; down: number[] } {
  if (dir === "down") return { across: [], down: hidden };
  if (dir === "grid" && hidden.length > 1)
    return { across: hidden.slice(-1), down: hidden.slice(0, -1) };
  return { across: hidden, down: [] };
}

/** Nested hidden axes, slowest first, ending in the panel itself. */
export type FoldRun = {
  axes: number[];
  extents: number[];
  /** Px between neighbouring blocks at each level. */
  gaps: number[];
  /** Px of one block at each level; `sizes[axes.length]` is one panel. */
  sizes: number[];
  /** Px the whole run spans. */
  total: number;
};

/** Shared by sizing and validation so they account for the same nested gaps. */
export function foldRun(axes: number[], shape: readonly number[], panel: number): FoldRun {
  const n = axes.length;
  const extents = axes.map((axis) => Math.max(1, shape[axis]));
  const gaps = axes.map((_, level) => UNFOLD_GAP_PX * (n - level));
  const sizes = new Array<number>(n + 1);
  sizes[n] = panel;
  for (let level = n - 1; level >= 0; level--)
    sizes[level] = extents[level] * sizes[level + 1] + (extents[level] - 1) * gaps[level];
  return { axes, extents, gaps, sizes, total: sizes[0] };
}
