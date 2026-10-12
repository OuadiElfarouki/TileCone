/**
 * An unfolded card: every hidden axis laid out beside the drawn plane, so the
 * whole tensor is on one canvas without a slider or a union.
 *
 * Each hidden index gets one panel, a copy of the plane drawn at that index.
 * Panels are nested in stride order, the way the tensor sits in memory: the
 * slowest hidden axis is the outermost block and the fastest the innermost, so
 * reading a row of panels left to right walks the indices in the order a
 * row-major layout stores them. `[D, R, C]` unfolded across is `R x (D*C)` with
 * one `R x C` panel per `d`; unfolded down it is `(D*R) x C`, which for the
 * default plane is exactly the tensor's row-major flattening.
 *
 * Panels at each level are separated by a gap that widens with the level, so
 * `[A, B, R, C]` across reads as A groups of B panels rather than A*B equals.
 * The gaps carry no elements, which is why geometry, painting and hit-testing
 * all go through here rather than mapping pixels linearly.
 *
 * Pure and DOM-free.
 */

import type { Box } from "../../core/region";
import { cardPx } from "./tiling";
import { hiddenAxes, type ViewCfg } from "./tensor-view";
import { foldRun, MAX_UNFOLD_H, MAX_UNFOLD_W, splitHiddenAxes, type FoldRun } from "./fold-layout";
export { MAX_UNFOLD_H, MAX_UNFOLD_W, UNFOLD_GAP_PX, type FoldRun } from "./fold-layout";

/** Which hidden axes run across the card and which run down it, slowest first. */
export function foldAxes(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes" | "unfold">
): { across: number[]; down: number[] } {
  return splitHiddenAxes(hiddenAxes(shape, cfg), cfg.unfold);
}

/** How many panels lie across and down. */
export function foldCounts(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes" | "unfold">
): { across: number; down: number } {
  const { across, down } = foldAxes(shape, cfg);
  const count = (axes: number[]) => axes.reduce((n, axis) => n * Math.max(1, shape[axis]), 1);
  return { across: count(across), down: count(down) };
}

/** Distance between the starts of two neighbouring blocks at `level`. */
const pitch = (run: FoldRun, level: number) => run.sizes[level + 1] + run.gaps[level];

/**
 * The panel a position along a run falls in, and where inside it.
 *
 * A position in a gap belongs to the nearer panel, so a pointer crossing a gap
 * keeps naming an element rather than flickering to none.
 */
export function runLocate(run: FoldRun, px: number): { digits: number[]; local: number } {
  const digits: number[] = [];
  let rest = px;
  run.axes.forEach((_, level) => {
    const step = pitch(run, level);
    let digit = Math.floor(rest / step);
    const into = rest - digit * step;
    // Past the block's end is the gap after it: the nearer side wins.
    if (into >= run.sizes[level + 1] && into - run.sizes[level + 1] > run.gaps[level] / 2) digit += 1;
    digit = Math.max(0, Math.min(run.extents[level] - 1, digit));
    digits.push(digit);
    rest -= digit * step;
  });
  const panel = run.sizes[run.axes.length];
  return { digits, local: Math.max(0, Math.min(panel - 1e-9, rest)) };
}

/**
 * The start of every panel along a run that a box reaches, in stride order.
 * A box's hidden intervals pick a sub-block of panels; every one of them gets
 * its own copy of the box's plane rectangle.
 */
export function runSpans(run: FoldRun, box?: Box): number[] {
  let offsets = [0];
  run.axes.forEach((axis, level) => {
    const lo = Math.max(0, box ? box[axis].lo : 0);
    const hi = Math.min(run.extents[level], box ? box[axis].hi : run.extents[level]);
    const step = pitch(run, level);
    const next: number[] = [];
    for (const offset of offsets)
      for (let digit = lo; digit < hi; digit++) next.push(offset + digit * step);
    offsets = next;
  });
  return offsets;
}

/** An unfolded card's panels: one run across, one down, and the panel size. */
export type Fold = {
  rank: number;
  across: FoldRun;
  down: FoldRun;
  panelW: number;
  panelH: number;
};

/**
 * The fold of an unfolded card, or null for any other card. `rows` and `cols`
 * are the plane's extents and `scale` its px per element.
 */
export function foldGeometry(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes" | "unfold"> | undefined,
  rows: number,
  cols: number,
  scale: number
): Fold | null {
  if (!cfg?.unfold) return null;
  const { w, h } = cardPx(rows, cols, scale);
  const { across, down } = foldAxes(shape, cfg);
  return {
    rank: shape.length,
    across: foldRun(across, shape, w),
    down: foldRun(down, shape, h),
    panelW: w,
    panelH: h,
  };
}

/** Px per element that keeps an unfolded card inside its budget. */
export function foldScale(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes" | "unfold">,
  rows: number,
  cols: number,
  px: number
): number {
  const fits = (scale: number) => {
    const fold = foldGeometry(shape, cfg, rows, cols, scale)!;
    return fold.across.total <= MAX_UNFOLD_W && fold.down.total <= MAX_UNFOLD_H;
  };
  if (fits(px)) return px;
  // Use the actual rounded panel sizes and nested gaps. Some directions
  // cannot fit even at the minimum; view validation and the menu reject them.
  let lo = 0;
  let hi = px;
  for (let i = 0; i < 48; i++) {
    const mid = (lo + hi) / 2;
    if (fits(mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** The hidden coordinates a pair of run positions names, as a full-rank array
 *  with zeros on the drawn axes. */
export function foldCoord(fold: Fold, across: number[], down: number[]): number[] {
  const coord = new Array<number>(fold.rank).fill(0);
  fold.across.axes.forEach((axis, level) => { coord[axis] = across[level]; });
  fold.down.axes.forEach((axis, level) => { coord[axis] = down[level]; });
  return coord;
}
