/**
 * What a pointer gesture on a tensor card selects: the element range a drag
 * covers, the tile it takes on the hidden axes, and what a Plan-view gesture
 * does. Pure, so the card, the hover preview and tests share one definition.
 */

import { Box, iv, Region } from "../../core/region";
import { GridGeom, snapSpan } from "./grid";
import { viewAxes, ViewCfg } from "./tensor-view";
import { gestureTile, seedTile, tileSpanAt } from "./tile-spec";

export type CellDrag = { r0: number; c0: number; r1: number; c1: number };

/**
 * The exact tensor element under a card cell while inspecting a plan.
 *
 * Plan tiles have their own lattice, so display-lattice snapping must not
 * participate here. Hidden axes keep the same semantics as a one-element
 * selection: the active slice when sliced, and index zero when projected.
 */
/** @internal Pure interaction seam exported for plan-view tests. */
export function planElementFromCell(
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom,
  cell: { row: number; col: number }
): number[] {
  return selectionBoxFromDrag(
    shape,
    cfg,
    geom,
    { r0: cell.row, c0: cell.col, r1: cell.row, c1: cell.col },
    false
  ).map((interval) => interval.lo);
}

/** Convert a visible-plane drag to the tensor region it visually promises.
 *
 * A gesture only names the two visible axes. On the others it takes the
 * tensor's tile at the hidden-axis position (`cfg.sliders`). Without a tile of
 * the tensor's own that is the view's reading: projection represents the union
 * across hidden axes, so a projection gesture selects their full extent, and
 * slice mode stays pinned to its sliders. A tile such as `H = 2` instead takes
 * the two heads containing the slider, in either mode, because the tile states
 * that extent explicitly. */
/** @internal Pure interaction seam exported for tensor-card tests. */
export function selectionBoxFromDrag(
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom,
  drag: CellDrag,
  snapToGrid: boolean
): Box {
  const [rLo, rHi] = snapToGrid
    ? snapSpan(drag.r0, drag.r1, geom.rowTile, geom.rows)
    : [Math.min(drag.r0, drag.r1), Math.max(drag.r0, drag.r1) + 1];
  const [cLo, cHi] = snapToGrid
    ? snapSpan(drag.c0, drag.c1, geom.colTile, geom.cols)
    : [Math.min(drag.c0, drag.c1), Math.max(drag.c0, drag.c1) + 1];
  // The display tile only matters on the visible axes, which were settled
  // above, so any value serves as the fallback here.
  const tile = gestureTile(shape, cfg, 1);
  return shape.map((extent, ax) => {
    if (ax === geom.rowAxis) return iv(rLo, rHi);
    if (ax === geom.colAxis) return iv(cLo, cHi);
    const at = tileSpanAt(cfg.sliders[ax] ?? 0, tile[ax], extent);
    return iv(at.lo, at.hi);
  });
}

/** Combine exactness across every cone currently visible on a tensor card. */
/** @internal Pure rendering seam exported for deterministic canvas tests. */
export function visibleApproximation(...regions: (Region | undefined)[]): {
  approximate: boolean;
  reasons: string[];
} {
  const inexact = regions.filter((region): region is Region => !!region && !region.exact);
  return {
    approximate: inexact.length > 0,
    reasons: [...new Set(inexact.flatMap((region) => region.reasons))],
  };
}

/**
 * What a gesture in the Plan view does.
 *
 * A tensor is divided once. While it has no tiling, a drag draws the tile the
 * whole tensor is covered by and a press without movement takes the proposed
 * one. Once it has a tiling, every gesture inspects the task under the press:
 * redrawing would move the lattice under the reader mid-study and change what
 * the figures beside it are about. Dividing it differently is a deliberate act,
 * by clearing the tiling or typing the extents.
 *
 * A drag always snaps to the drawn lattice, whatever the snap toggle says. An
 * extent is a quantity here, not a highlight, and `73 x 128` from wherever a
 * pointer stopped is not a tiling anyone chose.
 */
/** @internal Pure interaction seam exported for plan-view tests. */
export function planGesture(
  shape: number[],
  cfg: ViewCfg,
  geom: GridGeom,
  drag: CellDrag,
  tiled: boolean
): { kind: "inspect"; element: number[] } | { kind: "divide"; tile: number[]; element: number[] } {
  const moved = drag.r0 !== drag.r1 || drag.c0 !== drag.c1;
  if (tiled || !moved)
    return {
      kind: "inspect",
      element: planElementFromCell(shape, cfg, geom, { row: drag.r0, col: drag.c0 }),
    };
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const box = selectionBoxFromDrag(shape, cfg, geom, drag, true);
  const extentOn = (axis: number) => (axis >= 0 ? box[axis].hi - box[axis].lo : 1);
  // The drawn extents on the visible axes; the tensor's own tile, else one
  // element, on the others.
  const tile = seedTile(shape, cfg, 1);
  if (rowAxis >= 0) tile[rowAxis] = extentOn(rowAxis);
  if (colAxis >= 0) tile[colAxis] = extentOn(colAxis);
  return { kind: "divide", tile, element: box.map((interval) => interval.lo) };
}
