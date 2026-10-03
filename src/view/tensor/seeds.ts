/** Default extents: the planes cards draw, starter tiles, and first plan divisions. */

import { graphOutputs, producerNode, ResolvedGraph } from "../../core/graph";
import { Box } from "../../core/region";
import { tileOf } from "./grid";
import { viewAxes, ViewCfg } from "./tensor-view";
import { seedTile } from "./tile-spec";
import { planeExtents } from "./tiling";

/** The 2-D planes every card will draw, in the fixed row-major projection. */
export function planesOf(resolved: ResolvedGraph): { rows: number; cols: number }[] {
  return Object.values(resolved.tensors).map((t) => {
    const shape = t.resolved!;
    const { rowAxis, colAxis } = viewAxes(shape);
    return planeExtents(shape, rowAxis, colAxis);
  });
}

/**
 * Two tiles worth offering someone who has not drawn one: the graph's result,
 * and the last thing computed before it. They are the fastest path from a
 * loaded workspace to a cone worth reading, and both are one selection away.
 *
 * The box is one tile of the lattice currently drawn, so the offered tile is
 * the one the canvas would have snapped a click to.
 */
export function startingTiles(
  resolved: ResolvedGraph,
  tileScale: number,
  graphPx: number,
  viewCfgs: Record<string, ViewCfg> = {}
): { label: string; tensorId: string; box: Box }[] {
  const output = graphOutputs(resolved)[0];
  if (!output) return [];
  const producer = producerNode(resolved, output.id);
  const feeding = producer?.inputs.filter((id) => resolved.tensors[id].producer) ?? [];
  const previous = feeding.length ? resolved.tensors[feeding[feeding.length - 1]] : null;

  return [
    { tensor: output, label: "the output" },
    ...(previous ? [{ tensor: previous, label: "one step back" }] : []),
  ].map(({ tensor, label }) => {
    const shape = tensor.resolved!;
    const tile = seedTile(shape, viewCfgs[tensor.id], tileOf(shape, tileScale, graphPx, viewCfgs[tensor.id]));
    return {
      label,
      tensorId: tensor.id,
      box: shape.map((extent, axis) => ({ lo: 0, hi: Math.min(tile[axis], extent) })),
    };
  });
}

/**
 * The extents a tensor is first divided at: the tensor's own tile when it has
 * one, else the tile the canvas is drawing on its visible axes and one element
 * on the others, which is how a kernel grid usually assigns batch and head.
 *
 * The tile seeds a plan and never steers it again. Retiling on a detail change
 * would make a plan a function of the view, so a plan written down at one zoom
 * would mean something else at another.
 */
export function defaultPlanTile(
  resolved: ResolvedGraph,
  tensorId: string,
  tileScale: number,
  graphPx: number,
  viewCfgs: Record<string, ViewCfg> = {}
): number[] {
  const shape = resolved.tensors[tensorId].resolved!;
  return seedTile(shape, viewCfgs[tensorId], tileOf(shape, tileScale, graphPx, viewCfgs[tensorId]));
}
