/**
 * A tile plan: which produced tensors are divided into tasks, and how.
 *
 * Tiling a tensor is what writes it to memory. A task computes one complete
 * tile of one tiled tensor. It reads tiled tensors and graph inputs, and
 * computes every untiled tensor between them and its tile itself, so an untiled
 * intermediate is recomputed by each task that needs it. The task's demand is
 * therefore the bounded cone from its tile, stopped at every tiled tensor and
 * graph input.
 *
 * A plan keeps the graph it was checked against, so every query on it runs
 * against that graph.
 */

import { executeBoundedQuery } from "../executor";
import type { ResolvedGraph } from "../graph";
import type { BoundedCone } from "../propagate";
import { fromBox } from "../region";
import { isTile, PlanError, tileBox, TileFamily, tileFamily } from "./tile-family";

export type TilePlan = {
  readonly graph: ResolvedGraph;
  /** One family per planned tensor, in tensor-id order. */
  readonly families: ReadonlyMap<string, TileFamily>;
  /** Where every task stops: the tiled tensors and the graph inputs, sorted. */
  readonly frontier: readonly string[];
};

/** A task: tile `coord` of the family planned for `tensorId`. */
export type TaskRef = { readonly tensorId: string; readonly coord: readonly number[] };

/**
 * Check a plan against a graph. `tiles` maps each planned tensor to its tile
 * extents. Only produced tensors can be planned: a graph input has no task
 * that computes it.
 */
export function tilePlan(
  graph: ResolvedGraph,
  tiles: Readonly<Record<string, readonly number[]>>
): TilePlan {
  const families = new Map<string, TileFamily>();
  for (const tensorId of Object.keys(tiles).sort()) {
    const tensor = graph.tensors[tensorId];
    if (!tensor) throw new PlanError("PLAN_UNKNOWN_TENSOR", `unknown tensor "${tensorId}"`);
    if (!tensor.producer)
      throw new PlanError(
        "PLAN_GRAPH_INPUT",
        `"${tensorId}" is a graph input; no task produces it, so it has no tiles to plan`
      );
    families.set(tensorId, tileFamily(tensorId, tensor.resolved!, tiles[tensorId]));
  }
  const frontier = Object.values(graph.tensors)
    .filter((t) => !t.producer || families.has(t.id))
    .map((t) => t.id)
    .sort();
  return { graph, families, frontier };
}

/** The family a task belongs to, after checking that the task names one of its tiles. */
export function familyOf(plan: TilePlan, task: TaskRef): TileFamily {
  const family = plan.families.get(task.tensorId);
  if (!family) throw new PlanError("PLAN_UNPLANNED", `the plan does not tile "${task.tensorId}"`);
  if (!isTile(family, task.coord))
    throw new PlanError(
      "PLAN_TASK",
      `"${task.tensorId}" has no tile [${task.coord.join(", ")}]; its grid is [${family.grid.join(", ")}]`
    );
  return family;
}

/**
 * What one task reads and computes: the bounded cone from its tile, stopped at
 * the plan's frontier. `crossings` gives what it reads per operand slot, so a
 * tensor read through two slots appears twice; the cone's other produced
 * tensors are the untiled intermediates the task computes itself.
 */
export function taskDemand(plan: TilePlan, task: TaskRef): BoundedCone {
  const family = familyOf(plan, task);
  return executeBoundedQuery(plan.graph, {
    tensorId: task.tensorId,
    region: fromBox(tileBox(family, task.coord)),
    direction: "backward",
    frontier: plan.frontier,
  });
}
