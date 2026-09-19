/**
 * A tile plan: which produced tensors are divided into tasks, and how.
 *
 * A task computes one complete tile of one produced tensor. Its demand is what
 * the producing operation reads to compute that tile, which is the bounded
 * cone from the tile stopped at that operation's inputs.
 *
 * A plan keeps the graph it was checked against, so every query on it runs
 * against that graph.
 */

import { executeBoundedQuery } from "../executor";
import type { Node, ResolvedGraph } from "../graph";
import type { BoundedCone } from "../propagate";
import { fromBox } from "../region";
import { isTile, PlanError, tileBox, TileFamily, tileFamily } from "./tile-family";

export type TilePlan = {
  readonly graph: ResolvedGraph;
  /** One family per planned tensor, in tensor-id order. */
  readonly families: ReadonlyMap<string, TileFamily>;
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
  return { graph, families };
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

const producerIndex = new WeakMap<ResolvedGraph, Map<string, Node>>();

function producerOf(graph: ResolvedGraph, tensorId: string): Node {
  let index = producerIndex.get(graph);
  if (!index) {
    index = new Map();
    for (const node of graph.nodes) for (const out of node.outputs) index.set(out, node);
    producerIndex.set(graph, index);
  }
  return index.get(tensorId)!;
}

/**
 * What one task reads: the bounded cone from its tile, stopped at the inputs
 * of the operation that computes it. `crossings` gives the demand per operand
 * slot, so a tensor the operation reads through two slots appears twice.
 */
export function taskDemand(plan: TilePlan, task: TaskRef): BoundedCone {
  const family = familyOf(plan, task);
  return executeBoundedQuery(plan.graph, {
    tensorId: task.tensorId,
    region: fromBox(tileBox(family, task.coord)),
    direction: "backward",
    frontier: producerOf(plan.graph, task.tensorId).inputs,
  });
}
