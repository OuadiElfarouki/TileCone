/** Plan derivation shared by the plan actions and graph loading. */

import { ResolvedGraph } from "../core/graph";
import { supplyOf } from "../core/plan/interfaces";
import { TaskRef, tilePlan } from "../core/plan/plan";
import { isTile } from "../core/plan/tile-family";
import { defaultPlanTile } from "../view/tensor/seeds";
import { idRecord, operationForTensor } from "../view/workspace";
import { appendWorkspaceHistory, planEditOf, sameNumbers, sameTask, sameTiles } from "./history";
import { State } from "./types";

/**
 * How many plans can be kept for comparison. Two kept and the current one make
 * three columns, which is what the inspector's width holds legibly.
 */
export const MAX_KEPT_PLANS = 2;

export const NO_PLAN = {
  planTiles: idRecord<number[]>(),
  planTask: null,
  plan: null,
  planSupply: null,
} as const;

/**
 * The checked plan and the inspected task's supply for one graph.
 *
 * Every stored entry was checked when it was set, against the graph it is
 * stored beside. Entries are still checked one at a time here, so an entry the
 * graph cannot support is dropped instead of failing the whole plan, and a task
 * that no longer names a tile is cleared.
 */
export function derivePlan(
  resolved: ResolvedGraph | null,
  tiles: Record<string, number[]>,
  task: TaskRef | null,
  previous?: Pick<State, "plan" | "planTiles">
): Pick<State, "planTiles" | "planTask" | "plan" | "planSupply"> {
  if (!resolved) return NO_PLAN;
  /* A plan that divides the same tensors of the same graph at the same extents
   * is the same plan, and the panel holds it by identity: the family report,
   * the producer/consumer matrix and a report someone asked for by hand are all
   * memoized on it. Rebuilding it for an edit that did not change any tiling -
   * stepping the inspected task, most of all - discarded every one of those and
   * recomputed a family-wide analysis per arrow press.
   *
   * The comparison is against the stored tiling, which was validated when it
   * was stored, so equal tilings on one graph validate identically and the
   * checks below can be skipped with them. */
  if (
    previous?.plan &&
    previous.plan.graph === resolved &&
    sameTiles(previous.planTiles, tiles)
  ) {
    const family = task ? previous.plan.families.get(task.tensorId) : undefined;
    const kept = task && family && isTile(family, task.coord) ? task : null;
    return {
      planTiles: previous.planTiles,
      planTask: kept,
      plan: previous.plan,
      planSupply: kept ? supplyOf(previous.plan, kept) : null,
    };
  }
  const valid = Object.create(null) as Record<string, number[]>;
  for (const [tensorId, tile] of Object.entries(tiles)) {
    try {
      tilePlan(resolved, { [tensorId]: tile });
      valid[tensorId] = tile;
    } catch {
      // not plannable on this graph
    }
  }
  if (!Object.keys(valid).length) return NO_PLAN;
  const plan = tilePlan(resolved, valid);
  const family = task ? plan.families.get(task.tensorId) : undefined;
  const kept = task && family && isTile(family, task.coord) ? task : null;
  return { planTiles: valid, planTask: kept, plan, planSupply: kept ? supplyOf(plan, kept) : null };
}

/**
 * Tiling a consumer also tiles the produced tensors it reads.
 *
 * An untiled tensor is computed inside every task that reads it, so tiling the
 * consumer alone would open on a task that recomputes its whole upstream graph
 * and names no producer. Starting from one operation per task shows the
 * interface the view is for; clearing an operand's tiling is then what fuses
 * it. The extents are defaults like any other and can be changed or removed.
 */
export function withProducedInputs(
  state: Pick<State, "resolved" | "tileScale" | "graphPx" | "viewCfgs">,
  tiles: Record<string, number[]>,
  tensorId: string
): Record<string, number[]> {
  const resolved = state.resolved!;
  const producer = resolved.tensors[tensorId].producer;
  const node = producer && resolved.nodes.find((n) => n.id === producer.nodeId);
  if (!node) return tiles;
  const next = { ...tiles };
  for (const input of node.inputs)
    if (resolved.tensors[input]?.producer && !next[input])
      next[input] = defaultPlanTile(resolved, input, state.tileScale, state.graphPx, state.viewCfgs);
  return next;
}

/** The tile of a tiling that contains `element`. */
export const tileContaining = (tile: readonly number[], element: readonly number[]): number[] =>
  element.map((i, axis) => Math.floor(i / tile[axis]));

/**
 * Divide `tensorId` at `tile` and inspect the task holding `element`, together
 * with one history entry. Shared by the click that takes the tile under the
 * pointer and the drag that sets the extents first.
 */
export function inspectTask(
  state: State,
  set: (partial: Partial<State>) => void,
  tensorId: string,
  tile: number[],
  element: number[]
): void {
  const resolved = state.resolved!;
  try {
    tilePlan(resolved, { [tensorId]: tile });
  } catch {
    return; // callers offer extents from a gesture or a field; an invalid one changes nothing
  }
  const task = { tensorId, coord: tileContaining(tile, element) };
  const settled =
    sameNumbers(state.planTiles[tensorId] ?? [], tile) && sameTask(state.planTask, task);
  if (settled) {
    // Nothing moved. The operation highlight still follows the task, so that
    // clicking a tile again after looking elsewhere brings its row back.
    const selectedOp = operationForTensor(resolved, tensorId);
    if (selectedOp !== state.selectedOp) set({ selectedOp });
    return;
  }
  const tiles = idRecord(state.planTiles);
  tiles[tensorId] = tile;
  const completedTiles = withProducedInputs(state, tiles, tensorId);
  const next = derivePlan(resolved, completedTiles, task, state);
  if (!next.planTask) return;
  set({
    workspaceHistory: appendWorkspaceHistory(state.workspaceHistory, {
      selection: state.selection,
      nodeOffsets: state.nodeOffsets,
      plan: planEditOf(state),
    }),
    selectedOp: operationForTensor(resolved, tensorId),
    ...next,
  });
}
