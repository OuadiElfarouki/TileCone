/**
 * Producer/consumer interfaces between tile families.
 *
 * A consumer task depends on a producer task when the consumer's demand on the
 * producer's tensor meets the producer's tile. The intersection is the
 * witness: the elements of that tile the consumer reads, recorded per consumer
 * operand slot. Demand is taken at the consumer operation's own inputs, so a
 * graph input read by two operations gives neither of them a producer.
 *
 * A demand region can over-approximate. A dependency found through an exact
 * demand is definite. A dependency found only through over-approximated demand
 * is possible, because the true demand may not meet that tile. No true
 * dependency is omitted, since an over-approximated region contains the true
 * one.
 */

import { addFigures, byteFigure, figure, Figure, ratioFigure } from "../metrics";
import type { Crossing } from "../propagate";
import { Box, count, fromBox, intersect, Region } from "../region";
import { TaskRef, taskDemand, TilePlan } from "./plan";
import {
  PlanError,
  tileBox,
  tileOrdinal,
  tiles,
  tilesMeeting,
  tileVolume,
} from "./tile-family";

/**
 * What supplies a demand:
 * - `input`: a graph input, read from memory and produced by no task;
 * - `tasks`: a planned tensor, supplied by the tasks of its family;
 * - `unplanned`: a produced tensor the plan does not tile, so its producer
 *   tasks are not specified.
 */
export type Supplier = "input" | "tasks" | "unplanned";

/** One consumer operand slot's demand on one tensor. */
export type Demand = { slot: number; tensorId: string; region: Region; supplier: Supplier };

/** A producer task that a consumer task reads from. */
export type ProducerNeed = {
  task: TaskRef;
  /** For each consumer slot whose demand meets this tile, that demand intersected with the tile. */
  witnesses: { slot: number; region: Region }[];
  /** Distinct elements of the tile the consumer reads; `upper` when a witness is over-approximated. */
  used: Figure;
  /** Elements in the tile. */
  volume: number;
  /** True when some witness comes from an exact demand; false when the dependency is only possible. */
  definite: boolean;
};

export type Supply = {
  task: TaskRef;
  /** One entry per operand slot and tensor, sorted by tensor, then slot. */
  demand: Demand[];
  /** Each producer task once, however many slots read it; sorted by tensor, then row-major. */
  producers: ProducerNeed[];
  /** False when some demand is `unplanned`, so `producers` is not the full set. */
  complete: boolean;
};

function supplierOf(plan: TilePlan, tensorId: string): Supplier {
  if (!plan.graph.tensors[tensorId].producer) return "input";
  return plan.families.has(tensorId) ? "tasks" : "unplanned";
}

/** What one task reads, and which producer tasks supply it. */
export function supplyOf(plan: TilePlan, task: TaskRef): Supply {
  return joinDemand(plan, task, taskDemand(plan, task).crossings);
}

/**
 * The join behind `supplyOf`, over the crossings of any bounded cone from the
 * task's tile.
 *
 * @internal Exported so tests can join demand computed under lowered fallback
 * thresholds, which the checked executor does not accept.
 */
export function joinDemand(plan: TilePlan, task: TaskRef, crossings: readonly Crossing[]): Supply {
  const demand: Demand[] = crossings.map((c) => ({
    slot: c.slot,
    tensorId: c.tensorId,
    region: c.region,
    supplier: supplierOf(plan, c.tensorId),
  }));

  const byTensor = new Map<string, Map<number, ProducerNeed>>();
  for (const d of demand) {
    if (d.supplier !== "tasks") continue;
    const family = plan.families.get(d.tensorId)!;
    const byTile = byTensor.get(d.tensorId) ?? new Map<number, ProducerNeed>();
    byTensor.set(d.tensorId, byTile);

    // The tiles a region meets are the union of the tiles each box meets.
    const met = new Map<number, number[]>();
    for (const b of d.region.boxes)
      for (const coord of tilesMeeting(family, b)) met.set(tileOrdinal(family, coord), coord);

    for (const [ordinal, coord] of met) {
      const region = intersect(d.region, fromBox(tileBox(family, coord)));
      const need = byTile.get(ordinal) ?? {
        task: { tensorId: d.tensorId, coord },
        witnesses: [],
        used: figure(0, "exact"),
        volume: tileVolume(family, coord),
        definite: false,
      };
      need.witnesses.push({ slot: d.slot, region });
      need.definite ||= region.exact;
      byTile.set(ordinal, need);
    }
  }

  const producers: ProducerNeed[] = [];
  for (const tensorId of [...byTensor.keys()].sort())
    for (const [, need] of [...byTensor.get(tensorId)!].sort(([a], [b]) => a - b)) {
      const regions = need.witnesses.map((w) => w.region);
      const union = merged(regions);
      need.used = figure(count(union), union.exact ? "exact" : "upper", union.reasons);
      producers.push(need);
    }

  return {
    task: { tensorId: task.tensorId, coord: [...task.coord] },
    demand,
    producers,
    complete: demand.every((d) => d.supplier !== "unplanned"),
  };
}

/** Tasks a family may have before `interfaceOf` declines to evaluate it. */
export const DEFAULT_TASK_BUDGET = 4096;

/** One tensor's demand across every task of a consumer family. */
export type BoundaryDemand = {
  tensorId: string;
  supplier: Supplier;
  /** Tasks whose demand includes this tensor. */
  readers: number;
  /**
   * Bytes of this tensor each task reads, summed over tasks. A task counts an
   * element once however many of its slots read it.
   */
  summed: Figure;
  /** Bytes in the union of every task's demand. */
  distinct: Figure;
  /**
   * `summed / distinct`: the mean number of tasks that read a demanded element.
   * This is demand duplication, not a cache hit rate or a count of memory
   * transfers.
   */
  duplication: Figure;
  /** True when every task's demand on this tensor is exact, so `fanOut` counts only definite dependencies. */
  exact: boolean;
  /**
   * For a planned tensor, the number of consumer tasks that read each producer
   * tile, keyed by tile ordinal. Tiles no task reads are absent. Null for any
   * other supplier.
   */
  fanOut: ReadonlyMap<number, number> | null;
};

export type InterfaceReport =
  | { status: "evaluated"; tensorId: string; tasks: number; boundary: BoundaryDemand[] }
  /**
   * The family has more tasks than the budget. No figures are given: a sum over
   * part of the family is a lower bound on the total, while every other figure
   * here is exact or an upper bound.
   */
  | { status: "over-budget"; tensorId: string; tasks: number; budget: number };

/**
 * Demand across every task of the family planned for `tensorId`: per boundary
 * tensor, the summed and distinct bytes read, their ratio, and the fan-out of
 * each producer tile. Each task costs one bounded query, so families with more
 * than `budget` tasks are declined.
 */
export function interfaceOf(
  plan: TilePlan,
  tensorId: string,
  { budget = DEFAULT_TASK_BUDGET }: { budget?: number } = {}
): InterfaceReport {
  const family = plan.families.get(tensorId);
  if (!family) throw new PlanError("PLAN_UNPLANNED", `the plan does not tile "${tensorId}"`);
  if (!Number.isSafeInteger(budget) || budget < 1)
    throw new PlanError("PLAN_SIZE", `task budget must be a positive safe integer, not ${String(budget)}`);
  if (family.count > budget) return { status: "over-budget", tensorId, tasks: family.count, budget };

  type Acc = {
    supplier: Supplier;
    readers: number;
    summed: Figure;
    boxes: Box[];
    exact: boolean;
    reasons: Set<string>;
    fanOut: Map<number, number> | null;
  };
  const acc = new Map<string, Acc>();

  for (const coord of tiles(family)) {
    const supply = supplyOf(plan, { tensorId, coord });

    const perTensor = new Map<string, { supplier: Supplier; regions: Region[] }>();
    for (const d of supply.demand) {
      const entry = perTensor.get(d.tensorId) ?? { supplier: d.supplier, regions: [] };
      entry.regions.push(d.region);
      perTensor.set(d.tensorId, entry);
    }
    for (const [id, { supplier, regions }] of perTensor) {
      const a = acc.get(id) ?? {
        supplier,
        readers: 0,
        summed: figure(0, "exact"),
        boxes: [],
        exact: true,
        reasons: new Set<string>(),
        fanOut: supplier === "tasks" ? new Map<number, number>() : null,
      };
      acc.set(id, a);
      const union = merged(regions);
      a.readers++;
      a.summed = addFigures(a.summed, byteFigure(plan.graph.tensors[id], count(union), union));
      a.boxes.push(...union.boxes);
      a.exact &&= union.exact;
      union.reasons.forEach((r) => a.reasons.add(r));
    }

    for (const need of supply.producers) {
      const producer = plan.families.get(need.task.tensorId)!;
      const fanOut = acc.get(need.task.tensorId)!.fanOut!;
      const ordinal = tileOrdinal(producer, need.task.coord);
      fanOut.set(ordinal, (fanOut.get(ordinal) ?? 0) + 1);
    }
  }

  const boundary = [...acc.keys()].sort().map((id): BoundaryDemand => {
    const a = acc.get(id)!;
    const union: Region = { boxes: a.boxes, exact: a.exact, reasons: [...a.reasons].sort() };
    const distinct = byteFigure(plan.graph.tensors[id], count(union), union);
    return {
      tensorId: id,
      supplier: a.supplier,
      readers: a.readers,
      summed: a.summed,
      distinct,
      duplication: ratioFigure(a.summed, distinct),
      exact: a.exact,
      fanOut: a.fanOut,
    };
  });
  return { status: "evaluated", tensorId, tasks: family.count, boundary };
}

/**
 * The union of several regions on one tensor, as a region whose boxes may
 * overlap. It is only measured with `count`, which handles overlap, so it is
 * never canonicalized and cannot be coarsened by the box cap.
 */
function merged(regions: Region[]): Region {
  return {
    boxes: regions.flatMap((r) => r.boxes),
    exact: regions.every((r) => r.exact),
    reasons: [...new Set(regions.flatMap((r) => r.reasons))].sort(),
  };
}
