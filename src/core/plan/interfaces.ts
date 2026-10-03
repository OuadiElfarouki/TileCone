/**
 * Producer/consumer interfaces between tile families, and what a plan costs.
 *
 * A consumer task depends on a producer task when what the consumer reads of
 * the producer's tensor meets the producer's tile. The intersection is the
 * witness: the elements of that tile the consumer reads, recorded per operand
 * slot that reads them. A task reads only tiled tensors and graph inputs
 * (`plan.ts`), so every read has a supplier, and a graph input read by two
 * operations gives neither of them a producer.
 *
 * A demand region can over-approximate. A dependency found through an exact
 * demand is definite. A dependency found only through over-approximated demand
 * is possible, because the true demand may not meet that tile. No true
 * dependency is omitted, since an over-approximated region contains the true
 * one.
 */

import { sharedDemand, type SharedDemand } from "../demand";
import { graphOutputs } from "../graph";
import {
  addFigures,
  byteFigure,
  differenceFigure,
  figure,
  Figure,
  flopsOver,
  ratioFigure,
  sumFigures,
} from "../metrics";
import type { BoundedCone } from "../propagate";
import { canonicalize, count, fromBox, intersect, Region, unionOf } from "../region";
import { elementCount } from "../shapes";
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
 * - `tasks`: a tiled tensor, supplied by the tasks of its family.
 */
export type Supplier = "input" | "tasks";

/** What one operand slot of one operation in the task reads of one tensor. */
export type Demand = {
  node: string;
  slot: number;
  tensorId: string;
  region: Region;
  supplier: Supplier;
};

/** A producer task that a consumer task reads from. */
export type ProducerNeed = {
  task: TaskRef;
  /** For each operand slot whose demand meets this tile, that demand intersected with the tile. */
  witnesses: { node: string; slot: number; region: Region }[];
  /** Distinct elements of the tile the consumer reads; `upper` when a witness is over-approximated. */
  used: Figure;
  /** Elements in the tile. */
  volume: number;
  /** True when some witness comes from an exact demand; false when the dependency is only possible. */
  definite: boolean;
};

/** An untiled tensor a task computes on the way to its tile, and how much of it. */
export type Computed = { tensorId: string; region: Region };

export type Supply = {
  task: TaskRef;
  /** One entry per operand slot and tensor, sorted by tensor, then node, then slot. */
  demand: Demand[];
  /** Each producer task once, however many slots read it; sorted by tensor, then row-major. */
  producers: ProducerNeed[];
  /**
   * The untiled tensors the task computes itself, sorted by tensor. Empty when
   * its operation reads only tiled tensors and graph inputs.
   */
  computes: Computed[];
  /** FLOPs to compute the tile and everything in `computes`. */
  flops: Figure;
};

/** What one task reads, which producer tasks supply it, and what it computes. */
export function supplyOf(plan: TilePlan, task: TaskRef): Supply {
  return joinDemand(plan, task, taskDemand(plan, task));
}

/**
 * The join behind `supplyOf`, over any bounded cone from the task's tile to
 * the plan's frontier.
 *
 * @internal Exported so tests can join a cone computed under lowered fallback
 * thresholds, which the checked executor does not accept.
 */
export function joinDemand(
  plan: TilePlan,
  task: TaskRef,
  cone: Pick<BoundedCone, "crossings" | "tensors">
): Supply {
  const demand: Demand[] = cone.crossings.map((c) => ({
    node: c.node,
    slot: c.slot,
    tensorId: c.tensorId,
    region: c.region,
    supplier: plan.families.has(c.tensorId) ? "tasks" : "input",
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
      need.witnesses.push({ node: d.node, slot: d.slot, region });
      need.definite ||= region.exact;
      byTile.set(ordinal, need);
    }
  }

  const producers: ProducerNeed[] = [];
  for (const tensorId of [...byTensor.keys()].sort())
    for (const [, need] of [...byTensor.get(tensorId)!].sort(([a], [b]) => a - b)) {
      const regions = need.witnesses.map((w) => w.region);
      const union = unionOf(regions);
      need.used = figure(count(union), union.exact ? "exact" : "upper", union.reasons);
      producers.push(need);
    }

  // The cone's produced tensors that are not on the frontier are the ones the
  // task computes itself. Its own tile is computed too, and is not listed.
  const computes: Computed[] = [...cone.tensors]
    .filter(([id]) =>
      id !== task.tensorId && plan.graph.tensors[id].producer && !plan.families.has(id))
    .map(([tensorId, { region }]) => ({ tensorId, region }))
    .sort((a, b) => a.tensorId.localeCompare(b.tensorId));
  const work = new Map<string, { region: Region }>(computes.map((c) => [c.tensorId, c]));
  const own = cone.tensors.get(task.tensorId);
  if (own) work.set(task.tensorId, own);

  return {
    task: { tensorId: task.tensorId, coord: [...task.coord] },
    demand,
    producers,
    computes,
    flops: flopsOver(plan.graph, work).flops,
  };
}

/** Tasks a family may have before `interfaceOf` declines to evaluate it. */
export const DEFAULT_TASK_BUDGET = 4096;

/**
 * One tensor's demand across every task of a consumer family (`SharedDemand`,
 * with the tasks as readers), and who supplies it.
 */
export type BoundaryDemand = SharedDemand & {
  supplier: Supplier;
  /**
   * For a planned tensor, the number of consumer tasks that read each producer
   * tile, keyed by tile ordinal. Tiles no task reads are absent. Null for any
   * other supplier. Counts only definite dependencies when `exact`.
   */
  fanOut: ReadonlyMap<number, number> | null;
};

/**
 * What a set of tasks costs, in the figures a plan is compared by. The same
 * record describes one family and the whole plan.
 */
export type Work = {
  tasks: number;
  /**
   * Producer tasks each task reads from, summed over tasks: the edges of the
   * task graph. An upper bound when some edge is only possible.
   */
  dependencies: Figure;
  /** FLOPs summed over tasks. */
  flops: Figure;
  /**
   * The part of `flops` that is the same work done by more than one task: an
   * untiled intermediate that several tasks compute, or a row statistic that
   * every tile across a normalised axis computes again. `flops` less the work
   * of computing everything the tasks compute once.
   */
  recomputed: Figure;
  /**
   * Bytes the tasks read, each task counted separately: no reuse between
   * tasks. A task counts an element once however many slots read it.
   */
  read: Figure;
  /** Bytes in the union of what the tasks read, tensor by tensor: each element once. */
  readDistinct: Figure;
  /** Bytes of the tiled tensors, each written once. */
  written: Figure;
  /** `flops / (read + written)`. */
  intensity: Figure;
};

export type InterfaceReport =
  | {
      status: "evaluated";
      tensorId: string;
      tasks: number;
      boundary: BoundaryDemand[];
      work: Work;
    }
  /**
   * The family has more tasks than the budget. No figures are given: a sum over
   * part of the family is a lower bound on the total, while every other figure
   * here is exact or an upper bound.
   */
  | { status: "over-budget"; tensorId: string; tasks: number; budget: number };

/**
 * One family's report, with the per-tensor unions it was built from, so a
 * plan-wide total can measure each element once across families as well.
 */
type FamilyEvaluation = {
  report: InterfaceReport;
  /** Per tensor read, the union of every task's reads. */
  reads: Map<string, Region>;
  /** Per tensor computed, including the family's own, the union of what the tasks compute. */
  computed: Map<string, Region>;
};

function checkBudget(budget: number): void {
  if (!Number.isSafeInteger(budget) || budget < 1)
    throw new PlanError("PLAN_SIZE", `task budget must be a positive safe integer, not ${String(budget)}`);
}

function evaluateFamily(plan: TilePlan, tensorId: string, budget: number): FamilyEvaluation {
  const family = plan.families.get(tensorId);
  if (!family) throw new PlanError("PLAN_UNPLANNED", `the plan does not tile "${tensorId}"`);
  if (family.count > budget)
    return {
      report: { status: "over-budget", tensorId, tasks: family.count, budget },
      reads: new Map(),
      computed: new Map(),
    };

  // Per tensor read: one region per task that reads it, each the union of
  // that task's slots, so a task counts an element once.
  type Acc = { supplier: Supplier; regions: Region[]; fanOut: Map<number, number> | null };
  const acc = new Map<string, Acc>();
  const computed = new Map<string, Region[]>();
  let flops = figure(0, "exact");
  let edges = 0;
  let edgesExact = true;

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
        regions: [],
        fanOut: supplier === "tasks" ? new Map<number, number>() : null,
      };
      acc.set(id, a);
      a.regions.push(unionOf(regions));
    }

    for (const need of supply.producers) {
      const producer = plan.families.get(need.task.tensorId)!;
      const fanOut = acc.get(need.task.tensorId)!.fanOut!;
      const ordinal = tileOrdinal(producer, need.task.coord);
      fanOut.set(ordinal, (fanOut.get(ordinal) ?? 0) + 1);
      edges++;
      edgesExact &&= need.definite;
    }

    flops = addFigures(flops, supply.flops);
    const own = fromBox(tileBox(family, coord));
    for (const { tensorId: id, region } of [...supply.computes, { tensorId, region: own }]) {
      const regions = computed.get(id) ?? [];
      regions.push(region);
      computed.set(id, regions);
    }
  }

  const reads = new Map<string, Region>();
  const boundary = [...acc.keys()].sort().map((id): BoundaryDemand => {
    const a = acc.get(id)!;
    reads.set(id, unionOf(a.regions));
    return {
      ...sharedDemand(plan.graph.tensors[id], a.regions),
      supplier: a.supplier,
      fanOut: a.fanOut,
    };
  });

  const unions = new Map([...computed].map(([id, regions]) => [id, unionOf(regions)]));
  const tensor = plan.graph.tensors[tensorId];
  const work = workOf(plan, {
    tasks: family.count,
    dependencies: figure(edges, edgesExact ? "exact" : "upper"),
    flops,
    read: sumFigures(boundary.map((b) => b.summed)),
    readDistinct: sumFigures(boundary.map((b) => b.distinct)),
    written: byteFigure(tensor, elementCount(tensor.resolved!), {
      exact: true,
      reasons: [],
    }),
    computed: unions,
  });
  return {
    report: { status: "evaluated", tensorId, tasks: family.count, boundary, work },
    reads,
    computed: unions,
  };
}

/** The derived figures of a `Work`, from what was summed over its tasks. */
function workOf(
  plan: TilePlan,
  sums: Omit<Work, "recomputed" | "intensity"> & { computed: Map<string, Region> }
): Work {
  // Measured on canonical unions: tiles of one tensor merge back into few
  // boxes, which keeps a per-box cost off an exponential partition.
  const once = flopsOver(
    plan.graph,
    new Map([...sums.computed].map(([id, region]) => [id, { region: canonicalize(region) }]))
  ).flops;
  return {
    tasks: sums.tasks,
    dependencies: sums.dependencies,
    flops: sums.flops,
    recomputed: differenceFigure(sums.flops, once),
    read: sums.read,
    readDistinct: sums.readDistinct,
    written: sums.written,
    intensity: ratioFigure(sums.flops, addFigures(sums.read, sums.written)),
  };
}

/**
 * Demand across every task of the family planned for `tensorId`: per boundary
 * tensor, the summed and distinct bytes read, their ratio, and the fan-out of
 * each producer tile, and the family's `Work`. Each task costs one bounded
 * query, so families with more than `budget` tasks are declined.
 */
export function interfaceOf(
  plan: TilePlan,
  tensorId: string,
  { budget = DEFAULT_TASK_BUDGET }: { budget?: number } = {}
): InterfaceReport {
  checkBudget(budget);
  return evaluateFamily(plan, tensorId, budget).report;
}

export type PlanReport = {
  /** One report per tiled tensor, in tensor-id order. */
  families: InterfaceReport[];
  /**
   * The whole plan's work, or null when some family is over budget: a total
   * over part of the plan would understate it.
   */
  total: Work | null;
  /** Graph outputs the plan does not tile, and so never writes. */
  unwritten: string[];
};

/**
 * Every family of the plan, and their total. Reads and work shared between
 * families are measured once in the distinct figures, so an intermediate two
 * families both compute counts as recomputed.
 */
export function planReport(
  plan: TilePlan,
  { budget = DEFAULT_TASK_BUDGET }: { budget?: number } = {}
): PlanReport {
  checkBudget(budget);
  const evaluations = [...plan.families.keys()].map((id) => evaluateFamily(plan, id, budget));
  const unwritten = graphOutputs(plan.graph)
    .map((t) => t.id)
    .filter((id) => !plan.families.has(id))
    .sort();
  const families = evaluations.map((e) => e.report);

  const works: Work[] = [];
  for (const r of families) if (r.status === "evaluated") works.push(r.work);
  if (works.length < families.length) return { families, total: null, unwritten };

  const reads = new Map<string, Region[]>();
  const computed = new Map<string, Region[]>();
  for (const e of evaluations) {
    for (const [id, region] of e.reads) reads.set(id, [...(reads.get(id) ?? []), region]);
    for (const [id, region] of e.computed) computed.set(id, [...(computed.get(id) ?? []), region]);
  }
  const total = workOf(plan, {
    tasks: works.reduce((n, w) => n + w.tasks, 0),
    dependencies: sumFigures(works.map((w) => w.dependencies)),
    flops: sumFigures(works.map((w) => w.flops)),
    read: sumFigures(works.map((w) => w.read)),
    readDistinct: sumFigures(
      [...reads].map(([id, regions]) => {
        const union = unionOf(regions);
        return byteFigure(plan.graph.tensors[id], count(union), union);
      })
    ),
    written: sumFigures(works.map((w) => w.written)),
    computed: new Map([...computed].map(([id, regions]) => [id, unionOf(regions)])),
  });
  return { families, total, unwritten };
}
