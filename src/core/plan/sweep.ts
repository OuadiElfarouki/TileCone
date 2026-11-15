/**
 * A sweep of plans under a stated budget.
 *
 * The engineer states how many bytes a task may hold of its operands at a time
 * (`stepBytes`), and optionally how large an accumulator may be. The sweep
 * enumerates plans - which tensors are tiled, at which power-of-two extents,
 * with which chunks - keeps those within the budget, and returns the ones no
 * other candidate beats on every figure a plan is compared by.
 *
 * Nothing here models how a kernel holds anything. A step's payload is what
 * its terms read (`steps.ts`); the budget is a number the caller states; the
 * comparison is arithmetic (docs/STAGING.md, the scope rule).
 *
 * **How the space is searched.** The figures a candidate is ranked by separate
 * by tiled tensor once the set of tiled tensors is fixed: a family's task
 * count, FLOPs and bytes read depend on its own tile extents and not on any
 * other family's, because every task stops at the same frontier whatever the
 * other tilings are; and a step's payload depends on its own tile and chunks
 * alone. So each tiled tensor is measured on its own, one tile extent at a
 * time, and candidates are combinations of the options that survive a prune.
 * An option is pruned when another of the same tensor is no worse on task
 * count, FLOPs, bytes read and payload, and better on one of the three sums:
 * swapping it out then improves a sum and worsens nothing, so no unbeaten
 * combination contains it. Being better on payload alone is not enough, since
 * a plan's payload is a maximum and another tensor can set it. The rows
 * returned are then evaluated as whole plans by `planReport`, and those are
 * the figures reported.
 *
 * **Exact sets only.** The argument needs every measured figure to be exact.
 * Where a demand is over-approximated - a strided access past its enumeration
 * limit, a wide diagonal, a barrier - the union of the tasks' regions is not
 * the region of the whole, and the work a set does once is no longer one
 * number. Such a set is not searched, and is counted in `setsInexact`.
 *
 * **Chunks.** For one tile extent, plan costs do not depend on the chunks; only
 * the payload and the step count do. Each tile takes the chunking with the
 * fewest steps whose payload is within the budget, and a tile with none is
 * over the budget. A tensor without steps is one step per task: its payload is
 * the task's whole demand.
 *
 * **What is measured.** A payload is the largest over every task of the family
 * and every step of each, and an accumulator the largest over every task. No
 * task stands for another in general: a border task of a padded operation
 * reads less than an interior one, and a step that reaches a concatenated
 * operand reads more than the step before it. The one case measured on a
 * single step is the contraction whose operands are all read directly
 * (`measure`), where a step's payload is a product of its interval lengths.
 */

import { graphOutputs, nodeById, type ResolvedGraph } from "../graph";
import { type Figure, viewLayouts } from "../metrics";
import { planReport, taskSupplyOf, type Work } from "./interfaces";
import { type StepSpec, type TaskRef, tilePlan } from "./plan";
import { payloadOf, type Staged, stagingOf, stepOf, stepPayload, taskSteps } from "./steps";
import { PlanError, tiles as tilesOf } from "./tile-family";

export type Budget = Readonly<{
  /** Bytes a task may hold of its operands at a time: the step payload's limit. */
  stepBytes: number;
  /** Bytes an accumulator may take, when the caller bounds it. */
  accumulatorBytes?: number;
}>;

export type SweepOptions = Readonly<{
  budget: Budget;
  /**
   * The sets of tiled tensors to consider. Left out: every graph output tiled,
   * and each computed intermediate that is not a view either tiled or fused -
   * all combinations for up to `ALL_FUSIONS_UP_TO` such intermediates, and
   * otherwise none of them, each alone, and all of them.
   */
  tiled?: readonly (readonly string[])[];
  /** Combinations formed, across all sets. */
  maxPlans?: number;
  /** Rows evaluated as whole plans and returned. */
  maxRows?: number;
  /** Tasks in one family; a tile extent with more is not measured. */
  taskBudget?: number;
}>;

export const ALL_FUSIONS_UP_TO = 4;
export const DEFAULT_MAX_PLANS = 4096;
export const DEFAULT_MAX_ROWS = 64;
/**
 * Tasks in one family, by default. Every tile extent of every tensor is
 * measured task by task and step by step, so this bounds the sweep's work.
 * Finer tilings are counted in `overTasks`, not silently dropped.
 */
export const SWEEP_TASK_BUDGET = 256;
/** Chunkings tried per tile extent. */
const MAX_CHUNKINGS = 256;

export type SweepRow = Readonly<{
  tiles: Record<string, number[]>;
  /** Chunks per staged tensor; a tensor taken in one step has no entry. */
  steps: Record<string, StepSpec>;
  /** `planReport` of this plan, evaluated on its own. */
  total: Work;
  unwritten: string[];
  /** The largest step payload among the plan's tiled tensors. */
  stepBytes: number;
  /** The largest accumulator, or null where no tiled tensor has one. */
  accumulatorBytes: number | null;
  perTensor: Record<string, { stepsPerTask: number; stepBytes: number; accumulatorBytes: number | null }>;
}>;

export type Sweep = Readonly<{
  budget: Budget;
  /** Not beaten on recomputed FLOPs, bytes read and written, task count and
   * step payload; ordered by those, in that order. */
  rows: readonly SweepRow[];
  counts: Readonly<{
    /** Sets of tiled tensors searched, and how many there are in all. */
    fusionSets: number;
    fusionSetsTotal: number;
    /** Sets not searched because a figure in them is not exact. */
    setsInexact: number;
    /** Sets not searched because `maxPlans` was already reached. */
    setsSkipped: number;
    /** Tile extents measured, over all tensors and sets. */
    measured: number;
    /** Tile extents left unmeasured: more tasks than the task budget. */
    overTasks: number;
    /** Tile extents with no chunking within the budget. */
    overBudget: number;
    /** Tile extents whose chunkings were not all tried: more than
     * `MAX_CHUNKINGS`, so the smallest chunks of each axis were left out. */
    chunkingsCapped: number;
    /** Combinations formed, and combinations not formed past `maxPlans`. */
    combined: number;
    combinationsSkipped: number;
    /** Unbeaten combinations, of which `rows` holds the first `maxRows`. */
    unbeaten: number;
  }>;
}>;

/** Powers of two below an extent, and the extent itself. */
export function extentOptions(extent: number): number[] {
  const out: number[] = [];
  for (let e = 1; e < extent; e *= 2) out.push(e);
  out.push(extent);
  return out;
}

function product<T>(lists: readonly (readonly T[])[], cap = Infinity): T[][] {
  let out: T[][] = [[]];
  for (const list of lists) {
    const next: T[][] = [];
    for (const head of out)
      for (const item of list) {
        if (next.length >= cap) break;
        next.push([...head, item]);
      }
    out = next;
  }
  return out;
}

/** `a` beats `b`: no worse on any key, and better on one of the first
 * `strict` keys. */
function beats(a: readonly number[], b: readonly number[], strict: number): boolean {
  let better = false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > b[i]) return false;
    if (a[i] < b[i] && i < strict) better = true;
  }
  return better;
}

function unbeaten<T>(items: readonly T[], keys: (item: T) => readonly number[], strict = Infinity): T[] {
  const keyed = items.map((item) => ({ item, key: keys(item) }));
  return keyed.filter((a) => !keyed.some((b) => b !== a && beats(b.key, a.key, strict))).map((a) => a.item);
}

type Option = {
  tile: number[];
  spec: StepSpec | null;
  stepsPerTask: number;
  stepBytes: number;
  accumulatorBytes: number | null;
  tasks: number;
  flops: number;
  read: number;
};

type Counts = { measured: number; overTasks: number; overBudget: number; chunkingsCapped: number };

/** Thrown out of a measurement when a figure is not exact; the set is then
 * not searched. */
class Inexact extends Error {}

const exactly = (f: Figure): number => {
  if (f.status !== "exact") throw new Inexact();
  return f.value;
};

/** Every chunking of the reduced axes, fewest steps first. Past the cap the
 * smallest chunks of every axis are left out, so the whole axis always stays. */
function chunkingsOf(reduced: Staged["reduced"], counts: Counts): { chunks: number[]; steps: number }[] {
  const perAxis = reduced.map((axis) => extentOptions(axis.extent).reverse());
  let capped = false;
  while (perAxis.reduce((n, list) => n * list.length, 1) > MAX_CHUNKINGS) {
    const longest = perAxis.reduce((best, list, i) => (list.length > perAxis[best].length ? i : best), 0);
    perAxis[longest].pop();
    capped = true;
  }
  if (capped) counts.chunkingsCapped++;
  return product(perAxis)
    .map((chunks) => ({
      chunks,
      steps: chunks.reduce((n, c, i) => n * Math.ceil(reduced[i].extent / c), 1),
    }))
    .sort((a, b) => a.steps - b.steps);
}

/**
 * One tile extent of one tensor, with the other tiled tensors at their full
 * extent - which they do not affect. Null when it is not a candidate, with the
 * reason counted. Throws `Inexact` on a figure that is not exact.
 */
function measure(
  graph: ResolvedGraph,
  others: Record<string, number[]>,
  tensorId: string,
  tile: number[],
  budget: Budget,
  taskBudget: number,
  counts: Counts
): Option | null {
  const tiles = { ...others, [tensorId]: tile };
  const plan = tilePlan(graph, tiles);
  const family = plan.families.get(tensorId)!;
  if (family.count > taskBudget) {
    counts.overTasks++;
    return null;
  }
  counts.measured++;
  const staging = stagingOf(plan, tensorId);
  const tasks: TaskRef[] = [...tilesOf(family)].map((coord) => ({ tensorId, coord }));

  // One pass over the tasks: the family's sums, and what is the largest over
  // tasks - the whole demand of a task without steps, the accumulator of one
  // with them.
  let flops = 0;
  let read = 0;
  let wholeBytes = 0;
  let accumulatorBytes = 0;
  for (const task of tasks) {
    const supply = taskSupplyOf(plan, task);
    flops += exactly(supply.flops);
    const bytes = exactly(payloadOf(plan, supply.demand).bytes);
    read += bytes;
    wholeBytes = Math.max(wholeBytes, bytes);
    if (staging.staged) {
      const steps = taskSteps(plan, task);
      if ("staging" in steps) accumulatorBytes = Math.max(accumulatorBytes, exactly(steps.accumulator.bytes));
    }
  }

  let chosen: { spec: StepSpec | null; stepsPerTask: number; stepBytes: number; accumulatorBytes: number | null } | null = null;
  if (!staging.staged) {
    if (wholeBytes <= budget.stepBytes) chosen = { spec: null, stepsPerTask: 1, stepBytes: wholeBytes, accumulatorBytes: null };
  } else if (budget.accumulatorBytes === undefined || accumulatorBytes <= budget.accumulatorBytes) {
    const { reduced } = staging;
    // Where every operand of the contraction is read directly, each a different
    // tensor addressed by distinct loop axes, a step reads one box of each and
    // the box's size is the product of the step's interval lengths, wherever
    // the step sits. The first task's first step has the longest interval on
    // every axis - a tail is shorter - so it is the largest, and it alone is
    // measured. Anything else between a read and the contraction, a tensor in
    // two slots or a diagonal makes position matter, and every step is measured.
    const operands = nodeById(graph, staging.node)!.inputs;
    const positionFree =
      new Set(operands).size === operands.length &&
      operands.every((id) => !graph.tensors[id].producer || plan.families.has(id)) &&
      staging.domain.inputs.every((access) => new Set(access).size === access.length);
    const probed = positionFree ? tasks.slice(0, 1) : tasks;
    search: for (const { chunks, steps } of chunkingsOf(reduced, counts)) {
      const whole = chunks.every((c, i) => c === reduced[i].extent);
      const spec: StepSpec | null = whole
        ? null
        : { chunk: Object.fromEntries(reduced.flatMap((axis, i) => (chunks[i] === axis.extent ? [] : [[axis.label, chunks[i]]]))) };
      const stagedPlan = spec ? tilePlan(graph, tiles, { [tensorId]: spec }) : plan;
      // The largest payload over every task and step; the first one over the
      // budget rules the chunking out.
      let stepBytes = 0;
      for (const task of probed) {
        const taskOf = taskSteps(stagedPlan, task);
        if (!("staging" in taskOf)) continue search;
        for (let index = 0; index < (positionFree ? 1 : steps); index++) {
          stepBytes = Math.max(stepBytes, exactly(stepPayload(stagedPlan, stepOf(stagedPlan, taskOf, index)).bytes));
          if (stepBytes > budget.stepBytes) continue search;
        }
      }
      chosen = { spec, stepsPerTask: steps, stepBytes, accumulatorBytes };
      break;
    }
  }
  if (!chosen) {
    counts.overBudget++;
    return null;
  }
  return { tile, ...chosen, tasks: family.count, flops, read };
}

/** The sets of tiled tensors searched by default, and how many exist. */
function fusionSets(graph: ResolvedGraph): { sets: string[][]; total: number } {
  const outputs = graphOutputs(graph).map((t) => t.id).sort();
  const { views } = viewLayouts(graph);
  const candidates = Object.values(graph.tensors)
    .filter((t) => t.producer && !outputs.includes(t.id) && !views.has(t.producer.nodeId))
    .map((t) => t.id)
    .sort();
  const total = 2 ** candidates.length;
  const subsets: string[][] =
    candidates.length <= ALL_FUSIONS_UP_TO
      ? product(candidates.map((id) => [[], [id]])).map((parts) => parts.flat())
      : [[], ...candidates.map((id) => [id]), candidates];
  return { sets: subsets.map((subset) => [...outputs, ...subset].sort()), total };
}

type Candidate = {
  tiles: Record<string, number[]>;
  steps: Record<string, StepSpec>;
  options: Record<string, Option>;
  key: [recomputed: number, moved: number, tasks: number, stepBytes: number];
};

export function sweepPlans(graph: ResolvedGraph, options: SweepOptions): Sweep {
  const { budget } = options;
  const maxPlans = options.maxPlans ?? DEFAULT_MAX_PLANS;
  const maxRows = options.maxRows ?? DEFAULT_MAX_ROWS;
  const taskBudget = options.taskBudget ?? SWEEP_TASK_BUDGET;
  for (const [name, n] of [
    ["step budget", budget.stepBytes],
    ["accumulator budget", budget.accumulatorBytes ?? 1],
    ["plan limit", maxPlans],
    ["row limit", maxRows],
    ["task budget", taskBudget],
  ] as const)
    if (!(n >= 1) || !Number.isFinite(n)) throw new PlanError("PLAN_BUDGET", `the ${name} must be a positive number`);

  // A given set is checked as a tiling, each tensor once, and a set given
  // twice is searched once.
  let searched: { sets: string[][]; total: number };
  if (options.tiled) {
    const seen = new Set<string>();
    const sets: string[][] = [];
    for (const given of options.tiled) {
      const set = [...new Set(given)].sort();
      for (const id of set)
        if (!graph.tensors[id]) throw new PlanError("PLAN_UNKNOWN_TENSOR", `unknown tensor "${id}"`);
      tilePlan(graph, Object.fromEntries(set.map((id) => [id, [...graph.tensors[id].resolved]])));
      const key = set.join("\u0000");
      if (set.length && !seen.has(key)) {
        seen.add(key);
        sets.push(set);
      }
    }
    searched = { sets, total: sets.length };
  } else {
    const all = fusionSets(graph);
    searched = { sets: all.sets.filter((set) => set.length), total: all.total };
  }

  const counts: Counts = { measured: 0, overTasks: 0, overBudget: 0, chunkingsCapped: 0 };
  const candidates: Candidate[] = [];
  let combined = 0;
  let combinationsSkipped = 0;
  let setsSearched = 0;
  let setsInexact = 0;
  let setsSkipped = 0;

  for (const set of searched.sets) {
    if (combined >= maxPlans) {
      setsSkipped++;
      continue;
    }
    const full = Object.fromEntries(set.map((id) => [id, [...graph.tensors[id].resolved]]));
    const fronts: Option[][] = [];
    let once: number;
    let written: number;
    try {
      // What the set fixes whatever the tile extents: the work done once over
      // everything computed, and the bytes written.
      const base = planReport(tilePlan(graph, full), { budget: taskBudget }).total;
      if (!base) throw new Inexact();
      once = exactly(base.flops) - exactly(base.recomputed);
      written = exactly(base.written);
      for (const tensorId of set) {
        const measured: Option[] = [];
        for (const tile of product(graph.tensors[tensorId].resolved.map(extentOptions))) {
          const option = measure(graph, full, tensorId, tile, budget, taskBudget, counts);
          if (option) measured.push(option);
        }
        fronts.push(unbeaten(measured, (o) => [o.tasks, o.flops, o.read, o.stepBytes], 3));
      }
    } catch (e) {
      if (!(e instanceof Inexact)) throw e;
      setsInexact++;
      continue;
    }
    setsSearched++;
    const all = fronts.reduce((n, front) => n * front.length, 1);
    const combos = product(fronts, maxPlans - combined);
    combined += combos.length;
    combinationsSkipped += all - combos.length;

    for (const combo of combos) {
      const byTensor = Object.fromEntries(set.map((id, i) => [id, combo[i]]));
      const flops = combo.reduce((n, o) => n + o.flops, 0);
      candidates.push({
        tiles: Object.fromEntries(set.map((id, i) => [id, combo[i].tile])),
        steps: Object.fromEntries(set.flatMap((id, i) => (combo[i].spec ? [[id, combo[i].spec!]] : []))),
        options: byTensor,
        key: [
          flops - once,
          combo.reduce((n, o) => n + o.read, 0) + written,
          combo.reduce((n, o) => n + o.tasks, 0),
          Math.max(...combo.map((o) => o.stepBytes)),
        ],
      });
    }
  }

  const front = unbeaten(candidates, (c) => c.key).sort((a, b) => {
    for (let i = 0; i < a.key.length; i++) if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i];
    return JSON.stringify(a.tiles) < JSON.stringify(b.tiles) ? -1 : 1;
  });

  const rows: SweepRow[] = [];
  for (const candidate of front.slice(0, maxRows)) {
    const report = planReport(tilePlan(graph, candidate.tiles, candidate.steps), { budget: taskBudget });
    if (!report.total) continue;
    const accumulators = Object.values(candidate.options).flatMap((o) => (o.accumulatorBytes === null ? [] : [o.accumulatorBytes]));
    rows.push({
      tiles: candidate.tiles,
      steps: candidate.steps,
      total: report.total,
      unwritten: report.unwritten,
      stepBytes: candidate.key[3],
      accumulatorBytes: accumulators.length ? Math.max(...accumulators) : null,
      perTensor: Object.fromEntries(
        Object.entries(candidate.options).map(([id, o]) => [
          id,
          { stepsPerTask: o.stepsPerTask, stepBytes: o.stepBytes, accumulatorBytes: o.accumulatorBytes },
        ])
      ),
    });
  }

  return {
    budget,
    rows,
    counts: {
      fusionSets: setsSearched,
      fusionSetsTotal: searched.total,
      setsInexact,
      setsSkipped,
      ...counts,
      combined,
      combinationsSkipped,
      unbeaten: front.length,
    },
  };
}
