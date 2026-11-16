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
 * time, and candidates are built up one tensor at a time from the options that
 * survive a prune. An option, or a partial combination, is pruned when another
 * over the same tensors is no worse on task count, FLOPs, bytes read and
 * payload, and better on one of the three sums: swapping it out then improves
 * a sum and worsens nothing, so no unbeaten plan contains it. Being better on
 * payload alone is not enough, since a plan's payload is a maximum and another
 * tensor can set it. The rows returned are then evaluated as whole plans by
 * `planReport`, those are the figures reported, and a row whose report
 * disagrees with the figures it was ranked by is an error, not a row.
 *
 * **Exact sets only.** The argument needs every measured figure to be exact.
 * Where a demand is over-approximated - a strided access past its enumeration
 * limit, a wide diagonal, a barrier - the union of the tasks' regions is not
 * the region of the whole, and the work a set does once is no longer one
 * number. Such a set is not searched, and is counted in `setsInexact`.
 *
 * **Chunks.** For one tile extent, plan costs do not depend on the chunks; only
 * the payload and the step count do. Each tile takes the chunking with the
 * fewest steps whose payload is within the budget, and among several with as
 * few, the smallest payload. A tile with none is over the budget. A tensor
 * without steps is one step per task: its payload is the task's whole demand.
 *
 * **What is measured.** A payload is the largest over every task of the family
 * and every step of each, and an accumulator the largest over every task. No
 * task stands for another in general: a border task of a padded operation
 * reads less than an interior one, and a step that reaches a concatenated
 * operand reads more than the step before it. The one case measured on a
 * single step is the contraction whose operands are all read directly
 * (`measure`), where a step's payload is a product of its interval lengths.
 * The payload is the contraction's own reads per step; what a task's epilogue
 * reads once - a bias, a residual operand - is not part of it.
 *
 * **Limits.** Every limit that leaves something out is counted in the result:
 * tile extents with too many tasks or too many steps to measure, chunk
 * searches that were capped, sets not reached within the work limit, and
 * partial combinations dropped past `maxPlans`. When `combinationsDropped` is
 * not zero the rows are unbeaten among the combinations kept, not among all.
 */

import { graphOutputs, nodeById, type ResolvedGraph } from "../graph";
import { type Figure, viewLayouts } from "../metrics";
import type { BoundedCone } from "../propagate";
import { planReport, taskSupplyFrom, type Work } from "./interfaces";
import { type StepSpec, type TaskRef, taskDemand, tilePlan } from "./plan";
import { payloadOf, type Staged, stagingOf, stepOf, stepPayload, taskSteps } from "./steps";
import { PlanError, tiles as tilesOf } from "./tile-family";

export type Budget = Readonly<{
  /** Bytes a task may hold of its operands at a time: the limit on what one
   * step of its contraction reads. Epilogue reads are not counted. */
  stepBytes: number;
  /** Bytes an accumulator may take, when the caller bounds it. */
  accumulatorBytes?: number;
}>;

export type SweepOptions = Readonly<{
  budget: Budget;
  /**
   * The sets of tiled tensors to consider; an empty set is ignored. Left out:
   * every graph output tiled, and each computed intermediate that is not a
   * view either tiled or fused - all combinations for up to
   * `ALL_FUSIONS_UP_TO` such intermediates, and otherwise none of them, each
   * alone, and all of them.
   */
  tiled?: readonly (readonly string[])[];
  /** Partial combinations kept per set while building candidates. */
  maxPlans?: number;
  /** Rows evaluated as whole plans and returned. */
  maxRows?: number;
  /** Tasks in one family; a tile extent with more is not measured. */
  taskBudget?: number;
  /** Step evaluations for one chunking of one tile extent, where every step
   * must be measured; a chunking needing more is not tried. */
  stepWork?: number;
  /** Candidate visits plus task and step evaluations in all, each counted as
   * the number of operations in the graph, since a cone visits each; sets not
   * reached are not searched. Rejected candidates also consume work. */
  maxWork?: number;
}>;

export const ALL_FUSIONS_UP_TO = 4;
export const DEFAULT_MAX_PLANS = 4096;
export const DEFAULT_MAX_ROWS = 64;
/**
 * Tasks in one family, by default. Every tile extent of every tensor is
 * measured task by task, so this bounds the sweep's work. Finer tilings are
 * counted in `overTasks`, not silently dropped.
 */
export const SWEEP_TASK_BUDGET = 256;
export const DEFAULT_STEP_WORK = 4096;
export const DEFAULT_MAX_WORK = 200_000;
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
  perTensor: Record<
    string,
    {
      stepsPerTask: number;
      stepBytes: number;
      accumulatorBytes: number | null;
      /** The chunking was chosen from a capped search: one with fewer steps
       * may exist among the chunkings left out. */
      chunkingCapped: boolean;
    }
  >;
}>;

export type Sweep = Readonly<{
  budget: Budget;
  /** Not beaten on recomputed FLOPs, bytes read and written, task count and
   * step payload; ordered by those, in that order. */
  rows: readonly SweepRow[];
  /** Counts of tile extents are over the sets that were searched. */
  counts: Readonly<{
    /** Sets of tiled tensors searched, and how many there are in all. */
    fusionSets: number;
    fusionSetsTotal: number;
    /** Sets not searched because a figure in them is not exact. */
    setsInexact: number;
    /** Sets not searched because the work limit was reached. */
    setsSkipped: number;
    /** Tile extents measured, over all tensors and sets. */
    measured: number;
    /** Tile extents left unmeasured: more tasks than the task budget. */
    overTasks: number;
    /** Tile extents with no chunking within the budget. */
    overBudget: number;
    /** Tile extents whose fitting chunkings all need more step evaluations
     * than `stepWork`: within the budget or not is not known. */
    overSteps: number;
    /** Tile extents whose chunk search was capped at `MAX_CHUNKINGS`, the
     * smallest chunks of each axis left out. */
    chunkingsCapped: number;
    /** Of those, extents where only a left-out chunking fits. They are not
     * candidates, and are not over the budget either. */
    cappedUnresolved: number;
    /** Partial combinations dropped past `maxPlans` while building candidates. */
    combinationsDropped: number;
    /** Unbeaten candidates, of which `rows` holds the first `maxRows`. */
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

function product<T>(lists: readonly (readonly T[])[]): T[][] {
  let out: T[][] = [[]];
  for (const list of lists) out = out.flatMap((head) => list.map((item) => [...head, item]));
  return out;
}

/** Tile choices can grow exponentially with rank. Yield them in product order
 * so the work limit can stop the search without allocating the whole space. */
function* tileChoices(lists: readonly (readonly number[])[]): Generator<number[]> {
  if (lists.some((list) => !list.length)) return;
  const indices = lists.map(() => 0);
  while (true) {
    yield lists.map((list, axis) => list[indices[axis]]);
    let axis = lists.length - 1;
    while (axis >= 0 && ++indices[axis] === lists[axis].length) {
      indices[axis] = 0;
      axis--;
    }
    if (axis < 0) return;
  }
}

type Key = readonly [tasks: number, flops: number, read: number, stepBytes: number];

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

const byKey = (a: readonly number[], b: readonly number[]): number => {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return 0;
};

type Option = {
  tile: number[];
  spec: StepSpec | null;
  stepsPerTask: number;
  stepBytes: number;
  accumulatorBytes: number | null;
  chunkingCapped: boolean;
  tasks: number;
  flops: number;
  read: number;
};

type ExtentCounts = {
  measured: number;
  overTasks: number;
  overBudget: number;
  overSteps: number;
  chunkingsCapped: number;
  cappedUnresolved: number;
};
const noCounts = (): ExtentCounts => ({
  measured: 0,
  overTasks: 0,
  overBudget: 0,
  overSteps: 0,
  chunkingsCapped: 0,
  cappedUnresolved: 0,
});

/** Thrown out of a measurement when a figure is not exact; the set is then
 * not searched. */
class Inexact extends Error {}
/** Thrown when the sweep's work limit is reached. */
class OutOfWork extends Error {}

const exactly = (f: Figure): number => {
  if (f.status !== "exact") throw new Inexact();
  return f.value;
};

/** Every chunking of the reduced axes, fewest steps first. Past the cap the
 * smallest chunks of every axis are left out, so the whole axis always stays. */
function chunkingsOf(reduced: Staged["reduced"]): { list: { chunks: number[]; steps: number }[]; capped: boolean } {
  const perAxis = reduced.map((axis) => extentOptions(axis.extent).reverse());
  let capped = false;
  while (perAxis.reduce((n, list) => n * list.length, 1) > MAX_CHUNKINGS) {
    const longest = perAxis.reduce((best, list, i) => (list.length > perAxis[best].length ? i : best), 0);
    perAxis[longest].pop();
    capped = true;
  }
  const list = product(perAxis)
    .map((chunks) => ({
      chunks,
      steps: chunks.reduce((n, c, i) => n * Math.ceil(reduced[i].extent / c), 1),
    }))
    .sort((a, b) => a.steps - b.steps);
  return { list, capped };
}

/** `weight` is what one task or step evaluation costs in work units: the
 * number of operations in the graph, since a cone visits each. */
type Limits = { taskBudget: number; stepWork: number; weight: number; work: { left: number } };

/**
 * One tile extent of one tensor, with the other tiled tensors at their full
 * extent - which they do not affect. Null when it is not a candidate, with the
 * reason counted. Throws `Inexact` on a figure that is not exact, and
 * `OutOfWork` at the work limit.
 */
function measure(
  graph: ResolvedGraph,
  others: Record<string, number[]>,
  tensorId: string,
  tile: number[],
  budget: Budget,
  limits: Limits,
  counts: ExtentCounts
): Option | null {
  const spend = (n: number) => {
    limits.work.left -= n * limits.weight;
    if (limits.work.left < 0) throw new OutOfWork();
  };
  // Even a candidate rejected before measurement costs a visit. Otherwise a
  // high-rank family can walk millions of rejected tiles without spending work.
  spend(1);
  const tiles = { ...others, [tensorId]: tile };
  const plan = tilePlan(graph, tiles);
  const family = plan.families.get(tensorId)!;
  if (family.count > limits.taskBudget) {
    counts.overTasks++;
    return null;
  }
  spend(family.count);
  counts.measured++;
  const staging = stagingOf(plan, tensorId);
  const tasks: TaskRef[] = [...tilesOf(family)].map((coord) => ({ tensorId, coord }));

  // One pass over the tasks: the family's sums, and what is the largest over
  // tasks - the whole demand of a task without steps, the accumulator of one
  // with them. Each task's cone is computed once and kept for the chunk search.
  const cones: BoundedCone[] = [];
  let flops = 0;
  let read = 0;
  let wholeBytes = 0;
  let accumulatorBytes = 0;
  for (const task of tasks) {
    const cone = taskDemand(plan, task);
    cones.push(cone);
    const supply = taskSupplyFrom(plan, task, cone);
    flops += exactly(supply.flops);
    const bytes = exactly(payloadOf(plan, supply.demand).bytes);
    read += bytes;
    wholeBytes = Math.max(wholeBytes, bytes);
    if (staging.staged) {
      const steps = taskSteps(plan, task, cone);
      if ("staging" in steps) accumulatorBytes = Math.max(accumulatorBytes, exactly(steps.accumulator.bytes));
    }
  }

  type Chosen = Pick<Option, "spec" | "stepsPerTask" | "stepBytes" | "accumulatorBytes" | "chunkingCapped">;
  let chosen: Chosen | null = null;
  if (!staging.staged) {
    if (wholeBytes <= budget.stepBytes)
      chosen = { spec: null, stepsPerTask: 1, stepBytes: wholeBytes, accumulatorBytes: null, chunkingCapped: false };
    else counts.overBudget++;
  } else if (budget.accumulatorBytes !== undefined && accumulatorBytes > budget.accumulatorBytes) {
    counts.overBudget++;
  } else {
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
    const probed = positionFree ? [0] : tasks.map((_, i) => i);

    const specOf = (chunks: number[]): StepSpec | null =>
      chunks.every((c, i) => c === reduced[i].extent)
        ? null
        : { chunk: Object.fromEntries(reduced.flatMap((axis, i) => (chunks[i] === axis.extent ? [] : [[axis.label, chunks[i]]]))) };
    /** The largest payload of a chunking over every probed task and step, or
     * null as soon as one is over the budget. */
    const payloadWithin = (spec: StepSpec | null, steps: number): number | null => {
      const stagedPlan = spec ? tilePlan(graph, tiles, { [tensorId]: spec }) : plan;
      let largest = 0;
      for (const i of probed) {
        const taskOf = taskSteps(stagedPlan, tasks[i], cones[i]);
        if (!("staging" in taskOf)) throw new Error(`"${tensorId}" lost its steps under a chunking`);
        for (let index = 0; index < (positionFree ? 1 : steps); index++) {
          spend(1);
          largest = Math.max(largest, exactly(stepPayload(stagedPlan, stepOf(stagedPlan, taskOf, index)).bytes));
          if (largest > budget.stepBytes) return null;
        }
      }
      return largest;
    };

    const { list, capped } = chunkingsOf(reduced);
    if (capped) counts.chunkingsCapped++;
    // Fewest steps first; among the chunkings with as few steps as the first
    // that fits, the smallest payload, then the order tried.
    let tooMuchWork = false;
    let best: { spec: StepSpec | null; steps: number; stepBytes: number } | null = null;
    for (const { chunks, steps } of list) {
      if (best && steps > best.steps) break;
      if (!positionFree && tasks.length * steps > limits.stepWork) {
        // Later chunkings have at least as many steps, so none can be tried.
        tooMuchWork = true;
        break;
      }
      const spec = specOf(chunks);
      const stepBytes = payloadWithin(spec, steps);
      if (stepBytes !== null && (!best || stepBytes < best.stepBytes)) best = { spec, steps, stepBytes };
    }
    if (best) {
      chosen = { spec: best.spec, stepsPerTask: best.steps, stepBytes: best.stepBytes, accumulatorBytes, chunkingCapped: capped };
    } else if (tooMuchWork) {
      counts.overSteps++;
    } else if (capped) {
      // Nothing tried fits. The smallest payload of all is one element per
      // reduced axis, which the cap may have left out: if that fits, the
      // extent is within the budget by a chunking this search did not reach.
      const ones = reduced.map(() => 1);
      const steps = reduced.reduce((n, axis) => n * axis.extent, 1);
      if (!positionFree && tasks.length * steps > limits.stepWork) counts.overSteps++;
      else if (payloadWithin(specOf(ones), steps) !== null) counts.cappedUnresolved++;
      else counts.overBudget++;
    } else counts.overBudget++;
  }
  if (!chosen) return null;
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

type Partial = { options: Option[]; key: Key };

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
  const stepWork = options.stepWork ?? DEFAULT_STEP_WORK;
  const maxWork = options.maxWork ?? DEFAULT_MAX_WORK;
  for (const [name, n] of [["step budget", budget.stepBytes], ["accumulator budget", budget.accumulatorBytes ?? 1]] as const)
    if (!(n >= 1) || !Number.isFinite(n)) throw new PlanError("PLAN_BUDGET", `the ${name} must be a positive number`);
  for (const [name, n] of [["plan limit", maxPlans], ["row limit", maxRows], ["task budget", taskBudget], ["step work limit", stepWork], ["work limit", maxWork]] as const)
    if (!Number.isSafeInteger(n) || n < 1) throw new PlanError("PLAN_BUDGET", `the ${name} must be a positive integer`);

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

  const counts = noCounts();
  const limits: Limits = { taskBudget, stepWork, weight: Math.max(1, graph.topo.length), work: { left: maxWork } };
  const candidates: Candidate[] = [];
  let combinationsDropped = 0;
  let setsSearched = 0;
  let setsInexact = 0;
  let setsSkipped = 0;

  for (let s = 0; s < searched.sets.length; s++) {
    const set = searched.sets[s];
    const full = Object.fromEntries(set.map((id) => [id, [...graph.tensors[id].resolved]]));
    // Counted for this set only once it has been searched: a set dropped as
    // inexact or unfinished leaves no extents in the totals.
    const local = noCounts();
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
        for (const tile of tileChoices(graph.tensors[tensorId].resolved.map(extentOptions))) {
          const option = measure(graph, full, tensorId, tile, budget, limits, local);
          if (option) measured.push(option);
        }
        fronts.push(unbeaten(measured, (o) => [o.tasks, o.flops, o.read, o.stepBytes], 3));
      }
    } catch (e) {
      if (e instanceof Inexact) {
        setsInexact++;
        continue;
      }
      if (e instanceof OutOfWork) {
        // This set and every one after it.
        setsSkipped += searched.sets.length - s;
        break;
      }
      throw e;
    }
    setsSearched++;
    for (const name of Object.keys(local) as (keyof ExtentCounts)[]) counts[name] += local[name];

    // Build combinations one tensor at a time, pruning partial ones by the
    // same rule as options. Past `maxPlans` the best by key are kept.
    let partials: Partial[] = [{ options: [], key: [0, 0, 0, 0] }];
    for (const front of fronts) {
      const grown: Partial[] = [];
      for (const partial of partials)
        for (const option of front)
          grown.push({
            options: [...partial.options, option],
            key: [
              partial.key[0] + option.tasks,
              partial.key[1] + option.flops,
              partial.key[2] + option.read,
              Math.max(partial.key[3], option.stepBytes),
            ],
          });
      // Sorted first so the quadratic prune runs on a bounded number.
      grown.sort((a, b) => byKey(a.key, b.key));
      const bounded = grown.slice(0, 4 * maxPlans);
      const kept = unbeaten(bounded, (p) => p.key, 3);
      partials = kept.slice(0, maxPlans);
      combinationsDropped += grown.length - bounded.length + (kept.length - partials.length);
    }

    for (const { options: combo } of partials) {
      if (combo.length !== set.length) continue; // a tensor with no option: no plan
      const flops = combo.reduce((n, o) => n + o.flops, 0);
      candidates.push({
        tiles: Object.fromEntries(set.map((id, i) => [id, combo[i].tile])),
        steps: Object.fromEntries(set.flatMap((id, i) => (combo[i].spec ? [[id, combo[i].spec!]] : []))),
        options: Object.fromEntries(set.map((id, i) => [id, combo[i]])),
        key: [
          flops - once,
          combo.reduce((n, o) => n + o.read, 0) + written,
          combo.reduce((n, o) => n + o.tasks, 0),
          Math.max(...combo.map((o) => o.stepBytes)),
        ],
      });
    }
  }

  const front = unbeaten(candidates, (c) => c.key).sort(
    (a, b) => byKey(a.key, b.key) || (JSON.stringify(a.tiles) < JSON.stringify(b.tiles) ? -1 : 1)
  );

  const rows: SweepRow[] = [];
  for (const candidate of front.slice(0, maxRows)) {
    const report = planReport(tilePlan(graph, candidate.tiles, candidate.steps), { budget: taskBudget });
    const total = report.total;
    // The figures a row is ranked by were added up per tensor; the plan's own
    // report is the check on that. A disagreement is a fault in the sweep.
    if (
      !total ||
      total.recomputed.status !== "exact" ||
      total.recomputed.value !== candidate.key[0] ||
      exactly(total.read) + exactly(total.written) !== candidate.key[1] ||
      total.tasks !== candidate.key[2]
    )
      throw new Error(`the sweep ranked plan ${JSON.stringify(candidate.tiles)} by figures its report does not have`);
    const accumulators = Object.values(candidate.options).flatMap((o) => (o.accumulatorBytes === null ? [] : [o.accumulatorBytes]));
    rows.push({
      tiles: candidate.tiles,
      steps: candidate.steps,
      total,
      unwritten: report.unwritten,
      stepBytes: candidate.key[3],
      accumulatorBytes: accumulators.length ? Math.max(...accumulators) : null,
      perTensor: Object.fromEntries(
        Object.entries(candidate.options).map(([id, o]) => [
          id,
          {
            stepsPerTask: o.stepsPerTask,
            stepBytes: o.stepBytes,
            accumulatorBytes: o.accumulatorBytes,
            chunkingCapped: o.chunkingCapped,
          },
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
      combinationsDropped,
      unbeaten: front.length,
    },
  };
}
