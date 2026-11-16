/**
 * The steps of a task: its reduction, divided along the reduced loop axes.
 *
 * A task computes one tile of a tiled tensor. Where that tile is the result of
 * a contraction, the task's arithmetic is a box of the contraction's iteration
 * domain (`core/iteration.ts`): the tile on the output axes, every reduced axis
 * whole. A plan may divide the reduced axes into chunks, and a **step** is then
 * the tile on the output axes and one chunk on each reduced axis. Steps are
 * ordered by the declared loop order, last axis fastest.
 *
 * What is reported for a step is what any kernel running that step touches,
 * and nothing about how it holds it (docs/STAGING.md, the scope rule):
 *
 * - `reads`: what the step's terms read at the plan's frontier, per operation
 *   and operand slot, as a task's demand is reported;
 * - `needs`: the untiled tensors between those reads and the contraction, with
 *   the region of each the step's terms use;
 * - `flops`: the work of the step's own terms.
 *
 * A task has one **accumulator**, the contraction's result over the tile in the
 * accumulator dtype, and it is complete after the last step.
 *
 * The contraction need not produce the tiled tensor itself. Pointwise
 * operations between it and the tile - a bias, an activation, a cast - are the
 * task's **epilogue**: they apply to the finished accumulator, and what they
 * read is read once per task, not per step. Operations between a read and the
 * contraction's operands are followed by the bounded cone of each step's
 * operand region, so a prologue, a view and a fused contraction all report what
 * that step needs of them.
 *
 * A tensor whose tasks reach no contraction through pointwise operations has
 * no steps, and `stagingOf` says which operation is the reason. An equation
 * that reduces no axis - a transpose, an outer product - is not a contraction
 * here: it has nothing to divide.
 */

import { type DType, DTYPE_BYTES, dtypeFamily } from "../dtypes";
import { executeBoundedQuery } from "../executor";
import { nodeById, producerNode, type ResolvedGraph } from "../graph";
import {
  type IterationDomain,
  iterationBoxOfOutput,
  projectInput,
} from "../iteration";
import { byteFigure, figure, type Figure, sumFigures } from "../metrics";
import { getOp, opLabel } from "../ops/index";
import { limitsOf } from "../ops/limits";
import type { OpCtx, OpSpec } from "../ops/types";
import type { BoundedCone, Crossing } from "../propagate";
import { type Box, canonicalize, count, isEmpty, iv, type Region, sameBox, sortRegion, unionOf } from "../region";
import { familyOf, type StepSpec, type TaskRef, taskDemand, type TilePlan } from "./plan";
import { PlanError, tileBox, tiles as tilesOf } from "./tile-family";

/** One reduced loop axis of a staged tensor, as the plan divides it. */
export type ReducedAxis = Readonly<{
  /** Index into the domain's axes. */
  axis: number;
  label: string;
  extent: number;
  chunk: number;
  /** Chunks along the axis; the last is shorter where `chunk` does not divide. */
  count: number;
}>;

export type Staged = Readonly<{
  staged: true;
  tensorId: string;
  /** The contraction's node, and the tensor it produces. */
  node: string;
  output: string;
  /** The pointwise nodes between the contraction and the tiled tensor, from
   * the tiled tensor's producer back toward the contraction. */
  epilogue: readonly string[];
  domain: IterationDomain;
  /** In loop order, outermost first. */
  reduced: readonly ReducedAxis[];
  /** Steps per task. */
  steps: number;
  accumulator: DType;
}>;

export type Unstaged = Readonly<{ staged: false; tensorId: string; reason: string }>;
export type Staging = Staged | Unstaged;

type Internal = { staging: Staging; spec?: OpSpec; ctx?: OpCtx };
const stagingCache = new WeakMap<TilePlan, Map<string, Internal>>();

const fail = (message: string): never => {
  throw new PlanError("PLAN_STEPS", message);
};

function isComputedInTask(plan: TilePlan, tensorId: string): boolean {
  return Boolean(plan.graph.tensors[tensorId].producer) && !plan.families.has(tensorId);
}

/**
 * Find the contraction a tiled tensor's tasks perform: back from the tensor
 * through pointwise operations, following the one operand each of them takes
 * from inside the task. Stops with a reason at an operation that is neither
 * pointwise nor a contraction, at one with no operand computed in the task,
 * and at one with two.
 */
function findContraction(
  plan: TilePlan,
  tensorId: string
): { node: string; output: string; epilogue: string[] } | { reason: string } {
  const { graph } = plan;
  const name = (id: string) => graph.tensors[id].name ?? id;
  const epilogue: string[] = [];
  let tensor = tensorId;
  for (;;) {
    const node = producerNode(graph, tensor)!;
    const spec = getOp(node.op)!;
    if (spec.iteration) {
      const ctx = { inShapes: graph.shapesOf(node.inputs), outShapes: graph.shapesOf(node.outputs), attrs: node.attrs };
      // An equation that sums nothing - a transpose, an outer product - has
      // one term per result element and no reduction to divide.
      if (spec.iteration(ctx).axes.some((axis) => axis.role === "reduced"))
        return { node: node.id, output: tensor, epilogue };
      return { reason: `${name(tensor)} is computed by ${opLabel(node)}, which reduces no axis` };
    }
    if (!spec.pointwise)
      return { reason: `${name(tensor)} is computed by ${opLabel(node)}, which is not a contraction` };
    const computed = [...new Set(node.inputs.filter((id) => isComputedInTask(plan, id)))];
    if (computed.length === 0)
      return { reason: `no contraction is fused into the tasks of ${name(tensorId)}` };
    if (computed.length > 1)
      return {
        reason: `${opLabel(node)} combines ${computed.map(name).join(" and ")}, each computed in the task`,
      };
    epilogue.push(node.id);
    tensor = computed[0];
  }
}

function buildStaging(plan: TilePlan, tensorId: string): Internal {
  const { graph } = plan;
  const name = (id: string) => graph.tensors[id].name ?? id;
  if (!plan.families.has(tensorId))
    throw new PlanError("PLAN_UNPLANNED", `the plan does not tile "${tensorId}"`);
  const declared: StepSpec | undefined = plan.steps.get(tensorId);
  const found = findContraction(plan, tensorId);
  if ("reason" in found) {
    if (
      declared &&
      (Object.keys(declared.chunk ?? {}).length || declared.order?.length || declared.accumulator)
    )
      fail(`"${tensorId}" has no steps to declare: ${found.reason}`);
    return { staging: { staged: false, tensorId, reason: found.reason } };
  }

  const node = nodeById(graph, found.node)!;
  const spec = getOp(node.op)!;
  const ctx: OpCtx = {
    inShapes: graph.shapesOf(node.inputs),
    outShapes: graph.shapesOf(node.outputs),
    attrs: node.attrs,
  };
  const domain = spec.iteration!(ctx);
  const reducedLabels = domain.axes.filter((a) => a.role === "reduced").map((a) => a.label);
  const listed = reducedLabels.length ? reducedLabels.join(", ") : "none";

  const order = declared?.order ?? reducedLabels;
  if (
    order.length !== reducedLabels.length ||
    new Set(order).size !== order.length ||
    order.some((label) => !reducedLabels.includes(label))
  )
    fail(
      `the step order of "${tensorId}" must name each reduced axis once (${listed}); got [${order.join(", ")}]`
    );

  const chunk = declared?.chunk ?? {};
  for (const [label, extent] of Object.entries(chunk)) {
    const axis = domain.axes.find((a) => a.label === label);
    if (!axis) fail(`"${tensorId}" has no loop axis "${label}"; its reduced axes are ${listed}`);
    else if (axis.role === "output")
      fail(`axis "${label}" of "${tensorId}" indexes the result, so its extent is the tile's, not a chunk`);
    else if (!Number.isInteger(extent) || extent < 1 || extent > axis.extent)
      fail(`chunk ${extent} on axis "${label}" of "${tensorId}" must be an integer from 1 to ${axis.extent}`);
  }

  const reduced: ReducedAxis[] = order.map((label) => {
    const axis = domain.axes.findIndex((a) => a.label === label);
    const extent = domain.axes[axis].extent;
    const size = chunk[label] ?? extent;
    return { axis, label, extent, chunk: size, count: Math.max(1, Math.ceil(extent / size)) };
  });
  const outDType = graph.tensors[found.output].dtype;
  const accumulator = declared?.accumulator ?? (dtypeFamily(outDType) === "float" ? "f32" : outDType);
  if (!Object.prototype.hasOwnProperty.call(DTYPE_BYTES, accumulator)) fail(`"${accumulator}" is not a dtype an accumulator can have`);
  else if (dtypeFamily(accumulator) !== dtypeFamily(outDType))
    fail(
      `the accumulator of "${tensorId}" is ${accumulator}, but ${name(found.output)} is ${outDType}: ` +
        `a running sum is held in the result's own kind of number`
    );

  return {
    spec,
    ctx,
    staging: {
      staged: true,
      tensorId,
      node: found.node,
      output: found.output,
      epilogue: found.epilogue,
      domain,
      reduced,
      steps: reduced.reduce((n, a) => n * a.count, 1),
      accumulator,
    },
  };
}

function internalOf(plan: TilePlan, tensorId: string): Internal {
  let byTensor = stagingCache.get(plan);
  if (!byTensor) stagingCache.set(plan, (byTensor = new Map()));
  let hit = byTensor.get(tensorId);
  if (!hit) byTensor.set(tensorId, (hit = buildStaging(plan, tensorId)));
  return hit;
}

/**
 * How the tasks of a tiled tensor divide their reduction, or why they have no
 * steps. Throws `PLAN_STEPS` when the plan declares steps the graph does not
 * support: an unknown or output axis, a chunk outside its axis, an order that
 * is not the reduced axes, an accumulator of another kind of number than the
 * result, or any declaration on a tensor with no contraction.
 */
export function stagingOf(plan: TilePlan, tensorId: string): Staging {
  return internalOf(plan, tensorId).staging;
}

/** Check every step declaration of a plan against its graph. */
export function checkSteps(plan: TilePlan): void {
  for (const tensorId of plan.steps.keys()) stagingOf(plan, tensorId);
}

/** What a step reads at the frontier through one operand slot. */
export type StepRead = Crossing;

export type TaskSteps = Readonly<{
  task: TaskRef;
  staging: Staged;
  /** The region of the contraction's result the tile needs: the output
   * intervals of every step's box. */
  outBox: Box;
  accumulator: Readonly<{ dtype: DType; elements: number; bytes: Figure }>;
  /** What the epilogue reads, once per task. */
  epilogue: readonly StepRead[];
}>;

/**
 * The step structure of one task, or the reason it has none. `cone` is the
 * task's demand where the caller already has it.
 */
export function taskSteps(plan: TilePlan, task: TaskRef, cone?: BoundedCone): TaskSteps | Unstaged {
  const { staging } = internalOf(plan, task.tensorId);
  if (!staging.staged) return staging;
  const demand = cone ?? taskDemand(plan, task);
  const tile = tileBox(familyOf(plan, task), task.coord);
  if (
    demand.seed.tensorId !== task.tensorId ||
    demand.seed.region.boxes.length !== 1 ||
    !sameBox(demand.seed.region.boxes[0], tile)
  )
    throw new Error(`the cone given for task [${task.coord.join(", ")}] of "${task.tensorId}" is another task's`);
  // The epilogue is a chain of pointwise operations from one tile, so what it
  // needs of the contraction's result is one box. Anything else is a fault in
  // the walk above, not in the plan.
  const region = demand.tensors.get(staging.output)?.region;
  if (!region || region.boxes.length !== 1 || !region.exact)
    throw new Error(
      `task [${task.coord.join(", ")}] of "${task.tensorId}" does not need one exact box of "${staging.output}"`
    );
  const outBox = region.boxes[0];
  const elements = count(region);
  const epilogueNodes = new Set(staging.epilogue);
  return {
    task,
    staging,
    outBox,
    accumulator: {
      dtype: staging.accumulator,
      elements,
      bytes: figure(elements * DTYPE_BYTES[staging.accumulator], "exact"),
    },
    epilogue: demand.crossings.filter((c) => epilogueNodes.has(c.node)),
  };
}

export type Step = Readonly<{
  index: number;
  /** The chunk taken on each reduced axis, parallel to `staging.reduced`. */
  coord: readonly number[];
  /** The step's box of the contraction's iteration domain. */
  box: Box;
  reads: readonly StepRead[];
  /** Untiled tensors the step's terms use, the contraction's operands among
   * them, each with the region used. */
  needs: readonly { tensorId: string; region: Region }[];
  /** The work of the step's own terms. Work on what it `needs` is not included:
   * two steps may need the same element, and the task counts it once. */
  flops: Figure;
}>;

/** The iteration box of step `index`: steps are numbered in loop order, the
 * last reduced axis fastest. */
function stepBox(steps: TaskSteps, index: number): { coord: number[]; box: Box } {
  const { staging } = steps;
  if (!Number.isInteger(index) || index < 0 || index >= staging.steps)
    fail(`"${staging.tensorId}" has ${staging.steps} steps per task; there is no step ${index}`);
  const coord = new Array<number>(staging.reduced.length);
  let rest = index;
  for (let i = staging.reduced.length - 1; i >= 0; i--) {
    coord[i] = rest % staging.reduced[i].count;
    rest = Math.floor(rest / staging.reduced[i].count);
  }
  const box = [...iterationBoxOfOutput(staging.domain, steps.outBox)];
  staging.reduced.forEach((axis, i) => {
    const lo = coord[i] * axis.chunk;
    box[axis.axis] = iv(lo, Math.min(lo + axis.chunk, axis.extent));
  });
  return { coord, box };
}

const readOrder = (a: StepRead, b: StepRead) =>
  a.tensorId.localeCompare(b.tensorId) || a.node.localeCompare(b.node) || a.slot - b.slot;

/** One step of a task. */
export function stepOf(plan: TilePlan, steps: TaskSteps, index: number): Step {
  const { staging } = steps;
  const { spec, ctx } = internalOf(plan, staging.tensorId);
  const graph: ResolvedGraph = plan.graph;
  const node = nodeById(graph, staging.node)!;
  const { coord, box } = stepBox(steps, index);
  const diagEnum = limitsOf(ctx!).diagEnum;

  // A tensor computed in the task is followed once, from everything the step's
  // terms use of it through any slot, so a read behind it is one row however
  // many slots it fills - as the task's own demand reports it.
  const reads: StepRead[] = [];
  const computed = new Map<string, Region[]>();
  node.inputs.forEach((tensorId, slot) => {
    const region = projectInput(staging.domain, slot, box, diagEnum);
    if (isEmpty(region)) return;
    if (isComputedInTask(plan, tensorId))
      computed.set(tensorId, [...(computed.get(tensorId) ?? []), region]);
    else reads.push({ node: node.id, slot, tensorId, region });
  });
  const needs = new Map<string, Region[]>();
  const behind = new Map<string, StepRead & { regions: Region[] }>();
  for (const [tensorId, regions] of computed) {
    const cone = executeBoundedQuery(graph, {
      tensorId,
      region: canonicalize(unionOf(regions)),
      direction: "backward",
      frontier: plan.frontier,
    });
    for (const crossing of cone.crossings) {
      const key = `${crossing.node}\u0000${crossing.slot}\u0000${crossing.tensorId}`;
      const row = behind.get(key);
      if (row) row.regions.push(crossing.region);
      else behind.set(key, { ...crossing, regions: [crossing.region] });
    }
    for (const [id, result] of cone.tensors)
      if (isComputedInTask(plan, id)) needs.set(id, [...(needs.get(id) ?? []), result.region]);
  }
  for (const { regions, ...row } of behind.values())
    reads.push(regions.length === 1 ? row : { ...row, region: sortRegion(canonicalize(unionOf(regions))) });

  return {
    index,
    coord,
    box,
    reads: reads.sort(readOrder),
    needs: [...needs.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([tensorId, regions]) => ({ tensorId, region: canonicalize(unionOf(regions)) })),
    flops: figure(spec!.flopsForIteration!(box, ctx!), "exact"),
  };
}

/** Every step of a task, in loop order. */
export function* stepsOf(plan: TilePlan, steps: TaskSteps): Generator<Step> {
  for (let index = 0; index < steps.staging.steps; index++) yield stepOf(plan, steps, index);
}

/** What a step reads of each tensor, each element once however many slots or
 * operations read it. */
export type StepPayload = Readonly<{
  tensors: readonly { tensorId: string; region: Region; bytes: Figure }[];
  /** Bytes the step reads at the frontier, summed over tensors. */
  bytes: Figure;
}>;

export function stepPayload(plan: TilePlan, step: Step): StepPayload {
  return payloadOf(plan, step.reads);
}

/** The same measure over any rows of reads: a task's whole demand, for one. */
export function payloadOf(
  plan: TilePlan,
  reads: readonly { tensorId: string; region: Region }[]
): StepPayload {
  const byTensor = new Map<string, Region[]>();
  for (const read of reads)
    byTensor.set(read.tensorId, [...(byTensor.get(read.tensorId) ?? []), read.region]);
  const tensors = [...byTensor.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([tensorId, regions]) => {
      const region = canonicalize(unionOf(regions));
      return { tensorId, region, bytes: byteFigure(plan.graph.tensors[tensorId], count(region), region) };
    });
  return { tensors, bytes: sumFigures(tensors.map((t) => t.bytes)) };
}

/**
 * Whether a step's payload is the same wherever the step sits, given its
 * interval lengths. True where every operand of the contraction is read
 * directly, each a different tensor addressed by distinct loop axes: a step
 * then reads one box of each, whose size is the product of the step's interval
 * lengths. The first task's first step has the longest interval on every axis -
 * a tail is shorter - so it is the largest of the family. Anything between a
 * read and the contraction, a tensor in two slots or a diagonal makes position
 * matter.
 */
export function stepsPositionFree(plan: TilePlan, staging: Staged): boolean {
  const operands = nodeById(plan.graph, staging.node)!.inputs;
  return (
    new Set(operands).size === operands.length &&
    operands.every((id) => !isComputedInTask(plan, id)) &&
    staging.domain.inputs.every((access) => new Set(access).size === access.length)
  );
}

/** The largest step of a family and its largest accumulator, or why they were
 * not measured. A tensor without steps is one step per task: the whole demand. */
export type FamilyPayload =
  | Readonly<{ status: "measured"; staged: boolean; stepsPerTask: number; stepBytes: Figure; accumulatorBytes: Figure | null }>
  | Readonly<{ status: "unmeasured"; staged: boolean; stepsPerTask: number; reason: string }>;

/** Task and step evaluations `familyPayload` spends before declining. */
export const FAMILY_PAYLOAD_WORK = 512;

const largest = (figures: readonly Figure[]): Figure => {
  if (figures.some((f) => f.value === null)) return figure(0, "unknown", figures.flatMap((f) => f.reasons));
  const value = Math.max(0, ...figures.map((f) => f.value!));
  // The largest of exact figures is exact; one upper bound among them makes it
  // an upper bound, since the true largest cannot exceed the largest bound.
  const exact = figures.every((f) => f.status === "exact");
  return figure(value, exact ? "exact" : "upper", figures.flatMap((f) => f.reasons));
};

/**
 * Measure a family's largest step over every task and step, within `work`
 * evaluations. Where position cannot matter (`stepsPositionFree`) one step is
 * measured; otherwise every one is, and a family needing more than `work` is
 * declined with the count, never sampled.
 */
export function familyPayload(plan: TilePlan, tensorId: string, work = FAMILY_PAYLOAD_WORK): FamilyPayload {
  const { staging } = internalOf(plan, tensorId);
  const family = plan.families.get(tensorId)!; // internalOf checks that the tensor is planned.
  if (family.count === 0) {
    const zero = figure(0, "exact");
    return {
      status: "measured",
      staged: staging.staged,
      stepsPerTask: staging.staged ? staging.steps : 1,
      stepBytes: zero,
      accumulatorBytes: staging.staged ? zero : null,
    };
  }
  const tasks = function* (): Generator<TaskRef> {
    for (const coord of tilesOf(family)) yield { tensorId, coord };
  };
  if (!staging.staged) {
    if (family.count > work)
      return { status: "unmeasured", staged: false, stepsPerTask: 1, reason: `${family.count} tasks to measure` };
    const each = [...tasks()].map((task) => payloadOf(plan, taskDemand(plan, task).crossings).bytes);
    return { status: "measured", staged: false, stepsPerTask: 1, stepBytes: largest(each), accumulatorBytes: null };
  }
  const stepsPerTask = staging.steps;
  if (stepsPositionFree(plan, staging)) {
    const first = taskSteps(plan, { tensorId, coord: family.grid.map(() => 0) });
    if (!("staging" in first)) throw new Error(`"${tensorId}" lost its steps`);
    return {
      status: "measured",
      staged: true,
      stepsPerTask,
      stepBytes: stepPayload(plan, stepOf(plan, first, 0)).bytes,
      accumulatorBytes: first.accumulator.bytes,
    };
  }
  if (family.count * stepsPerTask > work)
    return {
      status: "unmeasured",
      staged: true,
      stepsPerTask,
      reason: `${family.count} tasks of ${stepsPerTask} step${stepsPerTask === 1 ? "" : "s"} to measure`,
    };
  const payloads: Figure[] = [];
  const accumulators: Figure[] = [];
  for (const task of tasks()) {
    const steps = taskSteps(plan, task);
    if (!("staging" in steps)) throw new Error(`"${tensorId}" lost its steps`);
    accumulators.push(steps.accumulator.bytes);
    for (const step of stepsOf(plan, steps)) payloads.push(stepPayload(plan, step).bytes);
  }
  return { status: "measured", staged: true, stepsPerTask, stepBytes: largest(payloads), accumulatorBytes: largest(accumulators) };
}
