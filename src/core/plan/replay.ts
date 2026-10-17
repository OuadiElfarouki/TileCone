/**
 * A family's tasks, run by run: where their demand lands, how far upstream it
 * reaches through the plan's own tasks, and the part of it the inspected task
 * demands too.
 *
 * The family is walked in listing order - row-major over its grid, which is a
 * listing and not an execution order - and cut into at most `REPLAY_FRAMES`
 * contiguous runs, so a long family replays in the time a short one does
 * without skipping a task. A family no longer than the cap has one task per
 * run.
 *
 * At depth 1 a run's demand is the bounded cone of its tiles at the plan's
 * frontier: what its tasks read from tiled tensors and graph inputs, and the
 * untiled tensors they compute. A cone relates each element on its own, so it
 * is the union of those tasks' demand. Each further level follows the plan:
 * the tiled tensors read at one level are produced by the tasks whose tiles
 * meet that demand, and those tasks compute whole tiles, so the next level is
 * the bounded cone of those tiles. Deeper levels are therefore tile-aligned,
 * which is what the plan computes, rather than the element-level cone of what
 * was read. Every region is exact for the plan, or widened with its reasons.
 *
 * The overlap with the inspected task, expanded to the same depth, is the
 * element-level picture of what the family's demand rows count: the elements
 * two tasks both read are the duplicate demand between them.
 */

import { linearRangeToBoxes } from "../ops/reshape";
import { coaccessWithin } from "../entangle";
import { producerNode } from "../graph";
import type { BoundedCone } from "../propagate";
import { getOp } from "../ops/index";
import { canonicalize, count, fromBox, intersect, isEmpty, markInexact, Region } from "../region";
import { readOrder } from "./interfaces";
import { familyOf, regionDemand, TaskRef, TilePlan } from "./plan";
import { coordinatePage, PlanError, tileBox, tileCoord, tileRange, tilesBox, type TileFamily } from "./tile-family";

/** What a run reaches on one tensor, and the part the inspected task reaches too. */
export type ReplayReach = {
  region: Region;
  shared: Region | null;
  /** The nearest level the run reaches this tensor at: 1 for its own tasks' demand. */
  level: number;
  /**
   * On a tiled tensor, the tiles the run needs: the producer tasks of what it
   * reads, whole tiles. Absent on graph inputs and untiled tensors.
   */
  producers?: Region;
};

export type ReplayFrame = {
  /** The run's first and last task in listing order; the same task for a run of one. */
  first: TaskRef;
  last: TaskRef;
  /** Family tasks in the run. */
  tasks: number;
  /**
   * The run's first and last step as task paths, family task first: one task
   * at depth 1, then each upstream task it was reached through (`familyReplay`).
   */
  firstPath: TaskRef[];
  lastPath: TaskRef[];
  /** Steps in the run. */
  steps: number;
  /** The run's tiles on the family's tensor. */
  region: Region;
  /**
   * Deeper than 1, the tiles the run's steps walk below the family task, per
   * tensor: the producer tasks on each path, which are what a step moves.
   * Empty at depth 1, where the family task is the step.
   */
  walked: ReadonlyMap<string, Region>;
  /** What the run reads or computes, per tensor other than its own. */
  reach: ReadonlyMap<string, ReplayReach>;
};

export type FamilyReplay = {
  /** The task the replay is about, whose demand the overlaps are measured against. */
  task: TaskRef;
  /** Levels of the plan's tasks the frames reach through; 1 is the run's own tasks. */
  depth: number;
  frames: ReplayFrame[];
  /** Tasks in the family; more than `frames.length` when runs hold several. */
  tasks: number;
  /** Steps in the replay: the family's tasks at depth 1, task paths deeper. */
  steps: number;
  /** The tensors whose tasks are the step at the deepest level reached. */
  stepTensors: string[];
};

/** Bounded queries a stepped replay may run before declining the depth. */
export const STEP_QUERY_BUDGET = 4096;

/**
 * Frames a stepped replay shows one step each before it cuts steps into runs.
 * Far above `REPLAY_FRAMES` on purpose: a step is the granularity a deeper
 * replay exists to show, and a run of steps paints their union - a whole row
 * of producer tiles at once - which is the picture depth 1 already gives.
 */
export const STEP_FRAMES = 2048;

/** Runs a replay walks at most, by default. */
export const REPLAY_FRAMES = 48;

/**
 * Listing positions cut into at most `cap` contiguous runs of near-equal
 * length, as `[first, end)` pairs that together cover every position once.
 */
export function replayRuns(total: number, cap: number): [number, number][] {
  const n = Math.max(1, Math.min(total, Math.floor(cap)));
  return Array.from({ length: n }, (_, i): [number, number] => [
    Math.floor((i * total) / n),
    Math.floor(((i + 1) * total) / n),
  ]);
}

/** The tiles of a family that a region meets, as one region of whole tiles. */
function tilesMet(family: TileFamily, region: Region): Region {
  return {
    boxes: region.boxes.map((b) => tilesBox(family, tileRange(family, b))),
    exact: region.exact,
    reasons: [...region.reasons],
  };
}

/** Two regions of one tensor as one, each box kept and canonicalized together. */
const merged = (a: Region, b: Region): Region => canonicalize({
  boxes: [...a.boxes, ...b.boxes],
  exact: a.exact && b.exact,
  reasons: [...new Set([...a.reasons, ...b.reasons])],
});

const depthLimits = new WeakMap<TilePlan, ReadonlyMap<string, number>>();

/**
 * How many levels of the plan's tasks a replay of this family can reach: 1
 * when its tasks read no tiled tensor, and one more than the deepest tiled
 * tensor they read otherwise.
 *
 * One topological pass counts tiled tensors along each upstream path. Untiled
 * tensors carry that count without adding a level, so shared fused prefixes
 * are visited once. Cache every limit by the immutable plan: subsequent frame,
 * fade and family queries are constant-time, and another tiling gets its own
 * index. Weak keys let discarded plans and their indexes be collected.
 */
export function replayDepthLimit(plan: TilePlan, tensorId: string): number {
  let limits = depthLimits.get(plan);
  if (!limits) {
    const carried = new Map<string, number>();
    const computed = new Map<string, number>();
    for (const node of plan.graph.topo) {
      let upstream = 0;
      for (const input of node.inputs) upstream = Math.max(upstream, carried.get(input) ?? 0);
      const limit = upstream + 1;
      for (const output of node.outputs) {
        computed.set(output, limit);
        carried.set(output, plan.families.has(output) ? limit : upstream);
      }
    }
    limits = computed;
    depthLimits.set(plan, limits);
  }
  return limits.get(tensorId) ?? 1;
}

/** `definite` is the union of the exact parts: what is reached for certain,
 *  where `region` may be widened by some path. */
type Reached = { region: Region; level: number; definite: Region | null };

/**
 * Everything the tasks computing `region` of `tensorId` reach, `depth` levels
 * of the plan deep, with the nearest level each tensor is reached at. A tiled
 * tensor reached before the last level is expanded through the tasks producing
 * the tiles it is read from.
 */
function reachOf(plan: TilePlan, tensorId: string, region: Region, depth: number): Map<string, Reached> {
  const reached = new Map<string, Reached>();
  const add = (id: string, part: Region, level: number) => {
    const prior = reached.get(id);
    const exact = part.exact ? part : null;
    reached.set(id, prior
      ? {
          region: merged(prior.region, part),
          level: Math.min(prior.level, level),
          definite: prior.definite && exact ? merged(prior.definite, exact) : prior.definite ?? exact,
        }
      : { region: part, level, definite: exact });
  };
  let wave = new Map<string, Region>([[tensorId, region]]);
  for (let level = 1; level <= depth && wave.size; level++) {
    const next = new Map<string, Region>();
    for (const [id, computed] of wave)
      for (const [read, { region: part }] of regionDemand(plan, id, computed).tensors) {
        if (read === id) continue;
        add(read, part, level);
        const family = plan.families.get(read);
        if (!family || level === depth) continue;
        // The producers of what was read compute whole tiles, and their own
        // demand is the next level.
        const tiles = tilesMet(family, part);
        const prior = next.get(read);
        next.set(read, prior ? merged(prior, tiles) : tiles);
      }
    wave = next;
  }
  return reached;
}

/**
 * The family's replay at `depth`.
 *
 * At depth 1 a step is one of the family's tasks, and its frame is everything
 * it reads and computes. Deeper, the family's tasks are no longer the unit: a
 * task needs the tasks producing its tiles first, and those are what move. A
 * step is then a path - a family task, one producer task it requires, one
 * producer of that, down to `depth - 1` levels of producers - and its frame is
 * each task on the path, the elements of the other operands each one is
 * combined with in the task it feeds (`coaccessWithin`), and what the last
 * task reads. For `Z = Y @ A` at depth 2 the step is one Y task, and it moves
 * with the one block of A it meets within the Z task, rather than with all of
 * the Z task's row of Y and column of A at once. A path stops early at a task
 * with no producers. Steps run in listing order, family task first, then its
 * producers in operand order.
 */
export function familyReplay(
  plan: TilePlan,
  task: TaskRef,
  { sampleCap = REPLAY_FRAMES, stepCap = STEP_FRAMES, depth = 1, budget = STEP_QUERY_BUDGET }:
    { sampleCap?: number; stepCap?: number; depth?: number; budget?: number } = {}
): FamilyReplay {
  const family = familyOf(plan, task);
  const levels = Math.max(1, Math.min(Math.floor(depth), replayDepthLimit(plan, task.tensorId)));
  if (levels > 1) return steppedReplay(plan, task, family, levels, stepCap, budget);
  const inspected = reachOf(plan, task.tensorId, fromBox(tileBox(family, task.coord)), levels);
  const frames = replayRuns(family.count, sampleCap).map(([first, end]): ReplayFrame => {
    const region: Region = {
      boxes: linearRangeToBoxes(first, end, [...family.grid]).map((coords) => tilesBox(family, coords)),
      exact: true,
      reasons: [],
    };
    const reach = new Map<string, ReplayReach>();
    for (const [tensorId, { region: reached, level }] of reachOf(plan, task.tensorId, region, levels)) {
      const mine = inspected.get(tensorId)?.region;
      const shared = mine ? intersect(reached, mine) : null;
      const producerFamily = plan.families.get(tensorId);
      reach.set(tensorId, {
        region: reached,
        shared: shared && !isEmpty(shared) ? shared : null,
        level,
        ...(producerFamily ? { producers: tilesMet(producerFamily, reached) } : {}),
      });
    }
    const firstTask = { tensorId: task.tensorId, coord: tileCoord(family, first) };
    const lastTask = { tensorId: task.tensorId, coord: tileCoord(family, end - 1) };
    return {
      first: firstTask,
      last: lastTask,
      tasks: end - first,
      firstPath: [firstTask],
      lastPath: [lastTask],
      steps: end - first,
      region,
      walked: new Map(),
      reach,
    };
  });
  return {
    task: { tensorId: task.tensorId, coord: [...task.coord] },
    depth: levels,
    frames,
    tasks: family.count,
    steps: family.count,
    stepTensors: [task.tensorId],
  };
}

/** What one step reaches on one tensor, before runs are merged. */
type StepReach = { region: Region; level: number; producers?: Region };

function steppedReplay(
  plan: TilePlan,
  task: TaskRef,
  family: TileFamily,
  levels: number,
  stepCap: number,
  budget: number
): FamilyReplay {
  const graph = plan.graph;
  const cones = new Map<string, BoundedCone>();
  const tileOf = (t: TaskRef): Region => fromBox(tileBox(plan.families.get(t.tensorId)!, t.coord));
  const coneOf = (t: TaskRef): BoundedCone => {
    const key = `${t.tensorId}|${t.coord.join(",")}`;
    let cone = cones.get(key);
    if (!cone) {
      if (cones.size >= budget)
        throw new PlanError(
          "PLAN_BUDGET",
          `stepping ${graph.tensors[task.tensorId].name} at depth ${levels} needs more than ${budget} task queries; choose a smaller depth`
        );
      cone = regionDemand(plan, t.tensorId, tileOf(t));
      cones.set(key, cone);
    }
    return cone;
  };
  /** A task's producer tasks: per tiled tensor it reads, in operand order, each in listing order. */
  const producersOf = (t: TaskRef): TaskRef[] => {
    const cone = coneOf(t);
    const tiled = cone.crossings.filter((c) => plan.families.has(c.tensorId));
    const own = producerNode(graph, t.tensorId)?.id ?? "";
    return readOrder(tiled, own, graph.topo).flatMap((tensorId) => {
      const producer = plan.families.get(tensorId)!;
      const coordinates: Region = {
        boxes: tiled.filter((c) => c.tensorId === tensorId).flatMap((c) => c.region.boxes.map((b) => tileRange(producer, b))),
        exact: true,
        reasons: [],
      };
      return [...coordinatePage(coordinates, 0, count(coordinates))].map((coord) => ({ tensorId, coord }));
    });
  };

  // Every step, depth first: each family task, then its producers, to depth - 1.
  const paths: TaskRef[][] = [];
  const walk = (path: TaskRef[]) => {
    const producers = path.length < levels ? producersOf(path[path.length - 1]) : [];
    if (!producers.length) paths.push(path);
    else for (const p of producers) walk([...path, p]);
  };
  for (let i = 0; i < family.count; i++) walk([{ tensorId: task.tensorId, coord: tileCoord(family, i) }]);

  /** One step's frame content, per tensor other than the family's own. */
  const stepReach = (path: TaskRef[]): Map<string, StepReach> => {
    const reach = new Map<string, StepReach>();
    const add = (id: string, region: Region, level: number, producers?: Region) => {
      const prior = reach.get(id);
      reach.set(id, prior
        ? {
            region: merged(prior.region, region),
            level: Math.min(prior.level, level),
            producers: prior.producers && producers ? merged(prior.producers, producers) : prior.producers ?? producers,
          }
        : { region, level, ...(producers ? { producers } : {}) });
    };
    for (let i = 1; i < path.length; i++) {
      const consumer = path[i - 1];
      const step = path[i];
      const tile = tileOf(step);
      add(step.tensorId, tile, i, tile);
      // Follow the step's tile through the task it feeds: from the operations
      // reading it across the frontier, through every fused operation its
      // values pass on the way to that task's tile, pairing it at each one
      // with the other operands it meets there and drawing the fused part it
      // becomes. A tile read through `D = relu(C); Y = D @ W` meets W at the
      // matmul, two operations after it is read.
      const cone = coneOf(consumer);
      const carried = new Map<string, Region>();
      for (const node of graph.topo) {
        const blocks = node.inputs.map((input, slot): Region | null => {
          if (input === step.tensorId) {
            const crossing = cone.crossings.find((c) => c.node === node.id && c.slot === slot);
            const block = crossing ? intersect(crossing.region, tile) : null;
            return block && !isEmpty(block) ? block : null;
          }
          return carried.get(input) ?? null;
        });
        if (blocks.every((b) => !b)) continue;
        const spec = getOp(node.op)!;
        const ctx = { inShapes: graph.shapesOf(node.inputs), outShapes: graph.shapesOf(node.outputs), attrs: node.attrs };
        for (const [outSlot, out] of node.outputs.entries()) {
          const outputs = cone.tensors.get(out)?.region;
          // Each output can independently be fused, tiled, or outside this task.
          if (!outputs || (out !== consumer.tensorId && plan.families.has(out))) continue;
          let reachedOut: Region | null = null;
          blocks.forEach((block, slot) => {
            if (!block) return;
            node.inputs.forEach((other, otherSlot) => {
              if (otherSlot === slot) return;
              const met = coaccessWithin(graph, node.id, slot, block, otherSlot, outputs, outSlot);
              if (met) add(other, met, i);
            });
            for (const b of block.boxes) {
              const part = intersect(spec.forward(slot, b, ctx)[outSlot], outputs);
              const marked = block.exact ? part : markInexact(part, ...block.reasons);
              reachedOut = reachedOut ? merged(reachedOut, marked) : marked;
            }
          });
          if (!reachedOut || isEmpty(reachedOut) || out === consumer.tensorId) continue;
          carried.set(out, reachedOut);
          add(out, reachedOut, i);
        }
      }
    }
    // What the last task on the path reads and computes, one level further.
    const last = path[path.length - 1];
    for (const [id, { region }] of coneOf(last).tensors) {
      if (id === last.tensorId) continue;
      const producer = plan.families.get(id);
      add(id, region, path.length, producer ? tilesMet(producer, region) : undefined);
    }
    reach.delete(task.tensorId);
    return reach;
  };

  const inspected = reachOf(plan, task.tensorId, tileOf(task), levels);
  const frames = replayRuns(paths.length, stepCap).map(([first, end]): ReplayFrame => {
    const run = paths.slice(first, end);
    const merges = new Map<string, StepReach>();
    for (const path of run)
      for (const [id, part] of stepReach(path)) {
        const prior = merges.get(id);
        merges.set(id, prior
          ? {
              region: merged(prior.region, part.region),
              level: Math.min(prior.level, part.level),
              producers: prior.producers && part.producers ? merged(prior.producers, part.producers) : prior.producers ?? part.producers,
            }
          : part);
      }
    const reach = new Map<string, ReplayReach>();
    for (const [id, part] of merges) {
      const mine = inspected.get(id)?.region;
      const shared = mine ? intersect(part.region, mine) : null;
      reach.set(id, { ...part, shared: shared && !isEmpty(shared) ? shared : null });
    }
    const owners = [...new Map(run.map((path) => [path[0].coord.join(","), path[0]])).values()];
    const walked = new Map<string, Region>();
    for (const path of run)
      for (const step of path.slice(1)) {
        const prior = walked.get(step.tensorId);
        walked.set(step.tensorId, prior ? merged(prior, tileOf(step)) : tileOf(step));
      }
    return {
      first: run[0][0],
      last: run[run.length - 1][0],
      tasks: owners.length,
      firstPath: run[0],
      lastPath: run[run.length - 1],
      steps: run.length,
      region: canonicalize({ boxes: owners.flatMap((t) => tileOf(t).boxes), exact: true, reasons: [] }),
      walked,
      reach,
    };
  });
  const deepest = Math.max(...paths.map((path) => path.length));
  return {
    task: { tensorId: task.tensorId, coord: [...task.coord] },
    depth: levels,
    frames,
    tasks: family.count,
    steps: paths.length,
    stepTensors: [...new Set(paths.filter((path) => path.length === deepest).map((path) => path[path.length - 1].tensorId))],
  };
}

/** One tensor upstream of a task, as the plan supplies it. */
export type LineageRow = {
  tensorId: string;
  /**
   * The nearest level the task reaches this tensor at: 1 for what the task
   * itself reads or computes, 2 for what the tasks producing those tiles do,
   * and so on.
   */
  level: number;
  /** A tiled tensor's producer tasks, a graph input read from memory, or an
   *  untiled tensor the tasks one level down compute themselves. */
  kind: "tasks" | "input" | "computed";
  /** What is read or computed of it, exact for the plan or widened with reasons. */
  region: Region;
  /** On a tiled tensor, the producer tasks as an exact cover of tile coordinates. */
  coordinates?: Region;
  /** The producer tasks reached through exact demand; the rest are only possible. */
  definiteCoordinates?: Region;
  /** On a tiled tensor, how many of its family's tasks those are, of how many,
   *  and how many of them are definite. */
  tasks?: number;
  definiteTasks?: number;
  of?: number;
};

/**
 * Everything one task depends on through the plan, to the graph's inputs: per
 * tensor upstream, the nearest level it is reached at and the tasks, input
 * elements or fused computation that supply it. The same walk a replay takes,
 * for one task and at the deepest level the plan has, so an output task can be
 * related to the tasks of any tensor above it without a sweep of its family.
 * Rows run by level, then in the order the source declares the tensors.
 */
export function taskLineage(plan: TilePlan, task: TaskRef): LineageRow[] {
  const family = familyOf(plan, task);
  const depth = replayDepthLimit(plan, task.tensorId);
  const order = new Map(Object.keys(plan.graph.tensors).map((id, i) => [id, i]));
  const rows = [...reachOf(plan, task.tensorId, fromBox(tileBox(family, task.coord)), depth)]
    .map(([tensorId, { region, level, definite }]): LineageRow => {
      const producer = plan.families.get(tensorId);
      if (producer) {
        // Exact covers of tile coordinates, even where the demand is a bound.
        const cover = (r: Region | null): Region => ({
          boxes: r ? r.boxes.map((b) => tileRange(producer, b)) : [],
          exact: true,
          reasons: [],
        });
        const coordinates = cover(region);
        const definiteCoordinates = cover(definite);
        return {
          tensorId, level, kind: "tasks", region, coordinates, definiteCoordinates,
          tasks: count(coordinates), definiteTasks: count(definiteCoordinates), of: producer.count,
        };
      }
      return { tensorId, level, kind: plan.graph.tensors[tensorId].producer ? "computed" : "input", region };
    });
  return rows.sort((a, b) => a.level - b.level || order.get(a.tensorId)! - order.get(b.tensorId)!);
}
