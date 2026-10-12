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
import { canonicalize, fromBox, intersect, isEmpty, Region } from "../region";
import { familyOf, regionDemand, TaskRef, TilePlan } from "./plan";
import { tileBox, tileCoord, tileRange, tilesBox, type TileFamily } from "./tile-family";

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
  /** Tasks in the run. */
  tasks: number;
  /** The run's tiles on the family's tensor. */
  region: Region;
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
};

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

type Reached = { region: Region; level: number };

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
    reached.set(id, prior
      ? { region: merged(prior.region, part), level: Math.min(prior.level, level) }
      : { region: part, level });
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

export function familyReplay(
  plan: TilePlan,
  task: TaskRef,
  { sampleCap = REPLAY_FRAMES, depth = 1 }: { sampleCap?: number; depth?: number } = {}
): FamilyReplay {
  const family = familyOf(plan, task);
  const levels = Math.max(1, Math.min(Math.floor(depth), replayDepthLimit(plan, task.tensorId)));
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
    return {
      first: { tensorId: task.tensorId, coord: tileCoord(family, first) },
      last: { tensorId: task.tensorId, coord: tileCoord(family, end - 1) },
      tasks: end - first,
      region,
      reach,
    };
  });
  return {
    task: { tensorId: task.tensorId, coord: [...task.coord] },
    depth: levels,
    frames,
    tasks: family.count,
  };
}
