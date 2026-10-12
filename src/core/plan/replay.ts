/**
 * A family's tasks, run by run: where their demand lands, and the part of it
 * the inspected task demands too.
 *
 * The family is walked in listing order - row-major over its grid, which is a
 * listing and not an execution order - and cut into at most `REPLAY_FRAMES`
 * contiguous runs, so a long family replays in the time a short one does
 * without skipping a task. A family no longer than the cap has one task per
 * run. Each run's demand is the bounded cone of its tiles, which is the union
 * of those tasks' demand: exact for the plan, or widened with the cone's
 * reasons.
 *
 * The overlap with the inspected task is the element-level picture of what
 * the family's demand rows count: the elements two tasks both read are the
 * duplicate demand between them.
 */

import { linearRangeToBoxes } from "../ops/reshape";
import { intersect, isEmpty, Region } from "../region";
import { familyOf, regionDemand, TaskRef, taskDemand, TilePlan } from "./plan";
import { tileCoord, tilesBox } from "./tile-family";

/** What a run reaches on one tensor, and the part the inspected task reaches too. */
export type ReplayReach = { region: Region; shared: Region | null };

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

/** Each tensor a cone reaches, other than the one it starts on. */
function reachOf(tensors: ReadonlyMap<string, { region: Region }>, own: string): Map<string, Region> {
  const reach = new Map<string, Region>();
  for (const [tensorId, { region }] of tensors) if (tensorId !== own) reach.set(tensorId, region);
  return reach;
}

export function familyReplay(
  plan: TilePlan,
  task: TaskRef,
  { sampleCap = REPLAY_FRAMES }: { sampleCap?: number } = {}
): FamilyReplay {
  const family = familyOf(plan, task);
  const inspected = reachOf(taskDemand(plan, task).tensors, task.tensorId);
  const frames = replayRuns(family.count, sampleCap).map(([first, end]): ReplayFrame => {
    const region: Region = {
      boxes: linearRangeToBoxes(first, end, [...family.grid]).map((coords) => tilesBox(family, coords)),
      exact: true,
      reasons: [],
    };
    const reach = new Map<string, ReplayReach>();
    for (const [tensorId, reached] of reachOf(regionDemand(plan, task.tensorId, region).tensors, task.tensorId)) {
      const mine = inspected.get(tensorId);
      const shared = mine ? intersect(reached, mine) : null;
      reach.set(tensorId, { region: reached, shared: shared && !isEmpty(shared) ? shared : null });
    }
    return {
      first: { tensorId: task.tensorId, coord: tileCoord(family, first) },
      last: { tensorId: task.tensorId, coord: tileCoord(family, end - 1) },
      tasks: end - first,
      region,
      reach,
    };
  });
  return { task: { tensorId: task.tensorId, coord: [...task.coord] }, frames, tasks: family.count };
}
