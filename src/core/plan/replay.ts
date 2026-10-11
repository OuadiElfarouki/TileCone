/**
 * A family's tasks, one at a time: where each task's demand lands, and the
 * part of it the inspected task demands too.
 *
 * The frames are tasks of the declared tiling in listing order - row-major
 * over the family's grid, which is a listing and not an execution order. A
 * family longer than `sampleCap` is shown through one task from each equal
 * stretch of the listing, so a long family replays in the time a short one
 * does. Every frame is a real task of the plan, and its regions are the
 * task's bounded cone: exact for the plan, or widened with the cone's reasons.
 *
 * The overlap with the inspected task is the element-level picture of what
 * the family's demand rows count: the elements two tasks both read are the
 * duplicate demand between them.
 */

import { Box, intersect, isEmpty, Region } from "../region";
import { familyOf, TaskRef, taskDemand, TilePlan } from "./plan";
import { tileBox, tileCoord } from "./tile-family";

/** What one task reaches on one tensor, and the part the inspected task reaches too. */
export type ReplayReach = { region: Region; shared: Region | null };

export type ReplayFrame = {
  task: TaskRef;
  /** The task's tile. */
  box: Box;
  /** What the task reads or computes, per tensor other than its own. */
  reach: ReadonlyMap<string, ReplayReach>;
};

export type FamilyReplay = {
  /** The task the replay is about, whose demand the overlaps are measured against. */
  task: TaskRef;
  frames: ReplayFrame[];
  /** Tasks in the family; more than `frames.length` when the replay is sampled. */
  tasks: number;
};

/** Frames a replay shows at most, by default. */
export const REPLAY_FRAMES = 48;

/**
 * Listing positions to replay: every one up to the cap, otherwise one from each
 * equal stretch of the listing. Taken from the stretch's middle rather than at
 * random, so the same plan always replays the same tasks.
 */
export function replayOrdinals(total: number, cap: number): number[] {
  const n = Math.max(1, Math.min(total, Math.floor(cap)));
  if (n === total) return Array.from({ length: total }, (_, i) => i);
  return Array.from({ length: n }, (_, i) => {
    const lo = Math.floor((i * total) / n);
    const hi = Math.floor(((i + 1) * total) / n);
    return lo + Math.floor((hi - lo) / 2);
  });
}

/** Each tensor a task's cone reaches, other than the task's own. */
function reachOf(plan: TilePlan, task: TaskRef): Map<string, Region> {
  const reach = new Map<string, Region>();
  for (const [tensorId, { region }] of taskDemand(plan, task).tensors)
    if (tensorId !== task.tensorId) reach.set(tensorId, region);
  return reach;
}

export function familyReplay(
  plan: TilePlan,
  task: TaskRef,
  { sampleCap = REPLAY_FRAMES }: { sampleCap?: number } = {}
): FamilyReplay {
  const family = familyOf(plan, task);
  const inspected = reachOf(plan, task);
  const frames = replayOrdinals(family.count, sampleCap).map((ordinal): ReplayFrame => {
    const other: TaskRef = { tensorId: task.tensorId, coord: tileCoord(family, ordinal) };
    const reach = new Map<string, ReplayReach>();
    for (const [tensorId, region] of reachOf(plan, other)) {
      const mine = inspected.get(tensorId);
      const shared = mine ? intersect(region, mine) : null;
      reach.set(tensorId, { region, shared: shared && !isEmpty(shared) ? shared : null });
    }
    return { task: other, box: tileBox(family, other.coord), reach };
  });
  return { task: { tensorId: task.tensorId, coord: [...task.coord] }, frames, tasks: family.count };
}
