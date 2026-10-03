/** Workspace undo: snapshots of selection, layout and plan. */

import { Graph, ResolvedGraph } from "../core/graph";
import { TaskRef } from "../core/plan/plan";
import { BaseGraphLayout } from "../view/graph/graph-scene";
import { NodeOffsets } from "../view/graph/node-layout";
import { type Selection } from "../view/workspace";
import { State } from "./types";

/** What a plan edit changes: the tile extents per planned tensor, and the task inspected. */
export type PlanEdit = { tiles: Record<string, number[]>; task: TaskRef | null };

export type WorkspaceSnapshot = {
  selection: Selection;
  nodeOffsets: NodeOffsets;
  /** Required so that no edit can record a snapshot that forgets the plan. */
  plan: PlanEdit;
  /**
   * The source and graph as they were, for the one edit that replaces them.
   *
   * Expanding a composite rewrites `dslText` and `draftText` with generated DSL
   * — the author's comments, formatting and names for intermediates all go —
   * and it is reached by clicking a glyph on the canvas, sometimes with the
   * source panel collapsed to a rail where none of that is even visible.
   * Without this, undo restored a selection into a graph that no longer had the
   * composite in it, and the shortcut sheet's promise was false for the largest
   * action in the app.
   *
   * Absent on the ordinary entries, which change neither.
   */
  source?: {
    dslText: string;
    draftText: string;
    graph: Graph;
    resolved: ResolvedGraph;
    baseLayout: BaseGraphLayout | null;
    graphPx: number;
    exampleIndex: number;
  };
};
export const WORKSPACE_HISTORY_LIMIT = 40;

export function planEditOf(state: Pick<State, "planTiles" | "planTask">): PlanEdit {
  return { tiles: state.planTiles, task: state.planTask };
}

/** Whether two tilings divide the same tensors at the same extents. */
export function sameTiles(
  a: Record<string, number[]>,
  b: Record<string, number[]>
): boolean {
  const left = Object.keys(a).sort();
  const right = Object.keys(b).sort();
  return (
    left.length === right.length &&
    left.every((id, i) => id === right[i] && sameNumbers(a[id], b[id]))
  );
}

/** Whether two plan edits divide the same tensors the same way and inspect the same task. */
export function samePlanEdit(a: PlanEdit, b: PlanEdit): boolean {
  return sameTiles(a.tiles, b.tiles) && sameTask(a.task, b.task);
}

export const sameNumbers = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((value, index) => value === b[index]);

export const sameTask = (a: TaskRef | null, b: TaskRef | null): boolean =>
  a === b || (!!a && !!b && a.tensorId === b.tensorId && sameNumbers(a.coord, b.coord));

export function appendWorkspaceHistory(
  history: WorkspaceSnapshot[],
  snapshot: WorkspaceSnapshot
): WorkspaceSnapshot[] {
  return [...history, snapshot].slice(-WORKSPACE_HISTORY_LIMIT);
}

type WorkspaceState = Pick<State, "workspaceHistory" | "selection" | "nodeOffsets" | "planTiles" | "planTask">;

/** The workspace as it stands: what an undo restores. */
export function workspaceSnapshot(state: Omit<WorkspaceState, "workspaceHistory">): WorkspaceSnapshot {
  return { selection: state.selection, nodeOffsets: state.nodeOffsets, plan: planEditOf(state) };
}

/**
 * The history with the workspace as it stands recorded as one undo step.
 * Every edit records through here, so no edit can record a snapshot that
 * leaves out part of the workspace. `changes` supplies what the state no
 * longer holds: a node move records the offsets from before the drag.
 */
export function recordWorkspace(
  state: WorkspaceState,
  changes: Partial<WorkspaceSnapshot> = {}
): WorkspaceSnapshot[] {
  return appendWorkspaceHistory(state.workspaceHistory, { ...workspaceSnapshot(state), ...changes });
}
