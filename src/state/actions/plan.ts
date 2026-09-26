import { tilePlan } from "../../core/plan/plan";
import { defaultPlanTile } from "../../view/tensor/seeds";
import { idRecord, operationForTensor } from "../../view/workspace";
import { appendWorkspaceHistory, planEditOf, sameNumbers, sameTask } from "../history";
import { derivePlan, inspectTask, tileContaining } from "../plan";
import { GetState, SetState, State } from "../types";

/** The declared tiling and the inspected task. */
export const planActions = (set: SetState, get: GetState): Pick<
  State,
  "setPlanTile"
  | "selectPlanTask"
  | "planTaskAt"
  | "setPlanTileAt"
  | "tilePlanTensor"
  | "movePlanTask"
> => ({
  setPlanTile: (tensorId, tile) => {
    const state = get();
    const { resolved, planTiles, planTask } = state;
    if (!resolved) return;
    const tiles = idRecord(planTiles);
    if (tile) {
      try {
        tilePlan(resolved, { [tensorId]: tile });
      } catch {
        return; // the panel validates before calling; an invalid tile changes nothing
      }
      if (planTiles[tensorId] && sameNumbers(planTiles[tensorId], tile)) return;
      tiles[tensorId] = [...tile];
    } else if (tensorId in tiles) delete tiles[tensorId];
    else return;
    // A task on the retiled tensor moves to the new tile holding its first element.
    const previous = planTask?.tensorId === tensorId ? planTiles[tensorId] : undefined;
    const task =
      planTask && previous
        ? tile
          ? {
              tensorId,
              coord: tileContaining(tile, planTask.coord.map((c, axis) => c * previous[axis])),
            }
          : null
        : planTask;
    set({
      workspaceHistory: appendWorkspaceHistory(state.workspaceHistory, {
        selection: state.selection,
        tensorOffsets: state.tensorOffsets,
        plan: planEditOf(state),
      }),
      ...derivePlan(resolved, tiles, task, state),
    });
  },

  selectPlanTask: (task) => {
    const state = get();
    if (!state.resolved) return;
    const next = derivePlan(state.resolved, state.planTiles, task, state);
    if (task && !next.planTask) return; // not a task of this plan
    if (sameTask(state.planTask, next.planTask)) {
      const selectedOp = next.planTask
        ? operationForTensor(state.resolved, next.planTask.tensorId)
        : state.selectedOp;
      if (selectedOp !== state.selectedOp) set({ selectedOp });
      return;
    }
    set({
      workspaceHistory: appendWorkspaceHistory(state.workspaceHistory, {
        selection: state.selection,
        tensorOffsets: state.tensorOffsets,
        plan: planEditOf(state),
      }),
      selectedOp: next.planTask ? operationForTensor(state.resolved, next.planTask.tensorId) : state.selectedOp,
      ...next,
    });
  },

  planTaskAt: (tensorId, element) => {
    const state = get();
    const resolved = state.resolved;
    if (!resolved?.tensors[tensorId]?.producer) return;
    const tile =
      state.planTiles[tensorId] ??
      defaultPlanTile(resolved, tensorId, state.tileScale, state.graphPx, state.viewCfgs);
    inspectTask(state, set, tensorId, tile, element);
  },

  setPlanTileAt: (tensorId, tile, element) => {
    const state = get();
    const resolved = state.resolved;
    if (!resolved?.tensors[tensorId]?.producer) return;
    inspectTask(state, set, tensorId, tile, element);
  },

  tilePlanTensor: (tensorId) => {
    const state = get();
    const resolved = state.resolved;
    if (!resolved?.tensors[tensorId]?.producer || state.planTiles[tensorId]) return;
    get().setPlanTile(
      tensorId,
      defaultPlanTile(resolved, tensorId, state.tileScale, state.graphPx, state.viewCfgs)
    );
  },

  movePlanTask: (axis, delta, record = true) => {
    const state = get();
    const { plan, planTask, resolved } = state;
    if (!plan || !planTask || !resolved) return;
    const family = plan.families.get(planTask.tensorId)!;
    if (axis < 0 || axis >= family.grid.length) return;
    const coord = [...planTask.coord];
    coord[axis] = Math.max(0, Math.min(family.grid[axis] - 1, coord[axis] + delta));
    if (coord[axis] === planTask.coord[axis]) return;
    set({
      workspaceHistory: record
        ? appendWorkspaceHistory(state.workspaceHistory, {
            selection: state.selection,
            tensorOffsets: state.tensorOffsets,
            plan: planEditOf(state),
          })
        : state.workspaceHistory,
      ...derivePlan(resolved, state.planTiles, { tensorId: planTask.tensorId, coord }, state),
    });
  },
});
