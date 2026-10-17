import { tilePlan } from "../../core/plan/plan";
import { tileBox } from "../../core/plan/tile-family";
import { hasUnitStride, stepCoordinate } from "../../core/coordinates";
import { defaultPlanTile } from "../../view/tensor/seeds";
import { idRecord, operationForTensor } from "../../view/workspace";
import { recordWorkspace, sameNumbers, sameTask, sameTiles } from "../history";
import { activePlanTrail, derivePlan, inspectTask, MAX_KEPT_PLANS, tileContaining } from "../plan";
import { revealHidden } from "../selection-edit";
import { GetState, SetState, State } from "../types";

/** The declared tiling and the inspected task. */
export const planActions = (set: SetState, get: GetState): Pick<
  State,
  "setPlanTile"
  | "selectPlanTask"
  | "followPlanTask"
  | "backPlanTask"
  | "planTaskAt"
  | "setPlanTileAt"
  | "tilePlanTensor"
  | "movePlanTask"
  | "keepPlan"
  | "dropKeptPlan"
  | "restoreKeptPlan"
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
      workspaceHistory: recordWorkspace(state),
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
      workspaceHistory: recordWorkspace(state),
      selectedOp: next.planTask ? operationForTensor(state.resolved, next.planTask.tensorId) : state.selectedOp,
      ...next,
    });
  },

  followPlanTask: (task) => {
    const state = get();
    const from = state.planTask;
    if (!from || sameTask(from, task)) return;
    const trail = activePlanTrail(state);
    get().selectPlanTask(task);
    // Only once the step has landed: a task the plan does not have is refused.
    if (sameTask(get().planTask, task))
      set({ planTrail: { from: [...(trail?.from ?? []), from], to: task } });
  },

  backPlanTask: (index) => {
    const trail = activePlanTrail(get());
    const target = trail?.from[index];
    if (!trail || !target) return;
    get().selectPlanTask(target);
    if (sameTask(get().planTask, target))
      set({ planTrail: index > 0 ? { from: trail.from.slice(0, index), to: target } : null });
  },

  planTaskAt: (tensorId, element) => {
    const state = get();
    const resolved = state.resolved;
    if (!resolved?.tensors[tensorId]?.producer) return;
    const tile =
      state.planTiles[tensorId] ??
      defaultPlanTile(resolved, tensorId, state.tileScale, state.viewCfgs);
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
      defaultPlanTile(resolved, tensorId, state.tileScale, state.viewCfgs)
    );
  },

  movePlanTask: (axis, delta, record = true) => {
    const state = get();
    const { plan, planTask, resolved } = state;
    if (!plan || !planTask || !resolved) return;
    const family = plan.families.get(planTask.tensorId)!;
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= family.grid.length || !Number.isSafeInteger(delta)) return;
    const coord = stepCoordinate(planTask.coord, family.grid, axis, delta, hasUnitStride(family.shape, axis));
    if (sameNumbers(coord, planTask.coord)) return;
    set({
      workspaceHistory: record
        ? recordWorkspace(state)
        : state.workspaceHistory,
      ...derivePlan(resolved, state.planTiles, { tensorId: planTask.tensorId, coord }, state),
    });
    revealHidden(get, planTask.tensorId, tileBox(family, coord));
  },

  /* `report` is the evaluation of the current tiling, which the caller holds:
     the plan report is computed in the query Worker, not in the store. */
  keepPlan: ({ total, unwritten }) => {
    const { plan, planTiles, keptPlans } = get();
    if (!plan || !total || keptPlans.length >= MAX_KEPT_PLANS) return;
    if (keptPlans.some((kept) => sameTiles(kept.tiles, planTiles))) return;
    const id = (keptPlans[keptPlans.length - 1]?.id ?? 0) + 1;
    set({ keptPlans: [...keptPlans, { id, tiles: planTiles, total, unwritten }] });
  },

  dropKeptPlan: (id) => {
    const { keptPlans } = get();
    if (keptPlans.some((kept) => kept.id === id))
      set({ keptPlans: keptPlans.filter((kept) => kept.id !== id) });
  },

  restoreKeptPlan: (id) => {
    const state = get();
    const { resolved, planTask, planTiles } = state;
    const kept = state.keptPlans.find((k) => k.id === id);
    if (!resolved || !kept || sameTiles(kept.tiles, planTiles)) return;
    // The inspected task moves to the kept tiling's tile holding its first
    // element, and is dropped when the kept plan does not tile its tensor.
    const tile = planTask ? kept.tiles[planTask.tensorId] : undefined;
    const task =
      planTask && tile
        ? {
            tensorId: planTask.tensorId,
            coord: tileContaining(
              tile,
              planTask.coord.map((c, axis) => c * planTiles[planTask.tensorId][axis])
            ),
          }
        : null;
    set({
      workspaceHistory: recordWorkspace(state),
      ...derivePlan(resolved, kept.tiles, task, state),
    });
  },
});
