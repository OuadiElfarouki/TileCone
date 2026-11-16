import { tilePlan } from "../../core/plan/plan";
import { tileBox } from "../../core/plan/tile-family";
import { hasUnitStride, stepCoordinate } from "../../core/coordinates";
import { defaultPlanTile } from "../../view/tensor/seeds";
import { idRecord, operationForTensor } from "../../view/workspace";
import { recordWorkspace, sameNumbers, sameTask, sameTiles } from "../history";
import { activePlanTrail, derivePlan, inspectTask, MAX_KEPT_PLANS, tileContaining } from "../plan";
import { revealHidden } from "../selection-edit";
import { GetState, SetState, State } from "../types";
import { isBudget, sameBudget } from "../../core/plan/presets";
import { sweepArtifact } from "../../worker/analysis-jobs";
import { analysisWorkerAvailable, cancelSweep, isAnalysisCancelled, sweepInWorker } from "../analysis-worker-client";

/** Counts sweeps asked for, so a result arriving after a later request, a
 * cleared sweep or a new budget is recognised and dropped. */
let sweepEpoch = 0;

/** The declared tiling and the inspected task. */
export const planActions = (set: SetState, get: GetState): Pick<
  State,
  "setPlanTile"
  | "selectPlanTask"
  | "followPlanTask"
  | "backPlanTask"
  | "planTaskAt"
  | "setPlanTileAt"
  | "setPlanSteps"
  | "setPlanBudget"
  | "runSweep"
  | "clearSweep"
  | "installSweepRow"
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

  setPlanSteps: (tensorId, steps) => {
    const state = get();
    const { resolved, planSteps, planTask } = state;
    if (!resolved || !state.planTiles[tensorId]) return;
    const declared = idRecord(planSteps);
    // A declaration that declares nothing is none: it would otherwise sit in
    // the state and take an undo step while changing no plan.
    if (steps && !Object.keys(steps.chunk ?? {}).length && !steps.order?.length && !steps.accumulator)
      steps = null;
    if (steps) declared[tensorId] = steps;
    else if (tensorId in declared) delete declared[tensorId];
    else return;
    const next = derivePlan(resolved, state.planTiles, planTask, state, declared);
    // `derivePlan` drops what the graph does not support, and an unsupported
    // declaration is then indistinguishable from none: nothing to record.
    if (next.planSteps === planSteps || (steps && !next.planSteps[tensorId])) return;
    set({ workspaceHistory: recordWorkspace(state), ...next });
  },

  setPlanBudget: (budget) => {
    const current = get().planBudget;
    if (!isBudget(budget) || sameBudget(budget, current)) return;
    // A sweep's rows are the plans within the budget it ran under.
    sweepEpoch++;
    cancelSweep();
    set({
      planBudget: {
        stepBytes: budget.stepBytes,
        ...(budget.accumulatorBytes === undefined ? {} : { accumulatorBytes: budget.accumulatorBytes }),
      },
      sweep: { status: "idle" },
    });
  },

  runSweep: async (scope) => {
    const { resolved, planTiles, planBudget, workerGraphId } = get();
    if (!resolved) return;
    const tiled = Object.keys(planTiles).sort();
    const options = {
      budget: planBudget,
      ...(scope === "current" && tiled.length ? { tiled: [tiled] } : {}),
    };
    const epoch = ++sweepEpoch;
    set({ sweep: { status: "running", scope } });
    // The graph and the budget the result answers for: a replacement of either
    // while the sweep ran makes it an answer to a question no longer asked.
    const current = () => epoch === sweepEpoch && get().resolved === resolved && get().planBudget === planBudget;
    try {
      const result = analysisWorkerAvailable()
        ? await sweepInWorker({ graphId: workerGraphId, graph: resolved, options })
        : sweepArtifact(resolved, options);
      if (current()) set({ sweep: { status: "done", scope, budget: planBudget, result } });
    } catch (error) {
      if (!current()) return;
      if (isAnalysisCancelled(error)) {
        set({ sweep: { status: "idle" } });
        return;
      }
      set({ sweep: { status: "failed", scope, message: error instanceof Error ? error.message : String(error) } });
    }
  },

  clearSweep: () => {
    sweepEpoch++;
    cancelSweep();
    if (get().sweep.status !== "idle") set({ sweep: { status: "idle" } });
  },

  installSweepRow: (index) => {
    const state = get();
    const { resolved, sweep } = state;
    const row = sweep.status === "done" ? sweep.result.rows[index] : undefined;
    if (!resolved || !row) return;
    const next = derivePlan(resolved, idRecord(row.tiles), state.planTask, state, idRecord(row.steps));
    if (next.planTiles === state.planTiles && next.planSteps === state.planSteps) return;
    set({ workspaceHistory: recordWorkspace(state), ...next, inspectorTab: "plan" });
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
    // `to` is the installed object, which `activePlanTrail` compares by identity.
    const landed = get().planTask;
    if (landed && sameTask(landed, task))
      set({ planTrail: { from: [...(trail?.from ?? []), from], to: landed } });
  },

  backPlanTask: (index) => {
    const trail = activePlanTrail(get());
    const target = trail?.from[index];
    if (!trail || !target) return;
    get().selectPlanTask(target);
    const landed = get().planTask;
    if (landed && sameTask(landed, target))
      set({ planTrail: index > 0 ? { from: trail.from.slice(0, index), to: landed } : null });
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
