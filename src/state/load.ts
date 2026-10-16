/** Installing a compiled graph, an example, or a shared workspace. */

import { validateSelection } from "../core/executor";
import { Graph, ResolvedGraph } from "../core/graph";
import { familyOf, tilePlan } from "../core/plan/plan";
import { fromBox } from "../core/region";
import { EXAMPLES } from "../examples/index";
import { BaseGraphLayout, readNodeKey } from "../view/graph/graph-scene";
import { NodeOffset, NodeOffsets } from "../view/graph/node-layout";
import { planesOf } from "../view/tensor/seeds";
import { defaultViewCfg, remapped, ViewCfg, viewCfgFits } from "../view/tensor/tensor-view";
import { graphScale, TILE_SCALE_MAX, TILE_SCALE_MIN } from "../view/tensor/tiling";
import { idRecord, MAX_NODE_OFFSET, operationForTensor } from "../view/workspace";
import { planRestoreArtifact } from "../worker/analysis-jobs";
import type { PlanRestoreArtifact } from "../worker/analysis-protocol";
import { recompute } from "./analysis";
import { restoredKeptPlans, NO_PLAN } from "./plan";
import { State, WorkspaceRestore } from "./types";

export function loadResolvedGraph(
  graph: Graph,
  resolved: ResolvedGraph,
  worker?: { graphId: number | null; layout: BaseGraphLayout; graphPx: number }
): Pick<
  State,
  | "graph" | "resolved" | "baseLayout" | "workerGraphId" | "loadError" | "diagnostics" | "selection" | "backwardRes" | "forwardRes"
  | "byTensorRes"
  | "entangled"
  | "perBox" | "focusedBox" | "pinnedBox" | "editBox" | "viewCfgs" | "preview" | "graphPx"
  | "hiddenBoxes" | "analysisGroup" | "workspaceHistory" | "nodeOffsets"
  | "planTiles" | "planTask" | "plan" | "planSupply" | "keptPlans"
  | "replayPlayback" | "reuseLens" | "replayDepth" | "placementCarry" | "resizingTensor"
> {
  const viewCfgs = Object.create(null) as Record<string, ViewCfg>;
  for (const t of Object.values(resolved.tensors)) viewCfgs[t.id] = defaultViewCfg(t.resolved!);
  return {
    graph,
    resolved,
    baseLayout: worker?.layout ?? null,
    workerGraphId: worker?.graphId ?? null,
    loadError: null,
    diagnostics: [],
    entangled: null,
    selection: null,
    // Undo entries refer to node IDs and coordinates in one resolved graph.
    // They must never survive a graph replacement or composite rewrite.
    workspaceHistory: [],
    nodeOffsets: idRecord<NodeOffset>(),
    backwardRes: null,
    byTensorRes: null,
    forwardRes: null,
    perBox: null,
    focusedBox: null,
    pinnedBox: null,
    editBox: null,
    hiddenBoxes: new Set<number>(),
    analysisGroup: null,
    replayPlayback: null,
    reuseLens: null,
    replayDepth: idRecord<number>(),
    preview: null,
    viewCfgs,
    graphPx: worker?.graphPx ?? graphScale(planesOf(resolved)),
    // A plan names tensors and tile coordinates in one graph, as the selection
    // does, and a kept plan's totals were measured on this graph alone.
    ...NO_PLAN,
    keptPlans: [],
    placementCarry: false,
    resizingTensor: null,
  };
}

/**
 * Whether two graphs have the same tensors and the same operations wired the
 * same way, so that they differ at most in sizes - the case a parameter edit
 * produces. Node and tensor ids are what positions and views are keyed by.
 */
export function sameStructure(a: Graph, b: Graph): boolean {
  const tensorsA = Object.keys(a.tensors);
  if (tensorsA.length !== Object.keys(b.tensors).length) return false;
  if (!tensorsA.every((id) => {
    const tensor = a.tensors[id];
    const other = b.tensors[id];
    return !!other && tensor.shape.length === other.shape.length &&
      tensor.dtype === other.dtype && tensor.role === other.role &&
      sameData(tensor.axisNames, other.axisNames);
  })) return false;
  if (a.nodes.length !== b.nodes.length) return false;
  const same = (x: string[], y: string[]) => x.length === y.length && x.every((v, i) => v === y[i]);
  const byId = new Map(b.nodes.map((node) => [node.id, node]));
  return a.nodes.every((node) => {
    const other = byId.get(node.id);
    return !!other && other.op === node.op && sameData(node.attrs, other.attrs) &&
      same(node.inputs, other.inputs) && same(node.outputs, other.outputs);
  });
}

/** Attribute order in the source does not change an operation's meaning. */
function sameData(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a))
    return Array.isArray(b) && a.length === b.length && a.every((value, i) => sameData(value, b[i]));
  if (!a || !b || typeof a !== "object" || typeof b !== "object" || Array.isArray(b)) return false;
  const left = a as Record<string, unknown>;
  const right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length &&
    keys.every((key) => Object.prototype.hasOwnProperty.call(right, key) && sameData(left[key], right[key]));
}

/**
 * A card's view carried onto its tensor's new shape: slider positions and a
 * tile clamped into the new extents, the chosen axes kept. A view that still
 * does not fit falls back to the default rather than failing the install.
 */
export function carriedViewCfg(shape: number[], previous: ViewCfg | undefined): ViewCfg {
  if (!previous || previous.sliders.length !== shape.length) return defaultViewCfg(shape);
  const cfg: ViewCfg = {
    projection: previous.projection,
    sliders: previous.sliders.map((v, axis) => Math.min(v, Math.max(1, shape[axis]) - 1)),
    ...(previous.tile ? { tile: previous.tile.map((v, axis) => Math.min(v, Math.max(1, shape[axis]))) } : {}),
    ...(previous.axes ? { axes: [previous.axes[0], previous.axes[1]] as [number, number] } : {}),
    ...(previous.unfold ? { unfold: previous.unfold } : {}),
    ...(previous.scale !== undefined ? { scale: previous.scale } : {}),
  };
  // A shape that grew past what unfolds keeps the rest of its view, sliced.
  const { unfold: _unfold, ...sliced } = cfg;
  return viewCfgFits(shape, cfg) ? cfg : viewCfgFits(shape, sliced) ? sliced : defaultViewCfg(shape);
}

/** Install one successful DSL compilation. Shared by the synchronous fallback
 * and the Worker path so they cannot drift in example/default-selection rules. */
export function installedDSLState(
  text: string,
  graph: Graph,
  resolved: ResolvedGraph,
  worker?: { graphId: number; layout: BaseGraphLayout; graphPx: number },
  previous?: Pick<State, "graph" | "viewCfgs" | "dslText">
): Partial<State> {
  const base = loadResolvedGraph(graph, resolved, worker);
  /* A run that changed only sizes keeps the workspace the reader arranged:
     each card's view here, and the placement of every node, which the canvas
     carries over once it has the new layout to measure against. The selection
     and the plan are still reset - their coordinates belong to the old sizes.
     Building the very text already built is a reload, and starts fresh. */
  if (previous?.graph && previous.dslText !== text && sameStructure(previous.graph, graph)) {
    for (const t of Object.values(resolved.tensors))
      base.viewCfgs[t.id] = carriedViewCfg(t.resolved!, previous.viewCfgs[t.id]);
    base.placementCarry = true;
  }
  const exampleIndex = EXAMPLES.findIndex((example) => example.dsl === text);
  const example = exampleIndex >= 0 ? EXAMPLES[exampleIndex] : null;
  const state: Partial<State> = {
    ...base,
    dslText: text,
    draftText: text,
    exampleIndex,
    focusNode: null,
    compiling: false,
  };
  if (example?.defaultSelection) {
    state.selection = {
      parts: [{
        tensorId: example.defaultSelection.tensor,
        box: example.defaultSelection.box.map(([lo, hi]) => ({ lo, hi })),
      }],
    };
    state.focusedBox = null;
    state.pinnedBox = null;
    // The starter tile is the one placed last, so the keyboard moves it.
    state.editBox = 0;
    Object.assign(state, recompute(resolved, state.selection));
  }
  state.loadError = null;
  state.diagnostics = [];
  return state;
}

/**
 * The workspace an experiment's source opens on, when that exact source is
 * built: the graph fresh, the example's selection, and the experiment's plan,
 * task and kept tilings, on the Plan view. Restored like a shared link, so the
 * kept plans are evaluated where a link's are. Canvas settings stay the
 * reader's own. Null for any other source.
 */
export function experimentWorkspace(
  text: string,
  state: Pick<State, "direction" | "showEntangled" | "tileScale" | "snapToGrid" | "axisMode">
): WorkspaceRestore | null {
  const example = EXAMPLES.find((candidate) => candidate.dsl === text);
  const experiment = example?.experiment;
  if (!example || !experiment) return null;
  const selection = example.defaultSelection;
  return {
    dsl: text,
    direction: state.direction,
    showEntangled: state.showEntangled,
    tileScale: state.tileScale,
    snapToGrid: state.snapToGrid,
    axisMode: state.axisMode,
    parts: selection
      ? [{ tensorId: selection.tensor, box: selection.box.map(([lo, hi]) => ({ lo, hi })) }]
      : null,
    planTiles: experiment.tiles,
    planTask: { tensorId: experiment.task.tensor, coord: experiment.task.coord },
    keptTilings: experiment.kept,
    planView: true,
  };
}

export const validWorkspaceFields = (workspace: WorkspaceRestore): boolean =>
  ["none", "backward", "forward", "both"].includes(workspace.direction) &&
  (workspace.showEntangled === undefined || typeof workspace.showEntangled === "boolean") &&
  Number.isFinite(workspace.tileScale) &&
  typeof workspace.snapToGrid === "boolean" &&
  ["symbolic", "numeric"].includes(workspace.axisMode);

/** Validate the graph-relative pieces of a shared workspace and install them
 * over a freshly loaded graph. Compilation itself may happen in either realm.
 * The browser passes measurements computed in its restore Worker transaction. */
export function restoredWorkspaceState(
  workspace: WorkspaceRestore,
  graph: Graph,
  resolved: ResolvedGraph,
  worker?: { graphId: number | null; layout: BaseGraphLayout; graphPx: number },
  restoredPlan?: PlanRestoreArtifact
): Partial<State> {
  const base = loadResolvedGraph(graph, resolved, worker);
  for (const [id, cfg] of Object.entries(workspace.viewCfgs ?? {})) {
    const shape = resolved.tensors[id]?.resolved;
    if (!shape || !viewCfgFits(shape, cfg)) throw new Error(`invalid view for tensor "${id}"`);
    base.viewCfgs[id] = {
      projection: cfg.projection,
      sliders: cfg.sliders.slice(),
      ...(cfg.tile ? { tile: cfg.tile.slice() } : {}),
      ...(cfg.axes && remapped(shape, cfg) ? { axes: [cfg.axes[0], cfg.axes[1]] as [number, number] } : {}),
      ...(cfg.unfold ? { unfold: cfg.unfold } : {}),
      ...(cfg.scale !== undefined ? { scale: cfg.scale } : {}),
    };
  }
  const checkedParts = (workspace.parts ?? []).map((part) => {
    const checked = validateSelection(resolved, {
      tensorId: part.tensorId,
      region: fromBox(part.box),
    });
    if (checked.region.boxes.length !== 1)
      throw new Error(`selection on tensor "${part.tensorId}" is empty`);
    return { tensorId: checked.tensorId, box: checked.region.boxes[0] };
  });
  const selection = checkedParts.length ? { parts: checkedParts } : null;
  const hiddenBoxes = new Set(workspace.hiddenBoxes ?? []);
  for (const index of hiddenBoxes) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= checkedParts.length)
      throw new Error("invalid disabled tile");
  }
  const analysisGroup = workspace.analysisGroup ?? null;
  if (analysisGroup !== null && !checkedParts.some((p) => p.tensorId === analysisGroup))
    throw new Error("invalid analysis group");
  const pinnedBox = workspace.pinnedBox ?? null;
  if (pinnedBox !== null && (!Number.isSafeInteger(pinnedBox) || pinnedBox < 0 ||
      pinnedBox >= checkedParts.length || hiddenBoxes.has(pinnedBox) ||
      checkedParts[pinnedBox].tensorId !== analysisGroup))
    throw new Error("invalid selected tile");
  if (workspace.draftText !== undefined && typeof workspace.draftText !== "string")
    throw new Error("invalid draft");
  const nodeIds = new Set(resolved.nodes.map((node) => node.id));
  const checkedOffsets = Object.create(null) as NodeOffsets;
  for (const [key, offset] of Object.entries(workspace.nodeOffsets ?? {})) {
    const named = readNodeKey(key);
    const placed = named !== null &&
      (named.kind === "tensor" ? !!resolved.tensors[named.id] : nodeIds.has(named.id));
    if (!placed ||
        !Number.isFinite(offset.dx) || !Number.isFinite(offset.dy) ||
        Math.abs(offset.dx) > MAX_NODE_OFFSET || Math.abs(offset.dy) > MAX_NODE_OFFSET)
      throw new Error(`invalid layout offset for node "${key}"`);
    if (Math.abs(offset.dx) >= 1e-6 || Math.abs(offset.dy) >= 1e-6)
      checkedOffsets[key] = { dx: offset.dx, dy: offset.dy };
  }
  const planTiles = idRecord(workspace.planTiles);
  const task = workspace.planTask ?? null;
  // A task that names no tile of the plan is refused rather than dropped: the
  // link's task is what its Plan view describes.
  const checkedPlan = tilePlan(resolved, planTiles);
  if (task) familyOf(checkedPlan, task);
  const restored = restoredPlan ?? planRestoreArtifact(resolved, {
    tiles: planTiles, task, kept: workspace.keptTilings ?? [],
  });
  const plan = {
    planTiles, planTask: task,
    plan: Object.keys(planTiles).length ? checkedPlan : null,
    planSupply: restored.supply,
  };
  return {
    ...base,
    ...plan,
    keptPlans: restoredKeptPlans(workspace.keptTilings ?? [], restored.reports),
    ...(plan.planTask ? { selectedOp: operationForTensor(resolved, plan.planTask.tensorId) } : {}),
    ...(workspace.planView ? { inspectorTab: "plan" as const } : {}),
    dslText: workspace.dsl,
    draftText: workspace.draftText ?? workspace.dsl,
    exampleIndex: EXAMPLES.findIndex((example) => example.dsl === workspace.dsl),
    focusNode: null,
    direction: workspace.direction,
    showEntangled: workspace.showEntangled ?? false,
    tileScale: Math.max(TILE_SCALE_MIN, Math.min(TILE_SCALE_MAX, Math.round(workspace.tileScale))),
    snapToGrid: workspace.snapToGrid,
    axisMode: workspace.axisMode,
    nodeOffsets: checkedOffsets,
    selection,
    hiddenBoxes,
    analysisGroup,
    pinnedBox,
    focusedBox: pinnedBox,
    compiling: false,
    ...recompute(resolved, selection),
  };
}
