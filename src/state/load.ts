/** Installing a compiled graph, an example, or a shared workspace. */

import { validateSelection } from "../core/executor";
import { Graph, ResolvedGraph } from "../core/graph";
import { fromBox } from "../core/region";
import { EXAMPLES } from "../examples/index";
import { BaseGraphLayout, readNodeKey } from "../view/graph/graph-scene";
import { NodeOffset, NodeOffsets } from "../view/graph/node-layout";
import { planesOf } from "../view/tensor/seeds";
import { defaultViewCfg, remapped, ViewCfg, viewCfgFits } from "../view/tensor/tensor-view";
import { graphScale, TILE_SCALE_MAX, TILE_SCALE_MIN } from "../view/tensor/tiling";
import { idRecord, MAX_NODE_OFFSET } from "../view/workspace";
import { recompute } from "./analysis";
import { NO_PLAN } from "./plan";
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
  | "perBox" | "focusedBox" | "pinnedBox" | "viewCfgs" | "preview" | "graphPx"
  | "hiddenBoxes" | "analysisGroup" | "workspaceHistory" | "nodeOffsets"
  | "planTiles" | "planTask" | "plan" | "planSupply"
  | "executionPlayback" | "executionScope"
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
    hiddenBoxes: new Set<number>(),
    analysisGroup: null,
    executionPlayback: null,
    executionScope: null,
    preview: null,
    viewCfgs,
    graphPx: worker?.graphPx ?? graphScale(planesOf(resolved)),
    // A plan names tensors and tile coordinates in one graph, as the selection does.
    ...NO_PLAN,
  };
}

/** Install one successful DSL compilation. Shared by the synchronous fallback
 * and the Worker path so they cannot drift in example/default-selection rules. */
export function installedDSLState(
  text: string,
  graph: Graph,
  resolved: ResolvedGraph,
  worker?: { graphId: number; layout: BaseGraphLayout; graphPx: number }
): Partial<State> {
  const base = loadResolvedGraph(graph, resolved, worker);
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
    Object.assign(state, recompute(resolved, state.selection));
  }
  state.loadError = null;
  state.diagnostics = [];
  return state;
}

export const validWorkspaceFields = (workspace: WorkspaceRestore): boolean =>
  ["none", "backward", "forward", "both"].includes(workspace.direction) &&
  (workspace.showEntangled === undefined || typeof workspace.showEntangled === "boolean") &&
  Number.isFinite(workspace.tileScale) &&
  typeof workspace.snapToGrid === "boolean" &&
  ["symbolic", "numeric"].includes(workspace.axisMode);

/** Validate the graph-relative pieces of a shared workspace and install them
 * over a freshly loaded graph. Compilation itself may happen in either realm. */
export function restoredWorkspaceState(
  workspace: WorkspaceRestore,
  graph: Graph,
  resolved: ResolvedGraph,
  worker?: { graphId: number | null; layout: BaseGraphLayout; graphPx: number }
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
  return {
    ...base,
    dslText: workspace.dsl,
    draftText: workspace.dsl,
    exampleIndex: -1,
    focusNode: null,
    direction: workspace.direction,
    showEntangled: workspace.showEntangled ?? false,
    tileScale: Math.max(TILE_SCALE_MIN, Math.min(TILE_SCALE_MAX, Math.round(workspace.tileScale))),
    snapToGrid: workspace.snapToGrid,
    axisMode: workspace.axisMode,
    nodeOffsets: checkedOffsets,
    selection,
    compiling: false,
    ...recompute(resolved, selection),
  };
}
