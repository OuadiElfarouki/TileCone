import { create } from "zustand";
import { Graph, ResolvedGraph, resolveGraph } from "../core/graph";
import { expandNode, isExpandable } from "../core/expand";
import { PropResult, propagateBackward, propagateForward } from "../core/propagate";
import { Region, empty, removeBoxAt, subtract, translateRegion, union } from "../core/region";
import { EXAMPLES } from "../examples/index";
import { parseDSL, toDSL } from "../parse/dsl";
import { parseGraphJSON } from "../parse/json";
import { TILE_SCALE_MAX, TILE_SCALE_MIN } from "./tiling";

export type Direction = "backward" | "forward" | "both";
/** Everything snaps to the tile grid, which is also the render grid. */
export type SelectMode = "cell" | "box" | "row" | "col" | "all";
/** How a new drag combines with the existing selection. Modifier keys override it. */
export type ComposeMode = "replace" | "union" | "subtract";

export type ViewCfg = {
  sliders: number[]; // index per hidden axis (full rank length; row/col entries ignored)
  projection: boolean; // union over hidden axes vs slice at slider
};

export function defaultViewCfg(shape: number[]): ViewCfg {
  return { sliders: shape.map(() => 0), projection: true };
}

/**
 * Which axes the grid draws, fixed row-major for every tensor: the last axis
 * (fastest-varying) is columns, the one before it is rows. There is no per-card
 * axis remapping — a different view of a tensor is a `transpose` node in the
 * graph, where it is part of the computation being explained rather than a
 * display setting that silently disagrees with the DSL.
 */
export function viewAxes(shape: number[]): { rowAxis: number; colAxis: number } {
  const rank = shape.length;
  return { rowAxis: rank >= 2 ? rank - 2 : -1, colAxis: rank >= 1 ? rank - 1 : -1 };
}

type State = {
  dslText: string;
  exampleIndex: number;
  graph: Graph | null;
  resolved: ResolvedGraph | null;
  loadError: string | null;

  selection: { tensorId: string; region: Region } | null;
  selectionHistory: { tensorId: string; region: Region }[];
  direction: Direction;
  selectMode: SelectMode;
  composeMode: ComposeMode;
  backwardRes: PropResult | null;
  forwardRes: PropResult | null;
  /** One propagation per selection box, so a highlighted region can be traced
   * back to the box that produced it. Null when there are too many boxes. */
  perBox: BoxProp[] | null;
  focusedBox: number | null;
  preview: PropResult | null; // hover preview (backward only)

  viewCfgs: Record<string, ViewCfg>;
  /** Global detail setting: shifts every tensor's tile by 2^tileScale. */
  tileScale: number;
  hideInert: boolean;
  countIntermediates: boolean;
  focusTensor: string | null;
  editorOpen: boolean;

  loadExample: (i: number) => void;
  applyDSL: (text: string) => void;
  applyJSON: (text: string) => void;
  setSelection: (tensorId: string, region: Region, compose?: ComposeMode) => void;
  clearSelection: () => void;
  undoSelection: () => void;
  /** Move the whole selection along one axis, clamped to the tensor. */
  moveSelection: (axis: number, delta: number) => void;
  deleteBox: (index: number) => void;
  /** Isolate one selection box's dependency cone (hover/click in the inspector). */
  setFocusedBox: (index: number | null) => void;
  setDirection: (d: Direction) => void;
  setSelectMode: (m: SelectMode) => void;
  setComposeMode: (m: ComposeMode) => void;
  setViewCfg: (tensorId: string, cfg: Partial<ViewCfg>) => void;
  setTileScale: (v: number) => void;
  setHideInert: (v: boolean) => void;
  setCountIntermediates: (v: boolean) => void;
  setFocusTensor: (id: string | null) => void;
  setEditorOpen: (v: boolean) => void;
  setPreviewCell: (tensorId: string | null, cell?: number[]) => void;
  expandNodeInPlace: (nodeId: string) => void;
};

/** Per-selection-box propagation results, aligned with `selection.region.boxes`. */
export type BoxProp = { backward: PropResult | null; forward: PropResult | null };

/** Above this many boxes, per-box attribution costs more than it is worth. */
export const MAX_PER_BOX_PROPS = 12;

function recompute(
  resolved: ResolvedGraph | null,
  selection: State["selection"],
  direction: Direction
): Pick<State, "backwardRes" | "forwardRes" | "perBox" | "focusedBox"> {
  if (!resolved || !selection || selection.region.boxes.length === 0)
    return { backwardRes: null, forwardRes: null, perBox: null, focusedBox: null };
  const backwardRes = direction !== "forward" ? propagateBackward(resolved, selection) : null;
  const forwardRes = direction !== "backward" ? propagateForward(resolved, selection) : null;

  const boxes = selection.region.boxes;
  let perBox: BoxProp[] | null = null;
  if (boxes.length === 1) {
    // the aggregate already is this box; no need to propagate twice
    perBox = [{ backward: backwardRes, forward: forwardRes }];
  } else if (boxes.length <= MAX_PER_BOX_PROPS) {
    perBox = boxes.map((b) => {
      const one = {
        tensorId: selection.tensorId,
        region: { boxes: [b], exact: selection.region.exact, reasons: selection.region.reasons },
      };
      return {
        backward: direction !== "forward" ? propagateBackward(resolved, one) : null,
        forward: direction !== "backward" ? propagateForward(resolved, one) : null,
      };
    });
  }
  return { backwardRes, forwardRes, perBox, focusedBox: null };
}

/** Apply a transform to the current selection's region and repropagate. */
function editSelection(
  get: () => State,
  set: (partial: Partial<State>) => void,
  fn: (region: Region, shape: number[]) => Region
): void {
  const { selection, resolved, direction, selectionHistory } = get();
  if (!selection || !resolved) return;
  const shape = resolved.tensors[selection.tensorId].resolved!;
  const region = fn(selection.region, shape);
  const sel = region.boxes.length === 0 ? null : { tensorId: selection.tensorId, region };
  set({
    selection: sel,
    selectionHistory: [...selectionHistory, selection].slice(-40),
    preview: null,
    ...recompute(resolved, sel, direction),
  });
}

function loadGraph(graph: Graph): Pick<
  State,
  | "graph" | "resolved" | "loadError" | "selection" | "backwardRes" | "forwardRes"
  | "perBox" | "focusedBox" | "viewCfgs" | "preview"
> {
  const resolved = resolveGraph(JSON.parse(JSON.stringify(graph)) as Graph);
  const viewCfgs: Record<string, ViewCfg> = {};
  for (const t of Object.values(resolved.tensors)) viewCfgs[t.id] = defaultViewCfg(t.resolved!);
  return {
    graph,
    resolved,
    loadError: null,
    selection: null,
    backwardRes: null,
    forwardRes: null,
    perBox: null,
    focusedBox: null,
    preview: null,
    viewCfgs,
  };
}

export const useStore = create<State>((set, get) => ({
  dslText: EXAMPLES[0].dsl,
  exampleIndex: 0,
  graph: null,
  resolved: null,
  loadError: null,
  selection: null,
  selectionHistory: [],
  direction: "backward",
  selectMode: "box",
  composeMode: "replace",
  backwardRes: null,
  forwardRes: null,
  perBox: null,
  focusedBox: null,
  preview: null,
  viewCfgs: {},
  tileScale: 0,
  hideInert: false,
  countIntermediates: false,
  focusTensor: null,
  editorOpen: false,

  loadExample: (i) => {
    const ex = EXAMPLES[i];
    try {
      const graph = parseDSL(ex.dsl);
      const base = loadGraph(graph);
      const st: Partial<State> = { ...base, dslText: ex.dsl, exampleIndex: i, focusTensor: null };
      if (ex.defaultSelection && base.resolved) {
        const region: Region = {
          boxes: [ex.defaultSelection.box.map(([lo, hi]) => ({ lo, hi }))],
          exact: true,
          reasons: [],
        };
        st.selection = { tensorId: ex.defaultSelection.tensor, region };
        Object.assign(st, recompute(base.resolved, st.selection!, get().direction));
      }
      set(st as State);
    } catch (e) {
      set({ loadError: (e as Error).message });
    }
  },

  applyDSL: (text) => {
    try {
      const graph = parseDSL(text);
      set({ ...loadGraph(graph), dslText: text });
    } catch (e) {
      set({ loadError: (e as Error).message });
    }
  },

  applyJSON: (text) => {
    try {
      const graph = parseGraphJSON(text);
      set({ ...loadGraph(graph), dslText: toDSL(graph) });
    } catch (e) {
      set({ loadError: (e as Error).message });
    }
  },

  setSelection: (tensorId, region, compose) => {
    const { selection, resolved, direction, composeMode, selectionHistory } = get();
    const mode = compose ?? composeMode;
    let next: Region = region;
    if (selection && selection.tensorId === tensorId && mode !== "replace") {
      next = mode === "union" ? union(selection.region, region) : subtract(selection.region, region);
    }
    const sel = next.boxes.length === 0 ? null : { tensorId, region: next };
    set({
      selection: sel,
      selectionHistory: selection ? [...selectionHistory, selection].slice(-40) : selectionHistory,
      preview: null,
      ...recompute(resolved, sel, direction),
    });
  },

  clearSelection: () => {
    const { selection, selectionHistory } = get();
    set({
      selection: null,
      selectionHistory: selection ? [...selectionHistory, selection].slice(-40) : selectionHistory,
      backwardRes: null,
      forwardRes: null,
      perBox: null,
      focusedBox: null,
      preview: null,
    });
  },

  undoSelection: () => {
    const { selectionHistory, resolved, direction } = get();
    if (!selectionHistory.length) return;
    const prev = selectionHistory[selectionHistory.length - 1];
    set({
      selection: prev,
      selectionHistory: selectionHistory.slice(0, -1),
      preview: null,
      ...recompute(resolved, prev, direction),
    });
  },

  moveSelection: (axis, delta) =>
    editSelection(get, set, (r, shape) => translateRegion(r, axis, delta, shape)),

  deleteBox: (index) => editSelection(get, set, (r) => removeBoxAt(r, index)),

  setFocusedBox: (index) => set({ focusedBox: index }),

  setDirection: (d) => {
    const { resolved, selection } = get();
    set({ direction: d, ...recompute(resolved, selection, d) });
  },

  setSelectMode: (m) => set({ selectMode: m }),
  setComposeMode: (m) => set({ composeMode: m }),

  setViewCfg: (tensorId, cfg) =>
    set((s) => ({ viewCfgs: { ...s.viewCfgs, [tensorId]: { ...s.viewCfgs[tensorId], ...cfg } } })),

  setTileScale: (v) =>
    set({ tileScale: Math.max(TILE_SCALE_MIN, Math.min(TILE_SCALE_MAX, Math.round(v))) }),

  setHideInert: (v) => set({ hideInert: v }),
  setCountIntermediates: (v) => set({ countIntermediates: v }),
  setFocusTensor: (id) => set({ focusTensor: id }),
  setEditorOpen: (v) => set({ editorOpen: v }),

  setPreviewCell: (tensorId, cell) => {
    const { resolved } = get();
    if (!tensorId || !cell || !resolved || resolved.nodes.length > 400) {
      if (get().preview) set({ preview: null });
      return;
    }
    try {
      const region: Region = {
        boxes: [cell.map((v) => ({ lo: v, hi: v + 1 }))],
        exact: true,
        reasons: [],
      };
      set({ preview: propagateBackward(resolved, { tensorId, region }) });
    } catch {
      set({ preview: null });
    }
  },

  expandNodeInPlace: (nodeId) => {
    const { graph } = get();
    if (!graph) return;
    try {
      const g2 = expandNode(graph, nodeId);
      set({ ...loadGraph(g2), dslText: get().dslText + `\n# (node ${nodeId} expanded in view)` });
    } catch (e) {
      set({ loadError: (e as Error).message });
    }
  },
}));

export { isExpandable, EXAMPLES, empty };
