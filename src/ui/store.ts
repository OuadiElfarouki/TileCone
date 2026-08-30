import { create } from "zustand";
import { Graph, ResolvedGraph, resolveGraph } from "../core/graph";
import { expandNode, isExpandable } from "../core/expand";
import { PropResult, propagateBackward, propagateForward } from "../core/propagate";
import {
  Box,
  Region,
  addPart,
  empty,
  removePart,
  subtractFromParts,
  translateAllParts,
  translatePart,
} from "../core/region";
import { EXAMPLES } from "../examples/index";
import { parseDSL } from "../parse/dsl";
import { TILE_SCALE_MAX, TILE_SCALE_MIN } from "./tiling";

export type Direction = "backward" | "forward" | "both";
/** Everything snaps to the tile grid, which is also the render grid. */
export type SelectMode = "cell" | "box" | "row" | "col" | "all";
/**
 * How a new drag combines with the existing selection. Modifier keys override it.
 * There is deliberately no "replace" mode: it is `clear` followed by a draw, so
 * offering it as a third button would be a second way to do one thing.
 * "replace" survives only as an internal operation, for restoring a shared link.
 */
export type ComposeMode = "union" | "subtract";
type Compose = ComposeMode | "replace";

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

  /** `region.boxes` are the user's ordered PARTS (identity-stable, may overlap),
   * never a canonicalized set. See the note in core/region.ts. */
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
  /** The part currently highlighted: the pinned one, else the hovered one. */
  focusedBox: number | null;
  /** Sticky focus set by clicking a part; survives the pointer leaving the row. */
  pinnedBox: number | null;
  /** True while a drag is being rubber-banded on a card, so Escape can cancel it. */
  dragging: boolean;
  preview: PropResult | null; // hover preview (backward only)

  viewCfgs: Record<string, ViewCfg>;
  /** Global detail setting: shifts every tensor's tile by 2^tileScale. */
  tileScale: number;
  hideInert: boolean;
  countIntermediates: boolean;
  focusTensor: string | null;

  loadExample: (i: number) => void;
  applyDSL: (text: string) => void;
  setSelection: (tensorId: string, region: Region, compose?: Compose) => void;
  clearSelection: () => void;
  undoSelection: () => void;
  /** Move the whole selection along one axis, clamped to the tensor. */
  moveSelection: (axis: number, delta: number) => void;
  deleteBox: (index: number) => void;
  /** Transient hover focus; ignored while a part is pinned. */
  hoverBox: (index: number | null) => void;
  /** Click a part to pin it, or the same part again to unpin. */
  togglePinBox: (index: number) => void;
  /** Drop both hover and pin. What Escape does. */
  clearFocus: () => void;
  setDragging: (v: boolean) => void;
  setDirection: (d: Direction) => void;
  setSelectMode: (m: SelectMode) => void;
  setComposeMode: (m: ComposeMode) => void;
  setViewCfg: (tensorId: string, cfg: Partial<ViewCfg>) => void;
  setTileScale: (v: number) => void;
  setHideInert: (v: boolean) => void;
  setCountIntermediates: (v: boolean) => void;
  setFocusTensor: (id: string | null) => void;
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
): Pick<State, "backwardRes" | "forwardRes" | "perBox"> {
  if (!resolved || !selection || selection.region.boxes.length === 0)
    return { backwardRes: null, forwardRes: null, perBox: null };
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
  return { backwardRes, forwardRes, perBox };
}

/**
 * Apply a transform to the selection's parts and repropagate.
 * `keepFocus` holds the focused part across edits that preserve indices (a move);
 * edits that reorder or remove parts drop it so a stale index can never be used.
 */
function editSelection(
  get: () => State,
  set: (partial: Partial<State>) => void,
  fn: (parts: Box[], shape: number[]) => Box[],
  keepFocus = false
): void {
  const { selection, resolved, direction, selectionHistory, focusedBox } = get();
  if (!selection || !resolved) return;
  const shape = resolved.tensors[selection.tensorId].resolved!;
  const parts = fn(selection.region.boxes, shape);
  const region: Region = { boxes: parts, exact: true, reasons: [] };
  const sel = parts.length === 0 ? null : { tensorId: selection.tensorId, region };
  const nextFocus =
    keepFocus && sel && focusedBox !== null && focusedBox < parts.length ? focusedBox : null;
  set({
    selection: sel,
    selectionHistory: [...selectionHistory, selection].slice(-40),
    focusedBox: nextFocus,
    pinnedBox: nextFocus === null ? null : get().pinnedBox,
    preview: null,
    ...recompute(resolved, sel, direction),
  });
}

function loadGraph(graph: Graph): Pick<
  State,
  | "graph" | "resolved" | "loadError" | "selection" | "backwardRes" | "forwardRes"
  | "perBox" | "focusedBox" | "pinnedBox" | "viewCfgs" | "preview"
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
    pinnedBox: null,
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
  composeMode: "union",
  backwardRes: null,
  forwardRes: null,
  perBox: null,
  focusedBox: null,
  pinnedBox: null,
  dragging: false,
  preview: null,
  viewCfgs: {},
  tileScale: 0,
  hideInert: false,
  countIntermediates: false,
  focusTensor: null,

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
        st.focusedBox = null;
        st.pinnedBox = null;
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

  setSelection: (tensorId, region, compose) => {
    const { selection, resolved, direction, composeMode, selectionHistory } = get();
    const mode = compose ?? composeMode;
    const drawn = region.boxes;
    let parts: Box[] = drawn;
    if (selection && selection.tensorId === tensorId && mode !== "replace") {
      parts = selection.region.boxes;
      for (const b of drawn)
        parts = mode === "union" ? addPart(parts, b) : subtractFromParts(parts, b);
    }
    const sel =
      parts.length === 0
        ? null
        : { tensorId, region: { boxes: parts, exact: true, reasons: [] } };
    set({
      selection: sel,
      selectionHistory: selection ? [...selectionHistory, selection].slice(-40) : selectionHistory,
      focusedBox: null,
      pinnedBox: null,
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
      pinnedBox: null,
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
      focusedBox: null,
      pinnedBox: null,
      preview: null,
      ...recompute(resolved, prev, direction),
    });
  },

  /** Moves only the focused part when one is focused, otherwise the whole selection. */
  moveSelection: (axis, delta) => {
    const focused = get().focusedBox;
    editSelection(
      get,
      set,
      (parts, shape) =>
        focused !== null && focused < parts.length
          ? translatePart(parts, focused, axis, delta, shape)
          : translateAllParts(parts, axis, delta, shape),
      true
    );
  },

  deleteBox: (index) => editSelection(get, set, (parts) => removePart(parts, index)),

  hoverBox: (index) => {
    if (get().pinnedBox !== null) return; // a pinned part outranks hovering
    set({ focusedBox: index });
  },

  togglePinBox: (index) => {
    const pinned = get().pinnedBox === index ? null : index;
    set({ pinnedBox: pinned, focusedBox: pinned });
  },

  clearFocus: () => set({ pinnedBox: null, focusedBox: null }),

  setDragging: (v) => set({ dragging: v }),

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
