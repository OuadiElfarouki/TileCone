import { create } from "zustand";
import { executeQuery } from "../core/executor";
import { Graph, ResolvedGraph, resolveGraph } from "../core/graph";
import { expandNode, isExpandable } from "../core/expand";
import { PropResult, mergeProps } from "../core/propagate";
import {
  Box,
  Region,
  addPart,
  empty,
  fromBox,
  subtractFromParts,
  translateAllParts,
  translatePart,
} from "../core/region";
import { EXAMPLES } from "../examples/index";
import { compileDSL } from "../parse/compiler";
import { graphScale, MAX_ELEM_PX, planeExtents, TILE_SCALE_MAX, TILE_SCALE_MIN } from "./tiling";

/** Which independently toggled cones are active in the workspace. `both` and
 * `none` are UI combinations; the checked executor keeps its smaller query API. */
export type Direction = "none" | "backward" | "forward" | "both";
export type ConeDirection = "backward" | "forward";
export type PanelSide = "left" | "right";
export type Theme = "light" | "dark";
export type TensorOffset = { dx: number; dy: number };
export type TensorOffsets = Record<string, TensorOffset>;
/**
 * One drawn tile. The tensor travels with the part rather than sitting above
 * the list, so tiles on different tensors coexist: comparing what two tensors
 * pull from a shared input is the reason the tool exists, and it cannot be done
 * if drawing on B discards the tile on A.
 */
export type SelPart = { tensorId: string; box: Box };
/**
 * The user's ordered parts (identity-stable, may overlap, may span tensors),
 * never a canonicalized set. See the note in core/region.ts.
 */
type Selection = { parts: SelPart[] } | null;

/** Parts drawn on one tensor, carrying the global index each one keeps. */
export function partsOn(
  selection: Selection,
  tensorId: string
): { index: number; box: Box }[] {
  if (!selection) return [];
  const out: { index: number; box: Box }[] = [];
  selection.parts.forEach((p, index) => {
    if (p.tensorId === tensorId) out.push({ index, box: p.box });
  });
  return out;
}

/** Distinct tensors carrying at least one part, in first-drawn order. */
export function selectedTensorIds(selection: Selection): string[] {
  const out: string[] = [];
  for (const p of selection?.parts ?? []) if (!out.includes(p.tensorId)) out.push(p.tensorId);
  return out;
}

/**
 * The tensor a whole-selection action applies to: the focused part's tensor,
 * else the most recently drawn one. Arrow keys resolve their axis indices
 * against one shape, and with parts on tensors of different rank there is no
 * single axis that means the same thing everywhere.
 */
export function anchorTensorId(selection: Selection, focusedBox: number | null): string | null {
  const parts = selection?.parts ?? [];
  if (!parts.length) return null;
  if (focusedBox !== null && parts[focusedBox]) return parts[focusedBox].tensorId;
  return parts[parts.length - 1].tensorId;
}
type WorkspaceSnapshot = { selection: Selection; tensorOffsets: TensorOffsets };

/** Panel geometry. VS Code semantics: drag to resize between the bounds, drag
 * far enough inward to collapse, click the rail to bring it back. */
export const PANEL_MIN = 232;
export const PANEL_MAX = 560;
/** Release below this and the panel collapses rather than clamping to the min. */
export const PANEL_COLLAPSE_AT = 168;
/** Width of the collapsed rail. */
export const PANEL_RAIL = 30;

/** Direct canvas gestures add by default; Alt subtracts. "replace" is internal
 * for examples, restored URL state, and operation-list probes. */
type Compose = "union" | "subtract" | "replace";

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
  selection: Selection;
  /** Chronological snapshots shared by selection edits and tensor moves. */
  workspaceHistory: WorkspaceSnapshot[];
  direction: Direction;
  theme: Theme;
  backwardRes: PropResult | null;
  forwardRes: PropResult | null;
  /** One propagation per selection box, so a highlighted region can be traced
   * back to the box that produced it. Null when there are too many boxes. */
  perBox: BoxProp[] | null;
  /** The part currently highlighted: the pinned one, else the hovered one. */
  focusedBox: number | null;
  /** Sticky focus set by clicking a part; survives the pointer leaving the row. */
  pinnedBox: number | null;
  /**
   * Parts whose dependency cone is not painted. Visibility is a separate concern
   * from focus: focus is *which row is emphasised* (one at a time, transient),
   * while this is *which cones contribute paint* (any number, sticky). Keeping
   * them apart lets a probe be parked — its numbers stay live in the footprint
   * table — without its cone competing for the canvas.
   *
   * Indexes into `selection.region.boxes`, so it is cleared by every edit that
   * can renumber the parts, exactly where `focusedBox` is cleared. Only a move
   * preserves it, because a move preserves order and length.
   */
  hiddenBoxes: Set<number>;
  /** True while any drag is in progress — a card rubber-band or a canvas pan —
   * so Escape can cancel the band and text selection can be suppressed. */
  dragging: boolean;
  preview: PropResult | null; // hover preview (backward only)

  viewCfgs: Record<string, ViewCfg>;
  /** Px per element for every card in this graph. A property of the resolved
   * graph, not of the view: derived once at load, so equal dimensions render at
   * equal lengths and no card resizes as a side effect of a view setting.
   * View controls never recompute this value. */
  graphPx: number;
  /** Global detail setting: shifts every tensor's tile by 2^tileScale. */
  tileScale: number;
  /**
   * Whether a drawn box is expanded to whole tiles. On, a drag reads as "these
   * cells", which is what the drawn lattice invites. Off, it cuts an arbitrary
   * element range — the same reach the inspector's range field already has, but
   * from the gesture. Analysis is unaffected either way: regions have always
   * been element-precise, only the gesture rounded.
   */
  snapToGrid: boolean;
  countIntermediates: boolean;
  focusTensor: string | null;
  /** Width in px of each side panel when open, and whether it is collapsed to a
   * rail. Collapsing keeps the remembered width so reopening restores it. */
  panelW: { left: number; right: number };
  panelCollapsed: { left: boolean; right: boolean };
  /** User displacement from dagre's collision-free base placement. */
  tensorOffsets: TensorOffsets;

  loadExample: (i: number) => void;
  applyDSL: (text: string) => void;
  setSelection: (tensorId: string, region: Region, compose?: Compose) => void;
  /** Install an exact part list. Used by link restore, where part *order* is
   * the payload: it is what assigns hues and footprint rows. */
  restoreSelection: (parts: SelPart[]) => void;
  clearSelection: () => void;
  undoWorkspace: () => void;
  /** Move the whole selection along one axis, clamped to the tensor.
   * `record` false appends no undo entry — used for auto-repeat, so holding an
   * arrow key is one undo step rather than forty. */
  moveSelection: (axis: number, delta: number, record?: boolean) => void;
  /** Replace one ordered selection part without renumbering its peers. */
  replaceBox: (index: number, box: Box) => void;
  deleteBox: (index: number) => void;
  /** Transient hover focus; ignored while a part is pinned. */
  hoverBox: (index: number | null) => void;
  /** Click a part to pin it, or the same part again to unpin. */
  togglePinBox: (index: number) => void;
  /** Drop both hover and pin. What Escape does. */
  clearFocus: () => void;
  /** Show/hide one part's cone, leaving its metrics untouched. */
  toggleBoxHidden: (index: number) => void;
  setDragging: (v: boolean) => void;
  setDirection: (d: Direction) => void;
  toggleDirection: (d: ConeDirection) => void;
  setTheme: (theme: Theme) => void;
  setViewCfg: (tensorId: string, cfg: Partial<ViewCfg>) => void;
  setTileScale: (v: number) => void;
  setSnapToGrid: (v: boolean) => void;
  setCountIntermediates: (v: boolean) => void;
  setFocusTensor: (id: string | null) => void;
  /** Preview a panel resize, clamped to the usable open range. */
  setPanelWidth: (side: PanelSide, w: number) => void;
  /** Commit a resize. A raw width under `PANEL_COLLAPSE_AT` collapses here,
   * after pointer capture has delivered the release event. */
  finishPanelResize: (side: PanelSide, w: number) => void;
  togglePanel: (side: PanelSide) => void;
  /** Live, unrecorded movement used while a drag owns pointer capture. */
  setTensorOffset: (tensorId: string, offset: TensorOffset) => void;
  /** Record one completed drag, restoring `before` when workspace undo runs. */
  commitTensorMove: (tensorId: string, before: TensorOffset) => void;
  setPreviewCell: (tensorId: string | null, cell?: number[]) => void;
  expandNodeInPlace: (nodeId: string) => void;
};

/** Per-selection-box propagation results, aligned with `selection.region.boxes`. */
export type BoxProp = { backward: PropResult | null; forward: PropResult | null };

/** Above this many boxes, per-box attribution costs more than it is worth. */
export const MAX_PER_BOX_PROPS = 12;

function initialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  try {
    const saved = window.localStorage.getItem("tilecone.theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // Storage is optional; the OS preference remains a complete fallback.
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/**
 * One propagation per part, merged into the aggregate the panels read.
 *
 * The executor stays a single-root primitive — a cone is defined from one
 * tensor — and multiplicity lives here, where it belongs: the workspace is what
 * holds several probes at once. Merging is a per-tensor union, which is also
 * what the propagator already does internally when two paths reconverge.
 *
 * Above `MAX_PER_BOX_PROPS` parts, per-part attribution is dropped and the
 * queries are grouped by tensor instead, so the cost is bounded by the number
 * of tensors drawn on rather than the number of tiles. Note that the grouped
 * result can only be equal or coarser than the per-part one: propagating a
 * union through an over-approximating op is never tighter than unioning the
 * separate propagations. Neither can under-approximate.
 */
function recompute(
  resolved: ResolvedGraph | null,
  selection: Selection,
  direction: Direction
): Pick<State, "backwardRes" | "forwardRes" | "perBox"> {
  const none = { backwardRes: null, forwardRes: null, perBox: null };
  if (!resolved || !selection || selection.parts.length === 0) return none;
  if (direction === "none") return none;

  const parts = selection.parts;
  const backs: PropResult[] = [];
  const fwds: PropResult[] = [];
  let perBox: BoxProp[] | null = null;

  if (parts.length <= MAX_PER_BOX_PROPS) {
    perBox = parts.map((p) => {
      const r = executeQuery(resolved, {
        tensorId: p.tensorId,
        region: fromBox(p.box),
        direction,
      });
      if (r.backward) backs.push(r.backward);
      if (r.forward) fwds.push(r.forward);
      return { backward: r.backward, forward: r.forward };
    });
  } else {
    const byTensor = new Map<string, Box[]>();
    for (const p of parts) {
      const cur = byTensor.get(p.tensorId);
      if (cur) cur.push(p.box);
      else byTensor.set(p.tensorId, [p.box]);
    }
    for (const [tensorId, boxes] of byTensor) {
      const r = executeQuery(resolved, {
        tensorId,
        region: { boxes, exact: true, reasons: [] },
        direction,
      });
      if (r.backward) backs.push(r.backward);
      if (r.forward) fwds.push(r.forward);
    }
  }
  return { backwardRes: mergeProps(backs), forwardRes: mergeProps(fwds), perBox };
}

/**
 * Apply a transform to the selection's parts and repropagate.
 * `keepFocus` holds the focused part across edits that preserve indices (a move);
 * edits that reorder or remove parts drop it so a stale index can never be used.
 */
function editSelection(
  get: () => State,
  set: (partial: Partial<State>) => void,
  fn: (parts: SelPart[], shapeOf: (tensorId: string) => number[]) => SelPart[],
  keepFocus = false,
  record = true
): void {
  const { selection, resolved, direction, workspaceHistory, tensorOffsets, focusedBox } = get();
  if (!selection || !resolved) return;
  const shapeOf = (tensorId: string) => resolved.tensors[tensorId].resolved!;
  const parts = fn(selection.parts, shapeOf);
  const sel = parts.length === 0 ? null : { parts };
  const nextFocus =
    keepFocus && sel && focusedBox !== null && focusedBox < parts.length ? focusedBox : null;
  set({
    selection: sel,
    workspaceHistory: record
      ? [...workspaceHistory, { selection, tensorOffsets }].slice(-40)
      : workspaceHistory,
    focusedBox: nextFocus,
    pinnedBox: nextFocus === null ? null : get().pinnedBox,
    // `keepFocus` marks the edits that preserve part order and length (a move).
    // Anything else can renumber the parts, which would leave these indexes
    // pointing at the wrong cone.
    hiddenBoxes: keepFocus ? get().hiddenBoxes : new Set<number>(),
    preview: null,
    ...recompute(resolved, sel, direction),
  });
}

/** The 2-D planes every card will draw, in the fixed row-major projection. */
export function planesOf(resolved: ResolvedGraph): { rows: number; cols: number }[] {
  return Object.values(resolved.tensors).map((t) => {
    const shape = t.resolved!;
    const { rowAxis, colAxis } = viewAxes(shape);
    return planeExtents(shape, rowAxis, colAxis);
  });
}

function loadResolvedGraph(graph: Graph, resolved: ResolvedGraph): Pick<
  State,
  | "graph" | "resolved" | "loadError" | "selection" | "backwardRes" | "forwardRes"
  | "perBox" | "focusedBox" | "pinnedBox" | "viewCfgs" | "preview" | "graphPx"
  | "hiddenBoxes" | "workspaceHistory" | "tensorOffsets"
> {
  const viewCfgs: Record<string, ViewCfg> = {};
  for (const t of Object.values(resolved.tensors)) viewCfgs[t.id] = defaultViewCfg(t.resolved!);
  return {
    graph,
    resolved,
    loadError: null,
    selection: null,
    // Undo entries refer to tensor IDs and coordinates in one resolved graph.
    // They must never survive a graph replacement or composite rewrite.
    workspaceHistory: [],
    tensorOffsets: {},
    backwardRes: null,
    forwardRes: null,
    perBox: null,
    focusedBox: null,
    pinnedBox: null,
    hiddenBoxes: new Set<number>(),
    preview: null,
    viewCfgs,
    graphPx: graphScale(planesOf(resolved)),
  };
}

function loadGraph(graph: Graph): ReturnType<typeof loadResolvedGraph> {
  return loadResolvedGraph(graph, resolveGraph(graph));
}

export const useStore = create<State>((set, get) => ({
  dslText: EXAMPLES[0].dsl,
  exampleIndex: 0,
  graph: null,
  resolved: null,
  loadError: null,
  selection: null,
  workspaceHistory: [],
  direction: "both",
  theme: initialTheme(),
  backwardRes: null,
  forwardRes: null,
  perBox: null,
  focusedBox: null,
  pinnedBox: null,
  hiddenBoxes: new Set<number>(),
  dragging: false,
  preview: null,
  viewCfgs: {},
  graphPx: MAX_ELEM_PX,
  tileScale: 0,
  snapToGrid: true,
  countIntermediates: false,
  focusTensor: null,
  panelW: { left: 330, right: 300 },
  panelCollapsed: { left: false, right: false },
  tensorOffsets: {},

  loadExample: (i) => {
    const ex = EXAMPLES[i];
    try {
      const program = compileDSL(ex.dsl);
      const base = loadResolvedGraph(program.graph, program.resolved);
      const st: Partial<State> = { ...base, dslText: ex.dsl, exampleIndex: i, focusTensor: null };
      if (ex.defaultSelection && base.resolved) {
        st.selection = {
          parts: [
            {
              tensorId: ex.defaultSelection.tensor,
              box: ex.defaultSelection.box.map(([lo, hi]) => ({ lo, hi })),
            },
          ],
        };
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
      const program = compileDSL(text);
      set({ ...loadResolvedGraph(program.graph, program.resolved), dslText: text });
    } catch (e) {
      set({ loadError: (e as Error).message });
    }
  },

  setSelection: (tensorId, region, compose) => {
    const { selection, resolved, direction, workspaceHistory, tensorOffsets } = get();
    const mode = compose ?? "union";
    const drawn = region.boxes;

    let parts: SelPart[];
    if (mode === "replace" || !selection) {
      parts = drawn.map((box) => ({ tensorId, box }));
    } else {
      // Compose against this tensor's own parts only. Parts on other tensors
      // are untouched: drawing on B is an addition to the workspace, not a
      // replacement of it, and a subtract gesture on B cannot reach into A.
      let mine = partsOn(selection, tensorId).map((p) => p.box);
      for (const b of drawn)
        mine = mode === "union" ? addPart(mine, b) : subtractFromParts(mine, b);
      // Rebuild in place so untouched parts keep their index -- and with it
      // their hue, their footprint row, and any hidden/pinned state.
      parts = [];
      let k = 0;
      for (const p of selection.parts) {
        if (p.tensorId !== tensorId) parts.push(p);
        else if (k < mine.length) parts.push({ tensorId, box: mine[k++] });
      }
      for (; k < mine.length; k++) parts.push({ tensorId, box: mine[k] });
    }
    const sel = parts.length === 0 ? null : { parts };
    set({
      selection: sel,
      // Null is a real workspace state: the first selection must be undoable
      // without also rewinding an earlier tensor move.
      workspaceHistory: [...workspaceHistory, { selection, tensorOffsets }].slice(-40),
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: new Set<number>(),
      preview: null,
      ...recompute(resolved, sel, direction),
    });
  },

  restoreSelection: (parts) => {
    const { resolved, direction } = get();
    const sel = parts.length === 0 ? null : { parts };
    set({
      selection: sel,
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: new Set<number>(),
      preview: null,
      ...recompute(resolved, sel, direction),
    });
  },

  clearSelection: () => {
    const { selection, workspaceHistory, tensorOffsets } = get();
    set({
      selection: null,
      workspaceHistory: selection
        ? [...workspaceHistory, { selection, tensorOffsets }].slice(-40)
        : workspaceHistory,
      backwardRes: null,
      forwardRes: null,
      perBox: null,
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: new Set<number>(),
      preview: null,
    });
  },

  undoWorkspace: () => {
    const { workspaceHistory, resolved, direction } = get();
    if (!workspaceHistory.length) return;
    const prev = workspaceHistory[workspaceHistory.length - 1];
    set({
      selection: prev.selection,
      tensorOffsets: prev.tensorOffsets,
      workspaceHistory: workspaceHistory.slice(0, -1),
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: new Set<number>(),
      preview: null,
      ...recompute(resolved, prev.selection, direction),
    });
  },

  /**
   * Moves the focused part when one is focused, otherwise every part on the
   * anchor tensor. `axis` is an index into one tensor's shape, so it cannot be
   * applied across tensors of different rank -- parts elsewhere hold still.
   */
  moveSelection: (axis, delta, record = true) => {
    const focused = get().focusedBox;
    editSelection(
      get,
      set,
      (parts, shapeOf) => {
        const anchor = anchorTensorId({ parts }, focused);
        if (!anchor) return parts;
        const shape = shapeOf(anchor);
        const local: number[] = [];
        const boxes: Box[] = [];
        parts.forEach((p, i) => {
          if (p.tensorId === anchor) {
            local.push(i);
            boxes.push(p.box);
          }
        });
        const at = focused !== null ? local.indexOf(focused) : -1;
        const moved =
          at >= 0
            ? translatePart(boxes, at, axis, delta, shape)
            : translateAllParts(boxes, axis, delta, shape);
        if (moved === boxes) return parts;
        const next = parts.slice();
        local.forEach((globalIndex, j) => {
          next[globalIndex] = { tensorId: anchor, box: moved[j] };
        });
        return next;
      },
      true,
      record
    );
  },

  replaceBox: (index, box) =>
    editSelection(
      get,
      set,
      (parts) => parts.map((part, i) => (i === index ? { ...part, box } : part)),
      true
    ),

  deleteBox: (index) =>
    editSelection(get, set, (parts) => parts.filter((_, i) => i !== index)),

  hoverBox: (index) => {
    if (get().pinnedBox !== null) return; // a pinned part outranks hovering
    set({ focusedBox: index });
  },

  togglePinBox: (index) => {
    const pinned = get().pinnedBox === index ? null : index;
    set({ pinnedBox: pinned, focusedBox: pinned });
  },

  clearFocus: () => set({ pinnedBox: null, focusedBox: null }),

  toggleBoxHidden: (index) => {
    const next = new Set(get().hiddenBoxes);
    if (!next.delete(index)) next.add(index);
    set({ hiddenBoxes: next });
  },

  setDragging: (v) => set({ dragging: v }),

  setDirection: (d) => {
    const { resolved, selection } = get();
    set({ direction: d, ...recompute(resolved, selection, d) });
  },

  toggleDirection: (axis) => {
    const { direction, resolved, selection } = get();
    const backward = direction === "backward" || direction === "both";
    const forward = direction === "forward" || direction === "both";
    const nextBackward = axis === "backward" ? !backward : backward;
    const nextForward = axis === "forward" ? !forward : forward;
    const next: Direction = nextBackward
      ? nextForward ? "both" : "backward"
      : nextForward ? "forward" : "none";
    set({ direction: next, ...recompute(resolved, selection, next) });
  },

  setTheme: (theme) => set({ theme }),

  setViewCfg: (tensorId, cfg) =>
    set((s) => ({ viewCfgs: { ...s.viewCfgs, [tensorId]: { ...s.viewCfgs[tensorId], ...cfg } } })),

  setSnapToGrid: (v) => set({ snapToGrid: v }),

  setTileScale: (v) =>
    set({ tileScale: Math.max(TILE_SCALE_MIN, Math.min(TILE_SCALE_MAX, Math.round(v))) }),

  setCountIntermediates: (v) => set({ countIntermediates: v }),
  setFocusTensor: (id) => set({ focusTensor: id }),

  setPanelWidth: (side, w) => {
    const { panelW, panelCollapsed } = get();
    const clamped = Math.round(Math.max(PANEL_MIN, Math.min(PANEL_MAX, w)));
    set({
      panelW: { ...panelW, [side]: clamped },
      panelCollapsed: { ...panelCollapsed, [side]: false },
    });
  },

  finishPanelResize: (side, w) => {
    const { panelW, panelCollapsed } = get();
    if (w < PANEL_COLLAPSE_AT) {
      // The preview never wrote an unusably narrow width, so the last open
      // width remains available when the rail is reopened.
      set({ panelCollapsed: { ...panelCollapsed, [side]: true } });
      return;
    }
    const clamped = Math.round(Math.max(PANEL_MIN, Math.min(PANEL_MAX, w)));
    set({
      panelW: { ...panelW, [side]: clamped },
      panelCollapsed: { ...panelCollapsed, [side]: false },
    });
  },

  togglePanel: (side) => {
    const { panelW, panelCollapsed } = get();
    const collapsed = !panelCollapsed[side];
    set({
      panelCollapsed: { ...panelCollapsed, [side]: collapsed },
      // reopening a panel that was dragged very narrow must still be usable
      panelW: collapsed ? panelW : { ...panelW, [side]: Math.max(PANEL_MIN, panelW[side]) },
    });
  },

  setTensorOffset: (tensorId, offset) => {
    const tensorOffsets = { ...get().tensorOffsets };
    if (Math.abs(offset.dx) < 1e-6 && Math.abs(offset.dy) < 1e-6) delete tensorOffsets[tensorId];
    else tensorOffsets[tensorId] = offset;
    set({ tensorOffsets });
  },

  commitTensorMove: (tensorId, before) => {
    const { selection, tensorOffsets, workspaceHistory } = get();
    const after = tensorOffsets[tensorId] ?? { dx: 0, dy: 0 };
    if (Math.abs(after.dx - before.dx) < 1e-6 && Math.abs(after.dy - before.dy) < 1e-6) return;
    const previousOffsets = { ...tensorOffsets };
    if (Math.abs(before.dx) < 1e-6 && Math.abs(before.dy) < 1e-6) delete previousOffsets[tensorId];
    else previousOffsets[tensorId] = before;
    set({
      workspaceHistory: [...workspaceHistory, { selection, tensorOffsets: previousOffsets }].slice(-40),
    });
  },

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
      set({
        preview: executeQuery(resolved, { tensorId, region, direction: "backward" }).backward,
      });
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
