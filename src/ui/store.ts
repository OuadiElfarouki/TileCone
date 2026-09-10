import { create } from "zustand";
import { executeQuery, validateSelection } from "../core/executor";
import { Entanglement, entangledWith } from "../core/entangle";
import { Graph, graphOutputs, ResolvedGraph } from "../core/graph";
import { expandNode } from "../core/expand";
import { PropResult, mergeProps } from "../core/propagate";
import {
  Box,
  Region,
  addPart,
  fromBox,
  subtractFromParts,
  translateAllParts,
  translatePart,
} from "../core/region";
import { EXAMPLES } from "../examples/index";
import type { CompilerDiagnostic } from "../parse/compiler";
import { compileDSL, tryCompileDSL } from "../parse/compiler";
import { toDSL } from "../parse/dsl";
import { graphScale, MAX_ELEM_PX, planeExtents, TILE_SCALE_MAX, TILE_SCALE_MIN } from "./tiling";
import { tileOf } from "./grid";
import type { AxisMode } from "./shape-label";
import type { TensorOffset, TensorOffsets } from "./tensor-layout";
import { defaultViewCfg, viewAxes, viewCfgFits, type ViewCfg } from "./tensor-view";

/** Which independently toggled views are active in the workspace. `none` is
 * the explicit figures-only state: analysis remains live while paint and rows hide. */
export type Direction = "none" | "backward" | "forward" | "both";
export type ConeDirection = "backward" | "forward";
export type PanelSide = "left" | "right";
export type Theme = "light" | "dark";
/** Defensive share-state bound; far beyond any usable graph arrangement while
 * preventing finite-but-overflowing coordinates from poisoning scene bounds. */
export const MAX_TENSOR_OFFSET = 1_000_000;
/**
 * One drawn tile. The tensor travels with the part rather than sitting above
 * the list, so tiles on different tensors coexist: comparing what two tensors
 * pull from a shared input is the reason the tool exists, and it cannot be done
 * if drawing on B discards the tile on A.
 */
export type SelPart = { tensorId: string; box: Box };

/**
 * Scope an aggregate result to enabled tiles, optionally to one focused tile.
 * Per-tile propagation stays cached; toggling visibility only re-merges those
 * results and never reruns the symbolic executor.
 */
export function enabledPropResult(
  aggregate: PropResult | null,
  perBox: BoxProp[] | null,
  hiddenBoxes: Set<number>,
  focusedBox: number | null,
  direction: ConeDirection
): PropResult | null {
  if (!perBox) return aggregate;
  if (focusedBox !== null && !hiddenBoxes.has(focusedBox))
    return perBox[focusedBox]?.[direction] ?? null;
  return mergeProps(
    perBox.flatMap((prop, index) =>
      hiddenBoxes.has(index) || !prop[direction] ? [] : [prop[direction]!]
    )
  );
}
/**
 * The user's ordered parts (identity-stable, may overlap, may span tensors),
 * never a canonicalized set. See the note in core/region.ts.
 */
type Selection = { parts: SelPart[] } | null;

/**
 * One tensor's enabled tiles as a single cone, for the inspector's analysis.
 *
 * `enabledPropResult` merges every enabled tile whatever it sits on, which is
 * the right scope for the canvas and the operations list: those describe the
 * selection. The inspector deliberately scopes cost to one output tensor's
 * tiles. Multi-output jobs could be modeled too, but require explicit output
 * and materialization semantics; union itself does not double-count work.
 *
 * Past `MAX_PER_BOX_PROPS` there are no per-tile cones to filter and the
 * grouped per-tensor query stands in - coarser, never mixed.
 */
export function groupPropResult(
  byTensorRes: Record<string, { backward: PropResult | null; forward: PropResult | null }> | null,
  perBox: BoxProp[] | null,
  parts: SelPart[],
  hiddenBoxes: Set<number>,
  focusedBox: number | null,
  tensorId: string | null,
  direction: ConeDirection
): PropResult | null {
  if (!tensorId) return null;
  if (!perBox) return byTensorRes?.[tensorId]?.[direction] ?? null;
  if (
    focusedBox !== null &&
    !hiddenBoxes.has(focusedBox) &&
    parts[focusedBox]?.tensorId === tensorId
  )
    return perBox[focusedBox]?.[direction] ?? null;
  return mergeProps(
    perBox.flatMap((prop, index) =>
      hiddenBoxes.has(index) || parts[index]?.tensorId !== tensorId || !prop[direction]
        ? []
        : [prop[direction]!]
    )
  );
}

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
/**
 * The operation a tile on this tensor is "at", for the operations list.
 *
 * Its producer, because that is the operation the tensor *is* the result of. A
 * graph input has no producer, and then the only honest answer is its consumer
 * when there is exactly one - with several, no single row is the one the reader
 * is looking at, and lighting an arbitrary one would be a guess presented as a
 * fact. `null` leaves the list unhighlighted, which is a true statement.
 */
export function operationForTensor(
  graph: ResolvedGraph | null,
  tensorId: string | null
): string | null {
  if (!graph || !tensorId) return null;
  const producer = graph.tensors[tensorId]?.producer;
  if (producer) return producer.nodeId;
  const consumers = graph.consumers[tensorId] ?? [];
  const distinct = [...new Set(consumers.map((c) => c.nodeId))];
  return distinct.length === 1 ? distinct[0] : null;
}

type WorkspaceSnapshot = {
  selection: Selection;
  tensorOffsets: TensorOffsets;
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
  source?: { dslText: string; draftText: string; graph: Graph; exampleIndex: number };
};
const WORKSPACE_HISTORY_LIMIT = 40;

function appendWorkspaceHistory(
  history: WorkspaceSnapshot[],
  snapshot: WorkspaceSnapshot
): WorkspaceSnapshot[] {
  return [...history, snapshot].slice(-WORKSPACE_HISTORY_LIMIT);
}

type WorkspaceRestore = {
  dsl: string;
  direction: Direction;
  /** Optional for callers restoring links written before entanglement existed. */
  showEntangled?: boolean;
  tileScale: number;
  snapToGrid: boolean;
  /** Whether a compact shape reads as semantic labels or numeric extents. */
  axisMode: AxisMode;
  tensorOffsets?: TensorOffsets;
  viewCfgs?: Record<string, ViewCfg>;
  parts: SelPart[] | null;
};

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

type State = {
  /** Source currently installed in `graph`/`resolved` and encoded by Share. */
  dslText: string;
  /** Editable source. It may differ from `dslText` while the built workspace
   * remains live; a successful `applyDSL` advances both together. */
  draftText: string;
  exampleIndex: number;
  graph: Graph | null;
  resolved: ResolvedGraph | null;
  loadError: string | null;
  /**
   * Every diagnostic from the last failed compile, in source order.
   *
   * `loadError` remains the first one's message because callers outside the
   * source panel only want "did it fail, and why" in one line. This is the
   * list the editor shows: the compiler reports all independent errors in one
   * pass, and reducing that to the first would put the author back on the
   * fix-one-recompile loop the collecting phases exist to end.
   */
  diagnostics: CompilerDiagnostic[];

  /** `region.boxes` are the user's ordered PARTS (identity-stable, may overlap),
   * never a canonicalized set. See the note in core/region.ts. */
  selection: Selection;
  /** Chronological snapshots shared by selection edits and tensor moves. */
  workspaceHistory: WorkspaceSnapshot[];
  direction: Direction;
  theme: Theme;
  backwardRes: PropResult | null;
  /**
   * One cone per tensor drawn on, so the inspector can analyse a tensor's
   * tiles without mixing in tiles that sit on a different tensor. Available in
   * both regimes: derived from `perBox` under the attribution cap, and taken
   * straight from the grouped queries above it, where per-tile results do not
   * exist to filter.
   */
  byTensorRes: Record<string, { backward: PropResult | null; forward: PropResult | null }> | null;
  forwardRes: PropResult | null;
  /** One propagation per selection box, so a highlighted region can be traced
   * back to the box that produced it. Null when there are too many boxes. */
  perBox: BoxProp[] | null;
  /** The part currently highlighted: the pinned one, else the hovered one. */
  focusedBox: number | null;
  /** Sticky focus set by clicking a part; survives the pointer leaving the row. */
  pinnedBox: number | null;
  /**
   * Parts excluded from merged analysis and dependency paint. Focus is still a
   * separate transient choice, but hiding the focused part clears that focus so
   * the inspector cannot name a tile it is no longer analysing.
   *
   * Indexes into `selection.parts`. Edits clear metadata for affected parts and
   * remap it by object identity for untouched parts on other tensors.
   */
  hiddenBoxes: Set<number>;
  /** True while any drag is in progress : a card rubber-band or a canvas pan :
   * so Escape can cancel the band and text selection can be suppressed. */
  dragging: boolean;
  preview: { backward: PropResult | null; forward: PropResult | null } | null; // bidirectional hover probe
  /**
   * What the selection is combined with, per selected tile.
   *
   * A separate toggle rather than a fourth `Direction`, because entanglement is
   * orthogonal to the cone: "what this tile reads" and "what it is multiplied
   * against" are both worth seeing at once, and folding them into one control
   * would make them alternatives.
   */
  showEntangled: boolean;
  /** Parallel to `selection.parts`; null without a selection or past the attribution cap. */
  entangled: Entanglement[][] | null;

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
   * element range : the same reach the inspector's range field already has, but
   * from the gesture. Analysis is unaffected either way: regions have always
   * been element-precise, only the gesture rounded.
   */
  snapToGrid: boolean;
  /**
   * Whether a compact shape reads as semantic labels or numeric extents. The
   * tensor details retain axis identity, symbolic extent, and bound extent as
   * separate facts without making every graph label carry all three.
   */
  axisMode: AxisMode;
  /** A one-shot request to bring a node into view. Carries the kind because an
   * operation row centres its operator, not the tensor it writes. Consumed by
   * the viewport and cleared, so asking twice acts twice. */
  focusNode: { kind: "tensor" | "op"; id: string } | null;
  /**
   * The row standing highlighted in the operations list.
   *
   * Distinct from `focusNode`, which is a one-shot "glide the viewport here"
   * request and clears the moment the glide lands. This one persists: it is the
   * operation the reader is currently working at, set from either end - click a
   * row and a starter tile appears on its output; touch a tile and its row
   * lights. A viewport gesture clears it, because panning away is the reader
   * saying they are looking somewhere else now.
   */
  selectedOp: string | null;
  /** Width in px of each side panel when open, and whether it is collapsed to a
   * rail. Collapsing keeps the remembered width so reopening restores it. */
  panelW: { left: number; right: number };
  panelCollapsed: { left: boolean; right: boolean };
  /** User displacement from dagre's collision-free base placement. */
  tensorOffsets: TensorOffsets;

  /** Compile and install an example immediately: app boot and tests. */
  loadExample: (i: number) => void;
  /** Put an example in the editor without replacing the built workspace. */
  stageExample: (i: number) => void;
  setDraftText: (text: string) => void;
  applyDSL: (text: string) => void;
  /** Compile, validate, and install a shared workspace as one transaction. */
  restoreWorkspace: (workspace: WorkspaceRestore) => boolean;
  setSelection: (tensorId: string, region: Region, compose?: Compose) => void;
  clearSelection: () => void;
  undoWorkspace: () => void;
  /** Move the whole selection along one axis, clamped to the tensor.
   * `record` false appends no undo entry : used for auto-repeat, so holding an
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
  /** Include/exclude one part from merged analysis and dependency paint. */
  toggleBoxHidden: (index: number) => void;
  setDragging: (v: boolean) => void;
  setDirection: (d: Direction) => void;
  toggleDirection: (d: ConeDirection) => void;
  setTheme: (theme: Theme) => void;
  setViewCfg: (tensorId: string, cfg: Partial<ViewCfg>) => void;
  setTileScale: (v: number) => void;
  setSnapToGrid: (v: boolean) => void;
  setAxisMode: (v: AxisMode) => void;
  setFocusNode: (node: { kind: "tensor" | "op"; id: string } | null) => void;
  /** Light one row of the operations list, or clear it with `null`. */
  setSelectedOp: (nodeId: string | null) => void;
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
  /** Restore Dagre's generated placement as one undoable workspace action. */
  resetTensorLayout: () => void;
  /** Preview the cone of the box a click would commit, not of one element:
   * a projection gesture selects whole hidden axes, so a cell-sized preview
   * would understate the cone the same gesture goes on to produce. */
  setPreviewBox: (tensorId: string | null, box?: Box, expectedView?: ViewCfg) => void;
  toggleEntangled: () => void;
  expandNodeInPlace: (nodeId: string) => void;
};

/** Per-part propagation results, aligned with `selection.parts`. */
export type BoxProp = { backward: PropResult | null; forward: PropResult | null };

/** Above this many boxes, per-box attribution costs more than it is worth. */
export const MAX_PER_BOX_PROPS = 12;

/**
 * Above this many nodes, hovering does not compute a preview cone.
 *
 * The bound is a frame budget, not a guess. A bidirectional query costs roughly
 * 3us per node, so a thousand nodes is about 6ms - comfortably inside a frame,
 * with the rest of it left for painting. The cap was previously less than half
 * this because the query ran once per *pointer event* rather than once per
 * frame, which on a high-polling-rate mouse is an order of magnitude more work
 * for the same picture; `useFrameThrottle` is what removed that multiplier.
 */
export const MAX_PREVIEW_NODES = 1000;

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
 * The executor stays a single-root primitive : a cone is defined from one
 * tensor : and multiplicity lives here, where it belongs: the workspace is what
 * holds several probes at once. Merging is a per-tensor union, which is also
 * what the propagator already does internally when two paths reconverge.
 *
 * Both cones are always computed. `direction` is a view filter over the result,
 * not a gate on producing it: the panel answers "what does this tile need" and
 * "what does it feed" from the same analysis, and a global mode should not
 * decide whether a number exists. Painting applies the filter instead.
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
  previous?: {
    selection: Selection;
    perBox: BoxProp[] | null;
    entangled: Entanglement[][] | null;
  }
): Pick<State, "backwardRes" | "forwardRes" | "perBox" | "entangled" | "byTensorRes"> {
  const none = {
    backwardRes: null, forwardRes: null, perBox: null, entangled: null, byTensorRes: null,
  };
  if (!resolved || !selection || selection.parts.length === 0) return none;

  const parts = selection.parts;
  const backs: PropResult[] = [];
  const fwds: PropResult[] = [];
  let perBox: BoxProp[] | null = null;
  const byTensorRes: State["byTensorRes"] = {};

  if (parts.length <= MAX_PER_BOX_PROPS) {
    // Geometry is the cache key rather than array position: deleting a part
    // renumbers its peers, and composition may recreate an equal SelPart
    // object. Reuse every unchanged cone and execute only new/edited parts.
    const cached = new Map<string, BoxProp[]>();
    const keyOf = (part: SelPart) =>
      `${part.tensorId}|${part.box.map((interval) => `${interval.lo}:${interval.hi}`).join(",")}`;
    if (previous?.selection && previous.perBox &&
        previous.selection.parts.length === previous.perBox.length) {
      previous.selection.parts.forEach((part, index) => {
        const key = keyOf(part);
        const entries = cached.get(key);
        if (entries) entries.push(previous.perBox![index]);
        else cached.set(key, [previous.perBox![index]]);
      });
    }
    perBox = parts.map((p) => {
      const hit = cached.get(keyOf(p))?.shift();
      if (hit) {
        if (hit.backward) backs.push(hit.backward);
        if (hit.forward) fwds.push(hit.forward);
        return hit;
      }
      const r = executeQuery(resolved, {
        tensorId: p.tensorId,
        region: fromBox(p.box),
        direction: "both",
      });
      if (r.backward) backs.push(r.backward);
      if (r.forward) fwds.push(r.forward);
      return { backward: r.backward, forward: r.forward };
    });
    // The same grouping the branch below gets for free. Merging per-part cones
    // that already exist is cheaper than re-querying, and both regimes have to
    // offer the inspector the same shape.
    const grouped = new Map<string, { backs: PropResult[]; fwds: PropResult[] }>();
    parts.forEach((part, index) => {
      const entry = grouped.get(part.tensorId) ?? { backs: [], fwds: [] };
      const prop = perBox![index];
      if (prop.backward) entry.backs.push(prop.backward);
      if (prop.forward) entry.fwds.push(prop.forward);
      grouped.set(part.tensorId, entry);
    });
    for (const [tensorId, entry] of grouped)
      byTensorRes[tensorId] = {
        backward: mergeProps(entry.backs),
        forward: mergeProps(entry.fwds),
      };
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
        direction: "both",
      });
      if (r.backward) backs.push(r.backward);
      if (r.forward) fwds.push(r.forward);
      byTensorRes[tensorId] = { backward: r.backward, forward: r.forward };
    }
  }
  // Entanglement is attributed per part for the same reason as the cones: hue
  // identifies the tile that produced a region. Respect the same cap, rather
  // than reintroducing unbounded synchronous work after cone attribution has
  // deliberately switched to a grouped query. Geometry is also cached so an
  // edit only recomputes the part that changed.
  let entangled: Entanglement[][] | null = null;
  if (parts.length <= MAX_PER_BOX_PROPS) {
    const keyOf = (part: SelPart) =>
      `${part.tensorId}|${part.box.map((interval) => `${interval.lo}:${interval.hi}`).join(",")}`;
    const cached = new Map<string, Entanglement[][]>();
    if (
      previous?.selection &&
      previous.entangled &&
      previous.selection.parts.length === previous.entangled.length
    ) {
      previous.selection.parts.forEach((part, index) => {
        const key = keyOf(part);
        const entries = cached.get(key);
        if (entries) entries.push(previous.entangled![index]);
        else cached.set(key, [previous.entangled![index]]);
      });
    }
    entangled = parts.map(
      (part) =>
        cached.get(keyOf(part))?.shift() ??
        entangledWith(resolved, part.tensorId, fromBox(part.box))
    );
  }
  return {
    backwardRes: mergeProps(backs),
    forwardRes: mergeProps(fwds),
    perBox,
    entangled,
    byTensorRes,
  };
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
  const {
    selection,
    resolved,
    workspaceHistory,
    tensorOffsets,
    focusedBox,
    perBox,
    entangled,
  } = get();
  if (!selection || !resolved) return;
  const shapeOf = (tensorId: string) => resolved.tensors[tensorId].resolved!;
  const parts = fn(selection.parts, shapeOf);
  const sel = parts.length === 0 ? null : { parts };
  const nextFocus =
    keepFocus && sel && focusedBox !== null && focusedBox < parts.length ? focusedBox : null;
  // Moving or editing a tile is working at its operation, so the list follows.
  // The anchor is the same tile the keyboard acts on, so the row that lights is
  // the one the reader is driving.
  const anchor = anchorTensorId(sel, nextFocus);
  set({
    selectedOp: operationForTensor(resolved, anchor),
    selection: sel,
    workspaceHistory: record
      ? appendWorkspaceHistory(workspaceHistory, { selection, tensorOffsets })
      : workspaceHistory,
    focusedBox: nextFocus,
    pinnedBox: nextFocus === null ? null : get().pinnedBox,
    // `keepFocus` marks the edits that preserve part order and length (a move).
    // Anything else can renumber the parts, which would leave these indexes
    // pointing at the wrong cone.
    hiddenBoxes: keepFocus ? get().hiddenBoxes : new Set<number>(),
    preview: null,
    ...recompute(resolved, sel, { selection, perBox, entangled }),
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

/**
 * Two tiles worth offering someone who has not drawn one: the graph's result,
 * and the last thing computed before it. They are the fastest path from a
 * loaded workspace to a cone worth reading, and both are one selection away.
 *
 * The box is one tile of the lattice currently drawn, so the offered tile is
 * the one the canvas would have snapped a click to.
 */
export function startingTiles(
  resolved: ResolvedGraph,
  tileScale: number,
  graphPx: number
): { label: string; tensorId: string; box: Box }[] {
  const output = graphOutputs(resolved)[0];
  if (!output) return [];
  const producer = resolved.nodes.find((node) => node.id === output.producer!.nodeId);
  const feeding = producer?.inputs.filter((id) => resolved.tensors[id].producer) ?? [];
  const previous = feeding.length ? resolved.tensors[feeding[feeding.length - 1]] : null;

  return [
    { tensor: output, label: "the output" },
    ...(previous ? [{ tensor: previous, label: "one step back" }] : []),
  ].map(({ tensor, label }) => {
    const shape = tensor.resolved!;
    const tile = tileOf(shape, tileScale, graphPx);
    const { rowAxis, colAxis } = viewAxes(shape);
    return {
      label,
      tensorId: tensor.id,
      box: shape.map((extent, axis) => ({
        lo: 0,
        hi: axis === rowAxis || axis === colAxis ? Math.min(tile, extent) : 1,
      })),
    };
  });
}

function loadResolvedGraph(graph: Graph, resolved: ResolvedGraph): Pick<
  State,
  | "graph" | "resolved" | "loadError" | "diagnostics" | "selection" | "backwardRes" | "forwardRes"
  | "byTensorRes"
  | "entangled"
  | "perBox" | "focusedBox" | "pinnedBox" | "viewCfgs" | "preview" | "graphPx"
  | "hiddenBoxes" | "workspaceHistory" | "tensorOffsets"
> {
  const viewCfgs: Record<string, ViewCfg> = {};
  for (const t of Object.values(resolved.tensors)) viewCfgs[t.id] = defaultViewCfg(t.resolved!);
  return {
    graph,
    resolved,
    loadError: null,
    diagnostics: [],
    entangled: null,
    selection: null,
    // Undo entries refer to tensor IDs and coordinates in one resolved graph.
    // They must never survive a graph replacement or composite rewrite.
    workspaceHistory: [],
    tensorOffsets: {},
    backwardRes: null,
    byTensorRes: null,
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

export const useStore = create<State>((set, get) => ({
  dslText: EXAMPLES[0].dsl,
  draftText: EXAMPLES[0].dsl,
  exampleIndex: 0,
  graph: null,
  resolved: null,
  loadError: null,
  diagnostics: [],
  showEntangled: false,
  entangled: null,
  selection: null,
  workspaceHistory: [],
  direction: "both",
  theme: initialTheme(),
  backwardRes: null,
  byTensorRes: null,
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
  // A new workspace opens on labels, because that is what the graph gained.
  // A shared link deliberately defaults the other way: see `share.ts`, where an
  // older payload keeps the numeric cards it was written against.
  axisMode: "symbolic",
  focusNode: null,
  selectedOp: null,
  panelW: { left: 330, right: 300 },
  panelCollapsed: { left: false, right: false },
  tensorOffsets: {},

  loadExample: (i) => get().applyDSL(EXAMPLES[i].dsl),

  stageExample: (i) => {
    set({
      draftText: EXAMPLES[i].dsl,
      loadError: null,
      diagnostics: [],
    });
  },

  setDraftText: (text) => set({ draftText: text }),

  applyDSL: (text) => {
    // `tryCompileDSL` rather than the throwing form: a thrown CompilationError
    // flattens to its first diagnostic's message, and the editor wants all of
    // them. Everything the compiler found in one pass reaches the panel.
    const result = tryCompileDSL(text);
    if (!result.ok) {
      set({
        draftText: text,
        diagnostics: result.diagnostics,
        loadError: `line ${result.diagnostics[0].span.start.line}: ${result.diagnostics[0].message}`,
      });
      return;
    }
    try {
      const program = result.program;
      const base = loadResolvedGraph(program.graph, program.resolved);
      // A source that is exactly a built-in example *is* that example, however
      // it got here - picked from the menu, restored from a link, or typed.
      // Deriving this from the text keeps the picker honest after an edit is
      // undone back to the original, which a remembered index could not.
      const exampleIndex = EXAMPLES.findIndex((ex) => ex.dsl === text);
      const example = exampleIndex >= 0 ? EXAMPLES[exampleIndex] : null;
      const st: Partial<State> = {
        ...base,
        dslText: text,
        draftText: text,
        exampleIndex,
        focusNode: null,
      };
      if (example?.defaultSelection && base.resolved) {
        st.selection = {
          parts: [
            {
              tensorId: example.defaultSelection.tensor,
              box: example.defaultSelection.box.map(([lo, hi]) => ({ lo, hi })),
            },
          ],
        };
        st.focusedBox = null;
        st.pinnedBox = null;
        Object.assign(st, recompute(base.resolved, st.selection!));
      }
      st.loadError = null;
      st.diagnostics = [];
      set(st as State);
    } catch (e) {
      // Compilation succeeded; anything failing here is a workspace-build
      // problem with no source span to attach it to.
      set({ draftText: text, loadError: (e as Error).message, diagnostics: [] });
    }
  },

  restoreWorkspace: ({
    dsl,
    direction,
    showEntangled,
    tileScale,
    snapToGrid,
    axisMode,
    tensorOffsets,
    viewCfgs,
    parts,
  }) => {
    try {
      if (
        !["none", "backward", "forward", "both"].includes(direction) ||
        (showEntangled !== undefined && typeof showEntangled !== "boolean") ||
        !Number.isFinite(tileScale) ||
        typeof snapToGrid !== "boolean" ||
        !["symbolic", "numeric"].includes(axisMode)
      ) return false;
      const program = compileDSL(dsl);
      const base = loadResolvedGraph(program.graph, program.resolved);
      for (const [id, cfg] of Object.entries(viewCfgs ?? {})) {
        const shape = program.resolved.tensors[id]?.resolved;
        if (!shape || !viewCfgFits(shape, cfg)) throw new Error(`invalid view for tensor "${id}"`);
        base.viewCfgs[id] = { projection: cfg.projection, sliders: cfg.sliders.slice() };
      }
      const checkedParts = (parts ?? []).map((part) => {
        const checked = validateSelection(program.resolved, {
          tensorId: part.tensorId,
          region: fromBox(part.box),
        });
        if (checked.region.boxes.length !== 1)
          throw new Error(`selection on tensor "${part.tensorId}" is empty`);
        return { tensorId: checked.tensorId, box: checked.region.boxes[0] };
      });
      const selection = checkedParts.length ? { parts: checkedParts } : null;
      const checkedOffsets: TensorOffsets = {};
      for (const [tensorId, offset] of Object.entries(tensorOffsets ?? {})) {
        if (!program.resolved.tensors[tensorId] ||
            !Number.isFinite(offset.dx) || !Number.isFinite(offset.dy) ||
            Math.abs(offset.dx) > MAX_TENSOR_OFFSET || Math.abs(offset.dy) > MAX_TENSOR_OFFSET)
          throw new Error(`invalid layout offset for tensor "${tensorId}"`);
        if (Math.abs(offset.dx) >= 1e-6 || Math.abs(offset.dy) >= 1e-6)
          checkedOffsets[tensorId] = { dx: offset.dx, dy: offset.dy };
      }
      const clampedTile = Math.max(
        TILE_SCALE_MIN,
        Math.min(TILE_SCALE_MAX, Math.round(tileScale))
      );
      set({
        ...base,
        dslText: dsl,
        draftText: dsl,
        exampleIndex: -1,
        focusNode: null,
        direction,
        showEntangled: showEntangled ?? false,
        tileScale: clampedTile,
        snapToGrid,
        axisMode,
        tensorOffsets: checkedOffsets,
        selection,
        ...recompute(program.resolved, selection),
      });
      return true;
    } catch {
      return false;
    }
  },

  setSelection: (tensorId, region, compose) => {
    const {
      selection,
      resolved,
      workspaceHistory,
      tensorOffsets,
      pinnedBox,
      hiddenBoxes,
      perBox,
      entangled,
    } = get();
    const mode = compose ?? "union";
    const drawn = region.boxes;
    // Drawing on a tensor is the other half of the link the operations list
    // makes: clicking a row puts a tile on its output, and putting a tile
    // anywhere lights the row that produced what it sits on.
    const drawnOp = operationForTensor(resolved, tensorId);

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
      // Rebuild in place and retain the object identity of parts on other
      // tensors. Their indices can still shift when this tensor loses parts,
      // so the index-based UI state below is remapped by identity.
      parts = [];
      let k = 0;
      for (const p of selection.parts) {
        if (p.tensorId !== tensorId) parts.push(p);
        else if (k < mine.length) parts.push({ tensorId, box: mine[k++] });
      }
      for (; k < mine.length; k++) parts.push({ tensorId, box: mine[k] });
    }
    const sel = parts.length === 0 ? null : { parts };
    const newIndex = new Map(parts.map((part, index) => [part, index]));
    const remap = (index: number): number | null => {
      if (mode === "replace" || !selection) return null;
      return newIndex.get(selection.parts[index]) ?? null;
    };
    const nextPinned = pinnedBox === null ? null : remap(pinnedBox);
    const nextHidden = new Set<number>();
    for (const index of hiddenBoxes) {
      const mapped = remap(index);
      if (mapped !== null) nextHidden.add(mapped);
    }
    set({
      selection: sel,
      selectedOp: sel ? drawnOp : null,
      // Null is a real workspace state: the first selection must be undoable
      // without also rewinding an earlier tensor move.
      workspaceHistory: appendWorkspaceHistory(workspaceHistory, { selection, tensorOffsets }),
      focusedBox: nextPinned,
      pinnedBox: nextPinned,
      hiddenBoxes: nextHidden,
      preview: null,
      ...recompute(resolved, sel, { selection, perBox, entangled }),
    });
  },

  clearSelection: () => {
    const { selection, workspaceHistory, tensorOffsets } = get();
    set({
      selection: null,
      selectedOp: null,
      workspaceHistory: selection
        ? appendWorkspaceHistory(workspaceHistory, { selection, tensorOffsets })
        : workspaceHistory,
      backwardRes: null,
      byTensorRes: null,
      forwardRes: null,
      perBox: null,
      entangled: null,
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: new Set<number>(),
      preview: null,
    });
  },

  undoWorkspace: () => {
    const { workspaceHistory, resolved, selection, perBox, entangled } = get();
    if (!workspaceHistory.length) return;
    const prev = workspaceHistory[workspaceHistory.length - 1];
    if (prev.source) {
      // Undoing a composite expansion: the graph itself goes back, so the
      // selection has to be restored against *that* graph rather than the
      // expanded one it was recorded beside.
      try {
        const program = compileDSL(prev.source.dslText);
        set({
          ...loadResolvedGraph(program.graph, program.resolved),
          dslText: prev.source.dslText,
          draftText: prev.source.draftText,
          exampleIndex: prev.source.exampleIndex,
          tensorOffsets: prev.tensorOffsets,
          selection: prev.selection,
          workspaceHistory: workspaceHistory.slice(0, -1),
          focusNode: null,
          ...recompute(program.resolved, prev.selection),
        });
      } catch (e) {
        set({ loadError: (e as Error).message });
      }
      return;
    }
    set({
      selection: prev.selection,
      tensorOffsets: prev.tensorOffsets,
      workspaceHistory: workspaceHistory.slice(0, -1),
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: new Set<number>(),
      preview: null,
      ...recompute(resolved, prev.selection, { selection, perBox, entangled }),
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
    if (index !== null && get().hiddenBoxes.has(index)) return;
    set({ focusedBox: index });
  },

  togglePinBox: (index) => {
    if (get().hiddenBoxes.has(index)) return;
    const pinned = get().pinnedBox === index ? null : index;
    set({ pinnedBox: pinned, focusedBox: pinned });
  },

  clearFocus: () => set({ pinnedBox: null, focusedBox: null }),

  toggleBoxHidden: (index) => {
    const next = new Set(get().hiddenBoxes);
    const hiding = !next.delete(index);
    if (hiding) next.add(index);
    const focused = get().focusedBox === index;
    set({
      hiddenBoxes: next,
      ...(hiding && focused ? { focusedBox: null, pinnedBox: null } : {}),
    });
  },

  setDragging: (v) => set({ dragging: v }),
  toggleEntangled: () => set({ showEntangled: !get().showEntangled }),

  // Direction is a view setting. It changes what is drawn and which section the
  // inspector shows, never what was analysed, so no repropagation follows.
  setDirection: (d) => set({ direction: d }),

  toggleDirection: (axis) => {
    const { direction } = get();
    const backward = direction === "backward" || direction === "both";
    const forward = direction === "forward" || direction === "both";
    const nextBackward = axis === "backward" ? !backward : backward;
    const nextForward = axis === "forward" ? !forward : forward;
    const next: Direction = nextBackward
      ? nextForward ? "both" : "backward"
      : nextForward ? "forward" : "none";
    set({ direction: next });
  },

  setTheme: (theme) => set({ theme }),

  setViewCfg: (tensorId, cfg) => {
    const state = get();
    const shape = state.resolved?.tensors[tensorId]?.resolved;
    const next = { ...state.viewCfgs[tensorId], ...cfg };
    if (!shape || !viewCfgFits(shape, next)) return;
    set({
      viewCfgs: { ...state.viewCfgs, [tensorId]: { ...next, sliders: next.sliders.slice() } },
      preview: null,
    });
  },

  setSnapToGrid: (v) => set({ snapToGrid: v }),
  setAxisMode: (v) => set({ axisMode: v }),

  setTileScale: (v) =>
    set({ tileScale: Math.max(TILE_SCALE_MIN, Math.min(TILE_SCALE_MAX, Math.round(v))) }),

  setFocusNode: (node) => set({ focusNode: node }),
  setSelectedOp: (nodeId) => set({ selectedOp: nodeId }),

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
      workspaceHistory: appendWorkspaceHistory(workspaceHistory, {
        selection,
        tensorOffsets: previousOffsets,
      }),
    });
  },

  resetTensorLayout: () => {
    const { selection, tensorOffsets, workspaceHistory } = get();
    if (!Object.keys(tensorOffsets).length) return;
    set({
      tensorOffsets: {},
      workspaceHistory: appendWorkspaceHistory(workspaceHistory, { selection, tensorOffsets }),
    });
  },

  setPreviewBox: (tensorId, box, expectedView) => {
    // A queued pointer probe may belong to the slice before a keyboard scrub.
    if (tensorId && expectedView && get().viewCfgs[tensorId] !== expectedView) return;
    const { resolved } = get();
    if (!tensorId || !box || !resolved || resolved.nodes.length > MAX_PREVIEW_NODES) {
      if (get().preview) set({ preview: null });
      return;
    }
    try {
      // `executeQuery` validates and defensively copies before propagating, so
      // the caller's box is never aliased into stored state.
      const region: Region = { boxes: [box], exact: true, reasons: [] };
      set({
        preview: executeQuery(resolved, { tensorId, region, direction: "both" }),
      });
    } catch {
      set({ preview: null });
    }
  },

  expandNodeInPlace: (nodeId) => {
    const state = get();
    const { graph } = state;
    if (!graph) return;
    try {
      const g2 = expandNode(graph, nodeId);
      // Source and graph remain one transaction: rerunning or sharing the text
      // must restore the same primitive graph currently shown in the workspace.
      const source = toDSL(g2);
      const program = compileDSL(source);
      // `loadResolvedGraph` clears the history, and rightly: its entries name
      // tensors and coordinates in the graph being replaced. The one entry that
      // survives is the one it cannot invalidate, because it is what to go back
      // *to* — recorded after the clear, for that reason.
      const restore: WorkspaceSnapshot = {
        selection: state.selection,
        tensorOffsets: state.tensorOffsets,
        source: {
          dslText: state.dslText,
          draftText: state.draftText,
          graph,
          exampleIndex: state.exampleIndex,
        },
      };
      set({
        ...loadResolvedGraph(program.graph, program.resolved),
        dslText: source,
        draftText: source,
        exampleIndex: -1,
        focusNode: null,
        workspaceHistory: [restore],
      });
    } catch (e) {
      set({ loadError: (e as Error).message });
    }
  },
}));
