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
import { parseImportJSON } from "../import/json";
import { preflightImport } from "../import/preflight";
import {
  ImportError,
  type ImportDiagnostic,
  type ImportFormat,
  type ImportReport,
  type ImportResult,
} from "../import/types";
import type { CompilerDiagnostic } from "../parse/compiler";
import { compileDSL, tryCompileDSL } from "../parse/compiler";
import { toDSL } from "../parse/dsl";
import { GraphError } from "../core/shapes";
import { resolveGraph } from "../core/graph";
import { graphScale, MAX_ELEM_PX, planeExtents, TILE_SCALE_MAX, TILE_SCALE_MIN } from "./tiling";
import { tileOf } from "./grid";
import type { AxisMode } from "./shape-label";
import type { TensorOffset, TensorOffsets } from "./tensor-layout";
import { defaultViewCfg, viewAxes, viewCfgFits, type ViewCfg } from "./tensor-view";

/**
 * Where the installed graph came from.
 *
 * `applyDSL` used to be the only way anything reached `graph`/`resolved`, which
 * made "the source" and "the DSL text" the same string by accident rather than
 * by design. They are not the same thing. An imported model has no DSL text and
 * must not be given one: `toDSL` uses `Tensor.name` as its identifier and
 * regenerates node ids, so a round trip through text renames the very nodes an
 * import report addresses. Generated DSL stays available as an explicit, lossy
 * conversion; it is not how an import is held.
 *
 * Both arms land on the same `Graph` → `resolveGraph` → `SymbolicExecutor`
 * pipeline. Nothing below this type knows which one it is looking at.
 */
export type WorkspaceSource =
  | { kind: "dsl"; text: string; exampleIndex: number }
  | {
      kind: "import";
      format: ImportFormat;
      fileName: string;
      report: ImportReport;
    };

/** The installed DSL text, or null when the workspace was imported. */
export const dslTextOf = (source: WorkspaceSource): string | null =>
  source.kind === "dsl" ? source.text : null;

/** Which built-in example is installed; -1 for edited or imported workspaces. */
export const exampleIndexOf = (source: WorkspaceSource): number =>
  source.kind === "dsl" ? source.exampleIndex : -1;

/**
 * A source that is exactly a built-in example *is* that example, however it got
 * here - picked from the menu, restored from a link, or typed back by hand.
 * Deriving the index from the text keeps the picker honest after an edit is
 * undone, which a remembered index could not.
 */
export const dslSource = (text: string): Extract<WorkspaceSource, { kind: "dsl" }> => ({
  kind: "dsl",
  text,
  exampleIndex: EXAMPLES.findIndex((ex) => ex.dsl === text),
});

/** Which independently toggled views are active in the workspace. `none` is
 * the explicit figures-only state: analysis remains live while paint and rows hide. */
export type Direction = "none" | "backward" | "forward" | "both";
export type ConeDirection = "backward" | "forward";
export type PanelSide = "left" | "right";
export type Theme = "light" | "dark";
/** The two classes of question the inspector answers; see `inspectorTab`. */
export type InspectorTab = "dependencies" | "execution";
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
 * Which tensors a highlight should reach: the drawn tiles' own, plus the cones
 * the direction filter is actually showing.
 *
 * Both cones are computed whatever the filter says, so this has to be derived
 * from what was *asked for* rather than from what was computed - otherwise
 * hiding a cone leaves its tensors lit. The graph canvas and the operations
 * list are two views of one answer, and they were drifting: the canvas followed
 * the filter while the list stayed on the union of both directions and never
 * changed. With both cones off, what stays lit is what the reader drew.
 */
export function involvedTensorIds(
  selection: Selection,
  backwardRes: PropResult | null,
  forwardRes: PropResult | null,
  perBox: BoxProp[] | null,
  hiddenBoxes: Set<number>,
  direction: Direction
): Set<string> {
  const involved = new Set<string>(selectedTensorIds(selection));
  const shown = [
    direction === "backward" || direction === "both"
      ? enabledPropResult(backwardRes, perBox, hiddenBoxes, null, "backward")
      : null,
    direction === "forward" || direction === "both"
      ? enabledPropResult(forwardRes, perBox, hiddenBoxes, null, "forward")
      : null,
  ];
  for (const res of shown) if (res) for (const id of res.tensors.keys()) involved.add(id);
  return involved;
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

/** Shared target for inspector analysis, movement, and hidden-axis controls. */
export function analysisTarget(parts: SelPart[], group: string | null, focus: number | null) {
  const tensorId = group && parts.some((part) => part.tensorId === group)
    ? group : anchorTensorId({ parts }, null);
  const focusedBox = focus !== null && parts[focus]?.tensorId === tensorId ? focus : null;
  const index = focusedBox ?? parts.reduce((last, part, i) => part.tensorId === tensorId ? i : last, -1);
  return { tensorId, focusedBox, index };
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
  source?: { source: WorkspaceSource; draftText: string; graph: Graph };
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
  /** Where the graph currently installed in `graph`/`resolved` came from. */
  source: WorkspaceSource;
  /** Editable DSL. It may differ from the installed source while the built
   * workspace remains live; a successful `applyDSL` advances both together.
   * An imported workspace leaves it empty: there is no text behind that graph,
   * and showing the previous model's source beside it would be a lie about
   * what is installed. */
  draftText: string;
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
  /**
   * Why the last import failed, addressed by node rather than by span.
   *
   * A DSL diagnostic underlines the text it is about, through the source map.
   * An imported graph has no source map and no text, so a failure names the
   * node instead and the canvas highlights it. Kept separate from
   * `diagnostics` for that reason: the two are addressed differently, and
   * merging them would force one presentation to fake the other's anchor.
   */
  importDiagnostics: ImportDiagnostic[];

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
  /**
   * The tensor whose tiles the inspector analyses, when the user has said so.
   *
   * Scope is deliberate state rather than something read off the pointer: a
   * draw, a pin, or a group header names it, and it survives until one of those
   * three names another. Hovering deliberately cannot reach it - a preview must
   * not change what the panel is *about*, and the group header sits inside the
   * hovered list, so letting hover re-scope made reaching for the header undo
   * the click that got there. Null falls back to the anchor tensor, which is
   * where a workspace with one group stays.
   */
  analysisGroup: string | null;
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

  /**
   * Which class of question the inspector is answering.
   *
   * `dependencies` is every figure that is a function of the graph and the
   * drawn region: exact, or a bound with its reason named. `execution` is the
   * figures that only exist once an execution is assumed - an order, a tiling
   * of the whole tensor, a fusion decision - which are modelled rather than
   * bounded and would be read as facts if they shared a panel with them.
   */
  inspectorTab: InspectorTab;

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
  /** Stage in DSL mode; replace immediately when no editor exists for an import. */
  chooseExample: (i: number) => void;
  setDraftText: (text: string) => void;
  applyDSL: (text: string) => void;
  /**
   * Install an already-converted model. The boundary a decoder lands on,
   * whatever decoded it. Returns whether it installed.
   */
  installImport: (result: ImportResult) => boolean;
  /** The JSON door: read an import document, then install it. */
  importJSON: (text: string, options?: { fileName?: string; format?: ImportFormat }) => boolean;
  /** Surface a failure that happened before bytes reached the JSON door. */
  reportImportError: (message: string) => void;
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
  /** Scope every readout below the tiles list to one tensor's tiles. Releases
   * the pin, which is itself a group choice and would otherwise outrank this
   * one and block hovering the group just chosen. */
  selectAnalysisGroup: (tensorId: string | null) => void;
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
  setInspectorTab: (tab: InspectorTab) => void;
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
    // A group is named by tensor, not by index, so an edit that renumbers parts
    // leaves it intact. Deleting the last tile on it is what retires it.
    analysisGroup: parts.some((part) => part.tensorId === get().analysisGroup)
      ? get().analysisGroup
      : null,
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

/**
 * Put a resolved graph in the workspace, whatever produced it.
 *
 * This is the install boundary, and it takes its source as an argument rather
 * than assuming one. Every path that replaces the graph goes through here -
 * compiling DSL, restoring a link, undoing an expansion, importing a model -
 * so the rule that source and graph are one transaction is structural instead
 * of a convention each call site remembers to repeat. An importer writes two
 * thousand nodes and has no text to show for them; that is a difference in
 * where the graph came from, and nothing below this line needs to know it.
 *
 * `draftText` is deliberately not set here. The editable buffer does not always
 * follow the installed source - an undo restores a draft that was mid-edit when
 * the expansion happened - so the callers that do own it say so.
 */
function installGraph(
  graph: Graph,
  resolved: ResolvedGraph,
  source: WorkspaceSource
): Pick<
  State,
  | "graph" | "resolved" | "source" | "loadError" | "diagnostics" | "importDiagnostics"
  | "selection" | "backwardRes" | "forwardRes"
  | "byTensorRes"
  | "entangled"
  | "perBox" | "focusedBox" | "pinnedBox" | "viewCfgs" | "preview" | "graphPx"
  | "hiddenBoxes" | "analysisGroup" | "workspaceHistory" | "tensorOffsets" | "focusNode"
  | "selectedOp"
> {
  const viewCfgs: Record<string, ViewCfg> = {};
  for (const t of Object.values(resolved.tensors)) viewCfgs[t.id] = defaultViewCfg(t.resolved!);
  return {
    graph,
    resolved,
    source,
    focusNode: null,
    selectedOp: null,
    loadError: null,
    diagnostics: [],
    importDiagnostics: [],
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
    analysisGroup: null,
    preview: null,
    viewCfgs,
    graphPx: graphScale(planesOf(resolved)),
  };
}

export const useStore = create<State>((set, get) => ({
  source: { kind: "dsl", text: EXAMPLES[0].dsl, exampleIndex: 0 },
  draftText: EXAMPLES[0].dsl,
  graph: null,
  resolved: null,
  loadError: null,
  diagnostics: [],
  importDiagnostics: [],
  showEntangled: false,
  inspectorTab: "dependencies",
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
  analysisGroup: null,
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
      importDiagnostics: [],
    });
  },

  chooseExample: (i) => {
    if (get().source.kind === "import") get().loadExample(i);
    else get().stageExample(i);
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
        importDiagnostics: [],
        loadError: `line ${result.diagnostics[0].span.start.line}: ${result.diagnostics[0].message}`,
      });
      return;
    }
    try {
      const program = result.program;
      const source = dslSource(text);
      const base = installGraph(program.graph, program.resolved, source);
      const example = source.exampleIndex >= 0 ? EXAMPLES[source.exampleIndex] : null;
      const st: Partial<State> = { ...base, draftText: text };
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
      set({
        draftText: text,
        loadError: (e as Error).message,
        diagnostics: [],
        importDiagnostics: [],
      });
    }
  },

  installImport: (result) => {
    // Preflight first, and not because resolution would miss these: it would
    // catch most of them, one at a time, in graph terms. This reports every
    // problem at once so a converter author is not on a fix-one-reload loop
    // over four hundred nodes, and it separates the dimensions the file simply
    // left free - a dynamic batch axis is ordinary, not a defect - from the
    // defects no binding can fix.
    const preflight = preflightImport(result);
    if (preflight.errors.length || preflight.unbound.length) {
      const asking = preflight.unbound.length
        ? [
            {
              severity: "error" as const,
              message:
                `${preflight.unbound.length === 1 ? "dimension" : "dimensions"}` +
                ` ${preflight.unbound.join(", ")} ${preflight.unbound.length === 1 ? "is" : "are"}` +
                ` left free by the model and need a value before any region can be measured`,
            },
          ]
        : [];
      const diagnostics = [...asking, ...preflight.errors];
      set({
        loadError: diagnostics[0].message,
        diagnostics: [],
        importDiagnostics: diagnostics,
      });
      return false;
    }

    // `resolveGraph`, never `resolveGraphCollecting`. The collecting form
    // prunes a failing node and everything downstream of it, which is right for
    // a line of hand-written DSL - one error, no invented cascade - and exactly
    // wrong here: a four-hundred-node model would install minus whatever hung
    // off the bad node, and every cone drawn on it would answer a question
    // about a shorter model than the one that was opened. That is a subset of
    // the truth, the one failure this engine treats as critical. An unresolved
    // import fails, named.
    let resolved;
    try {
      resolved = resolveGraph(result.graph);
    } catch (e) {
      const error = e as Error;
      const subject = e instanceof GraphError ? e.subject : undefined;
      set({
        loadError: error.message,
        diagnostics: [],
        importDiagnostics: [
          { severity: "error", message: error.message, ...(subject ? { subject } : {}) },
        ],
      });
      return false;
    }
    const { origin } = result.report;
    set({
      ...installGraph(result.graph, resolved, {
        kind: "import",
        format: origin.format,
        fileName: origin.fileName,
        report: result.report,
      }),
      // No text stands behind an imported graph. Leaving the previous model's
      // source in the editor would offer a Run that silently replaces what was
      // just imported with something else entirely.
      draftText: "",
    });
    return true;
  },

  importJSON: (text, options) => {
    let result: ImportResult;
    try {
      result = parseImportJSON(text, options ?? {});
    } catch (e) {
      const diagnostics =
        e instanceof ImportError
          ? e.diagnostics
          : [{ severity: "error" as const, message: (e as Error).message }];
      set({
        loadError: diagnostics[0].message,
        diagnostics: [],
        importDiagnostics: diagnostics,
      });
      return false;
    }
    return get().installImport(result);
  },

  reportImportError: (message) =>
    set({
      loadError: message,
      diagnostics: [],
      importDiagnostics: [{ severity: "error", message }],
    }),

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
      const base = installGraph(program.graph, program.resolved, dslSource(dsl));
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
        draftText: dsl,
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
      // Drawing releases the pin, and the analysis follows the pointer to this
      // tensor. Remapping it was never able to keep a pin on the tensor being
      // drawn on - those parts are rebuilt, so their identity is gone - and
      // kept one on any *other* tensor, which is exactly backwards: it left the
      // panel describing the tile the reader had just left while the tile they
      // drew sat dimmed in another group.
      focusedBox: null,
      pinnedBox: null,
      hiddenBoxes: nextHidden,
      // A subtract gesture can empty the tensor it was aimed at; the store then
      // holds no group rather than one nothing is drawn on.
      analysisGroup: parts.some((part) => part.tensorId === tensorId) ? tensorId : null,
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
      analysisGroup: null,
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
      //
      // Resolving the stored graph rather than recompiling the stored text:
      // the snapshot already holds the graph, and going back through the
      // compiler made undo depend on there being source text to compile, which
      // an imported workspace does not have.
      try {
        const resolved = resolveGraph(prev.source.graph);
        set({
          ...installGraph(prev.source.graph, resolved, prev.source.source),
          draftText: prev.source.draftText,
          tensorOffsets: prev.tensorOffsets,
          selection: prev.selection,
          workspaceHistory: workspaceHistory.slice(0, -1),
          ...recompute(resolved, prev.selection),
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
      // The restored selection may not contain the group at all, and undo is
      // not the place to guess which of its tensors the reader meant.
      analysisGroup: null,
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
    const state = get();
    const target = analysisTarget(state.selection?.parts ?? [], state.analysisGroup,
      state.perBox ? state.focusedBox : null);
    const focused = target.focusedBox;
    editSelection(
      get,
      set,
      (parts, shapeOf) => {
        const anchor = target.tensorId;
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
    const { hiddenBoxes, pinnedBox, selection, analysisGroup } = get();
    if (hiddenBoxes.has(index)) return;
    const pinned = pinnedBox === index ? null : index;
    // Pinning a tile is a deliberate click on that tile, so it names the group
    // as well as the tile - otherwise clicking a row in another group would
    // emphasise a tile the readout below was not about. Unpinning leaves the
    // group where the pin put it, which is what Escape should return to.
    const tensorId = pinned === null ? analysisGroup : selection?.parts[pinned]?.tensorId ?? null;
    set({ pinnedBox: pinned, focusedBox: pinned, analysisGroup: tensorId });
  },

  clearFocus: () => set({ pinnedBox: null, focusedBox: null }),

  selectAnalysisGroup: (tensorId) =>
    set({ analysisGroup: tensorId, pinnedBox: null, focusedBox: null }),

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
  setInspectorTab: (tab) => set({ inspectorTab: tab }),

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
    // Expansion rewrites the workspace as generated DSL, and an imported model
    // must not take that path. `toDSL` identifies tensors by name and
    // regenerates node ids, so the round trip renames exactly the nodes the
    // import report addresses - the report would still be displayed, now
    // pointing at nodes that no longer exist. Refusing is the explicit
    // limitation; converting an import to DSL is a separate, declared-lossy
    // action rather than a side effect of clicking a glyph.
    if (state.source.kind === "import") {
      set({
        loadError:
          "expanding a composite rewrites the workspace as generated DSL," +
          " which would rename the nodes the import report names",
      });
      return;
    }
    try {
      const g2 = expandNode(graph, nodeId);
      // Source and graph remain one transaction: rerunning or sharing the text
      // must restore the same primitive graph currently shown in the workspace.
      const text = toDSL(g2);
      const program = compileDSL(text);
      // `installGraph` clears the history, and rightly: its entries name
      // tensors and coordinates in the graph being replaced. The one entry that
      // survives is the one it cannot invalidate, because it is what to go back
      // *to* — recorded after the clear, for that reason.
      const restore: WorkspaceSnapshot = {
        selection: state.selection,
        tensorOffsets: state.tensorOffsets,
        source: { source: state.source, draftText: state.draftText, graph },
      };
      set({
        ...installGraph(program.graph, program.resolved, dslSource(text)),
        draftText: text,
        workspaceHistory: [restore],
      });
    } catch (e) {
      set({ loadError: (e as Error).message });
    }
  },
}));

/** Four components derive the same boolean from the theme to mix tile hues. */
export const useDark = (): boolean => useStore((s) => s.theme === "dark");
