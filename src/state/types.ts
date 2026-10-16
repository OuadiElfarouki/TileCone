import { StoreApi } from "zustand";
import { Entanglement } from "../core/entangle";
import { Graph, ResolvedGraph } from "../core/graph";
import { PlanReport, TaskSupply, Work } from "../core/plan/interfaces";
import { TaskRef, TilePlan } from "../core/plan/plan";
import { PropResult } from "../core/propagate";
import { Box, Region } from "../core/region";
import { CompilerDiagnostic } from "../parse/compiler";
import { BaseGraphLayout } from "../view/graph/graph-scene";
import { NodeOffset, NodeOffsets } from "../view/graph/node-layout";
import { AxisMode } from "../view/tensor/shape-label";
import { ViewCfg } from "../view/tensor/tensor-view";
import {
  BoxProp,
  CanvasTool,
  ConeDirection,
  Direction,
  ReplayPlayback,
  InspectorTab,
  PanelSide,
  type Selection,
  SelPart,
  Theme,
} from "../view/workspace";
import { WorkspaceSnapshot } from "./history";

/**
 * A plan kept for comparison: its tiling, the whole-plan work it was evaluated
 * to, and the graph outputs it does not write, whose work that total leaves out.
 */
export type KeptPlan = {
  id: number;
  tiles: Record<string, number[]>;
  total: Work;
  unwritten: readonly string[];
};

export type WorkspaceRestore = {
  dsl: string;
  /** An unrun (possibly invalid) draft restored beside the last built source. */
  draftText?: string;
  hiddenBoxes?: number[];
  analysisGroup?: string | null;
  pinnedBox?: number | null;
  direction: Direction;
  /** Optional for callers restoring links written before entanglement existed. */
  showEntangled?: boolean;
  tileScale: number;
  snapToGrid: boolean;
  /** Whether a compact shape reads as semantic labels or numeric extents. */
  axisMode: AxisMode;
  nodeOffsets?: NodeOffsets;
  viewCfgs?: Record<string, ViewCfg>;
  parts: SelPart[] | null;
  /** Tile extents per planned tensor; absent restores nothing tiled. */
  planTiles?: Record<string, number[]>;
  /** The inspected task, which must be a tile of `planTiles`. */
  planTask?: TaskRef | null;
  /**
   * Tilings kept for comparison, oldest first. Restoring evaluates each one
   * again, so the caller supplies tilings and never figures.
   */
  keptTilings?: Record<string, number[]>[];
  /** Open on the Plan view rather than whichever view is current. */
  planView?: boolean;
};

/** Direct canvas gestures add by default; Alt subtracts. "replace" is internal
 * for examples, restored URL state, and operation-list probes. */
export type Compose = "union" | "subtract" | "replace";

export type State = {
  /** Source currently installed in `graph`/`resolved` and encoded by Share. */
  dslText: string;
  /** Editable source. It may differ from `dslText` while the built workspace
   * remains live; a successful `applyDSL` advances both together. */
  draftText: string;
  exampleIndex: number;
  graph: Graph | null;
  resolved: ResolvedGraph | null;
  /** Structural layout computed beside compilation in the analysis Worker. */
  baseLayout: BaseGraphLayout | null;
  /** Worker-side identity of `resolved`, used by later plan analysis requests. */
  workerGraphId: number | null;
  compiling: boolean;
  autosaveStatus: "saved" | "saving" | "unavailable" | null;
  workspaceNotice: string | null;
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
   * The part the keyboard edits: the tile just drawn, or the one last clicked
   * in the tiles list, until a press anywhere else releases it. Separate from
   * the pin, which narrows the analysis to one tile; selecting a tile to move
   * it must not hide the comparison the reader drew it for. With none, the
   * arrows, Delete and `h` act on nothing (`keyboardTile`).
   */
  editBox: number | null;
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
   * drawn region: exact, or a bound with its reason named. `plan` is every
   * figure that rests on a declared tiling: exact for that tiling, or bounded.
   * An execution the graph does not fix - which tensors are written, which are
   * computed where they are needed - is a choice the plan states, never one
   * the inspector assumes.
   */
  inspectorTab: InspectorTab;
  /** A replay of the inspected task's family on the cards. It never changes
   *  `selection` or `plan`; those remain underneath it. */
  replayPlayback: ReplayPlayback | null;
  /**
   * How many levels of the plan's tasks a replay of each family follows
   * upstream, by the family's tensor. Absent is 1: the family's own tasks.
   * View state, cleared with the graph its tensor ids belong to.
   */
  replayDepth: Record<string, number>;

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
   * lights. A click on empty canvas, a press elsewhere outside the graph, or
   * Escape clears it. Panning, zooming and fit keep it: moving the view is not
   * choosing anything else.
   */
  selectedOp: string | null;
  /** Width in px of each side panel when open, and whether it is collapsed to a
   * rail. Collapsing keeps the remembered width so reopening restores it. */
  panelW: { left: number; right: number };
  panelCollapsed: { left: boolean; right: boolean };
  /** User displacement from dagre's collision-free base placement, keyed by
   * scene node key so operation nodes move on the same terms as cards. */
  nodeOffsets: NodeOffsets;
  /**
   * Set when a run changed only sizes: the canvas should keep every node where
   * it was on screen, measured against the new layout, rather than starting
   * from the generated one. The canvas consumes it with `carryPlacement`.
   */
  placementCarry: boolean;
  /**
   * What a press on a card or an operation does. `select` keeps the graph's
   * placement fixed while tiles are drawn and operations picked; `move` makes
   * every node furniture, dragged from anywhere on it, for laying a graph out
   * to be looked at or shown. Holding Ctrl/Cmd borrows `move` for one gesture
   * without changing this.
   */
  canvasTool: CanvasTool;
  /** Transient resize mode; the Move tool remains selected when it ends. */
  resizingTensor: string | null;
  setResizingTensor: (id: string | null) => void;
  setCardScale: (id: string, scale: number, record?: boolean) => void;
  /** Size and position together, for live gestures and their cancellation. */
  setCardGeometry: (id: string, scale: number, offset: NodeOffset, record?: boolean) => void;
  /** Make room at requested sizes, keeping this tensor anchored. Undoable. */
  makeRoomForCard: (id: string, layout: BaseGraphLayout) => void;
  commitCardResize: (id: string, beforeScale: number, beforeOffset: NodeOffset) => void;

  /**
   * Tile extents per planned tensor (see `core/plan`). Independent of the
   * canvas lattice: detail, zoom and projection never change it. Cleared when
   * the graph is replaced, like the selection.
   */
  planTiles: Record<string, number[]>;
  /** The task the Plan view describes. */
  planTask: TaskRef | null;
  /** Derived from `planTiles` against `resolved`; null when nothing is tiled. */
  plan: TilePlan | null;
  /** Derived: what `planTask` reads and which producer tasks supply it. */
  planSupply: TaskSupply | null;
  /**
   * Plans kept for comparison with the current one, oldest first, at most
   * `MAX_KEPT_PLANS`. Each holds the totals it was evaluated to; the graph
   * cannot change under them, since replacing it clears them. Not workspace
   * history, and not part of a share link.
   */
  keptPlans: KeptPlan[];

  /** Compile and install an example immediately: app boot and tests. */
  loadExample: (i: number) => void;
  /** Browser entry point: compilation, inference, and layout run in a Worker. */
  loadExampleAsync: (i: number) => Promise<boolean>;
  /** Put an example in the editor without replacing the built workspace. */
  stageExample: (i: number) => void;
  setDraftText: (text: string) => void;
  applyDSL: (text: string) => void;
  applyDSLAsync: (text: string) => Promise<boolean>;
  /** Compile, validate, and install a shared workspace as one transaction. */
  restoreWorkspace: (workspace: WorkspaceRestore) => boolean;
  restoreWorkspaceAsync: (workspace: WorkspaceRestore) => Promise<boolean>;
  setSelection: (tensorId: string, region: Region, compose?: Compose) => void;
  clearSelection: () => void;
  undoWorkspace: () => void;
  /** Move the whole selection along one axis, clamped to the tensor.
   * `record` false appends no undo entry : used for auto-repeat, so holding an
   * arrow key is one undo step rather than forty. */
  moveSelection: (axis: number, delta: number, record?: boolean) => void;
  /** Nudge with the current snap unit; stride-one axes carry in row-major order. */
  nudgeSelection: (axis: number, sign: -1 | 1, multiplier?: number, record?: boolean) => void;
  /** Replace one ordered selection part without renumbering its peers. */
  replaceBox: (index: number, box: Box) => void;
  deleteBox: (index: number) => void;
  /** Transient hover focus; ignored while a part is pinned. */
  hoverBox: (index: number | null) => void;
  /** Click a part to pin it, or the same part again to unpin. */
  togglePinBox: (index: number) => void;
  /** Drop both hover and pin. What Escape does. */
  clearFocus: () => void;
  /** Release the keyboard's tile (`editBox`) without touching focus or pin. */
  releaseEditBox: () => void;
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
  setCanvasTool: (tool: CanvasTool) => void;
  setViewCfg: (tensorId: string, cfg: Partial<ViewCfg>) => void;
  /**
   * Give a tensor a tile of its own, one extent per axis, or return it to the
   * canvas default with `null`. A gesture setting like snap and detail: it
   * changes future gestures and is not an undo step. With `refit`, the part at
   * that index is replaced by the tile of the new extents that contains its
   * lower corner, as one undoable selection edit.
   */
  setTensorTile: (tensorId: string, tile: number[] | null, refit?: number) => void;
  /**
   * Draw `axes` as the tensor card's rows and columns, or the default pair with
   * `null`. Presentation only. The hidden-axis positions move onto the tile the
   * tensor's view is about (`viewAnchorIndex`), so the tile being studied stays
   * on screen when the axes it was seen through become hidden.
   */
  setViewAxes: (tensorId: string, axes: [number, number] | null) => void;
  setTileScale: (v: number) => void;
  setSnapToGrid: (v: boolean) => void;
  setAxisMode: (v: AxisMode) => void;
  setInspectorTab: (tab: InspectorTab) => void;
  setReplayPlayback: (playback: ReplayPlayback | null) => void;
  /** Set the replay depth of one family; values below 1 are 1. */
  setReplayDepth: (tensorId: string, depth: number) => void;
  updateReplayPlayback: (patch: Partial<ReplayPlayback>) => void;
  /** Tile a produced tensor with these extents, or stop tiling it with `null`. Undoable. */
  setPlanTile: (tensorId: string, tile: number[] | null) => void;
  /** Inspect one task, or none. Undoable. */
  selectPlanTask: (task: TaskRef | null) => void;
  /**
   * Inspect the task whose tile contains `element`. A produced tensor the plan
   * does not tile yet is tiled at `defaultTile` first; a graph input has no
   * tasks and is ignored.
   */
  planTaskAt: (tensorId: string, element: number[]) => void;
  /** Divide `tensorId` into tiles of `tile` and inspect the task holding `element`. */
  setPlanTileAt: (tensorId: string, tile: number[], element: number[]) => void;
  /** Tile a produced tensor at the default extents, leaving the inspected task alone. */
  tilePlanTensor: (tensorId: string) => void;
  /** Step tasks, carrying along stride-one axes; clamp at the outer grid ends. */
  movePlanTask: (axis: number, delta: number, record?: boolean) => void;
  /** Keep the current tiling and its evaluation for comparison; one without a total is not kept. */
  keepPlan: (report: PlanReport) => void;
  /** Stop keeping a plan. */
  dropKeptPlan: (id: number) => void;
  /** Make a kept plan's tiling the current one. Undoable. */
  restoreKeptPlan: (id: number) => void;
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
  setNodeOffset: (key: string, offset: NodeOffset) => void;
  /** Record one completed drag, restoring `before` when workspace undo runs. */
  commitNodeMove: (key: string, before: NodeOffset) => void;
  /** Restore Dagre's generated placement as one undoable workspace action. */
  resetNodeLayout: () => void;
  /** Install offsets that keep a size-only run's nodes in place; no undo step. */
  carryPlacement: (offsets: NodeOffsets) => void;
  /** Preview the cone of the box a click would commit, not of one element:
   * a projection gesture selects whole hidden axes, so a cell-sized preview
   * would understate the cone the same gesture goes on to produce. */
  setPreviewBox: (tensorId: string | null, box?: Box, expectedView?: ViewCfg) => void;
  toggleEntangled: () => void;
  expandNodeInPlace: (nodeId: string) => void;
};

export type SetState = StoreApi<State>["setState"];
export type GetState = () => State;
