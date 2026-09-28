import { StoreApi } from "zustand";
import { Entanglement } from "../core/entangle";
import { Graph, ResolvedGraph } from "../core/graph";
import { PlanReport, Supply, Work } from "../core/plan/interfaces";
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
  ConeDirection,
  Direction,
  ExecutionPlayback,
  ExecutionScope,
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
  /** A visual replay of the deterministic probes behind the reuse estimate.
   * It never changes `selection` or `plan`; those remain underneath it. */
  executionPlayback: ExecutionPlayback | null;
  /**
   * The attribution Execution set aside while it scopes the panel to the one
   * tile its sweep is about.
   *
   * A sweep is defined by a single tile, and the figures beside it are about
   * that tile, so the view disables every other one on the way in and puts
   * them back on the way out - whatever happened in between. Parts are held by
   * identity rather than index, the same way `setSelection` remaps them, so an
   * edit that renumbers the tiles cannot restore the wrong ones; a part that no
   * longer exists simply drops out.
   */
  executionScope: ExecutionScope | null;

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
  /** User displacement from dagre's collision-free base placement, keyed by
   * scene node key so operation nodes move on the same terms as cards. */
  nodeOffsets: NodeOffsets;
  /**
   * Whether operation nodes answer a drag. Off, only cards move and an
   * operation keeps the rank dagre gave it, which is the reading most graphs
   * want: the generated row order is what makes a chain legible. On, the whole
   * scene is furniture, for laying a graph out to be looked at or shown.
   */
  moveOps: boolean;

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
  planSupply: Supply | null;
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
  setMoveOps: (v: boolean) => void;
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
   * Step one part `steps` tiles along `axis`, where a tile is the extent its
   * tensor's tile has on that axis. An off-lattice part first lands its lower
   * edge on the lattice, as an arrow nudge does. Works on hidden axes too, and
   * the view follows the part there.
   */
  stepTile: (index: number, axis: number, steps: number) => void;
  /**
   * Draw `axes` as the tensor card's rows and columns, or the default pair with
   * `null`. Presentation only. With `keep`, the hidden-axis positions move onto
   * that part, so the tile being studied stays on screen when the axes it was
   * seen through become hidden.
   */
  setViewAxes: (tensorId: string, axes: [number, number] | null, keep?: number) => void;
  setTileScale: (v: number) => void;
  setSnapToGrid: (v: boolean) => void;
  setAxisMode: (v: AxisMode) => void;
  setInspectorTab: (tab: InspectorTab) => void;
  setExecutionPlayback: (playback: ExecutionPlayback | null) => void;
  updateExecutionPlayback: (patch: Partial<ExecutionPlayback>) => void;
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
  /** Step the inspected task `delta` tiles along `axis`, stopping at the grid's edge. */
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
  /** Preview the cone of the box a click would commit, not of one element:
   * a projection gesture selects whole hidden axes, so a cell-sized preview
   * would understate the cone the same gesture goes on to produce. */
  setPreviewBox: (tensorId: string | null, box?: Box, expectedView?: ViewCfg) => void;
  toggleEntangled: () => void;
  expandNodeInPlace: (nodeId: string) => void;
};

export type SetState = StoreApi<State>["setState"];
export type GetState = () => State;
