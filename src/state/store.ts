import { create } from "zustand";
import { EXAMPLES } from "../examples/index";
import { NodeOffset } from "../view/graph/node-layout";
import { ViewCfg } from "../view/tensor/tensor-view";
import { MAX_ELEM_PX } from "../view/tensor/tiling";
import { CanvasTool, idRecord, Theme } from "../view/workspace";
import { layoutActions } from "./actions/layout";
import { planActions } from "./actions/plan";
import { selectionActions } from "./actions/selection";
import { viewActions } from "./actions/view";
import { workspaceActions } from "./actions/workspace";
import { NO_PLAN } from "./plan";
import { State } from "./types";

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

/** A workspace preference, like the theme: remembered, never shared in a link.
 *  The generated layout is what a link should reproduce, plus whatever the
 *  author moved, and not which tool their own canvas was in. */
function initialCanvasTool(): CanvasTool {
  if (typeof window === "undefined") return "select";
  try {
    return window.localStorage.getItem("tilecone.canvasTool") === "move" ? "move" : "select";
  } catch {
    // Storage is optional; the select default is the complete fallback.
    return "select";
  }
}

export const useStore = create<State>((set, get) => {
  return ({
  dslText: EXAMPLES[0].dsl,
  draftText: EXAMPLES[0].dsl,
  exampleIndex: 0,
  graph: null,
  resolved: null,
  baseLayout: null,
  workerGraphId: null,
  compiling: false,
    autosaveStatus: null,
    workspaceNotice: null,
  loadError: null,
  diagnostics: [],
  showEntangled: false,
  inspectorTab: "dependencies",
  replayPlayback: null,
  replayDepth: idRecord<number>(),
  ...NO_PLAN,
  keptPlans: [],
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
  viewCfgs: idRecord<ViewCfg>(),
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
  nodeOffsets: idRecord<NodeOffset>(),
  placementCarry: false,
  canvasTool: initialCanvasTool(),
  ...workspaceActions(set, get),
  ...selectionActions(set, get),
  ...viewActions(set, get),
  ...planActions(set, get),
  ...layoutActions(set, get),
  });
});

/** Four components derive the same boolean from the theme to mix tile hues. */
export const useDark = (): boolean => useStore((s) => s.theme === "dark");
