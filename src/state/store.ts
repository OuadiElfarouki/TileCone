import { create } from "zustand";
import { EXAMPLES } from "../examples/index";
import { NodeOffset } from "../view/graph/node-layout";
import { ViewCfg } from "../view/tensor/tensor-view";
import { MAX_ELEM_PX } from "../view/tensor/tiling";
import { idRecord, MAX_PER_BOX_PROPS, Theme } from "../view/workspace";
import { layoutActions } from "./actions/layout";
import { planActions } from "./actions/plan";
import { selectionActions } from "./actions/selection";
import { viewActions } from "./actions/view";
import { workspaceActions } from "./actions/workspace";
import { recompute } from "./analysis";
import { executionScoping } from "./execution-scope";
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
 *  author moved, and not whether their own canvas was unlocked. */
function initialMoveOps(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem("tilecone.moveOps") === "on";
  } catch {
    // Storage is optional; the locked default is the complete fallback.
    return false;
  }
}

export const useStore = create<State>((commit, get) => {
  // Keep scope transactional with edits, including undo and the first draw in
  // an empty Execution view. Playback ticks must never trigger propagation.
  const set: typeof commit = (update, replace) => commit((previous) => {
    const patch = typeof update === "function" ? update(previous) : update;
    let next = { ...previous, ...patch };
    const changed = next.selection !== previous.selection ||
      next.resolved !== previous.resolved || next.inspectorTab !== previous.inspectorTab;
    if (!changed) return patch;
    if (next.inspectorTab === "execution") {
      next = { ...next, ...executionScoping(next, "execution") };
      const anchor = next.focusedBox;
      if (next.selection && next.resolved && anchor !== null &&
          next.selection.parts.length > MAX_PER_BOX_PROPS) {
        // One extra query, regardless of selection size. Empty entries preserve
        // global tile indices/colors without computing every tile's cone.
        const part = next.selection.parts[anchor];
        const one = recompute(next.resolved, { parts: [part] });
        next = {
          ...next,
          ...one,
          perBox: next.selection.parts.map((_, index) => index === anchor
            ? one.perBox![0] : { backward: null, forward: null }),
          entangled: null,
        };
      }
    } else {
      next = { ...next, ...executionScoping(next, next.inspectorTab) };
      if (previous.inspectorTab === "execution" &&
          previous.selection && previous.selection.parts.length > MAX_PER_BOX_PROPS) {
        Object.assign(next, recompute(next.resolved, next.selection));
      }
    }
    return next;
  }, replace);
  return ({
  dslText: EXAMPLES[0].dsl,
  draftText: EXAMPLES[0].dsl,
  exampleIndex: 0,
  graph: null,
  resolved: null,
  baseLayout: null,
  workerGraphId: null,
  compiling: false,
  loadError: null,
  diagnostics: [],
  showEntangled: false,
  inspectorTab: "dependencies",
  executionPlayback: null,
  executionScope: null,
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
  moveOps: initialMoveOps(),
  ...workspaceActions(set, get),
  ...selectionActions(set, get),
  ...viewActions(set, get),
  ...planActions(set, get),
  ...layoutActions(set, get),
  });
});

/** Four components derive the same boolean from the theme to mix tile hues. */
export const useDark = (): boolean => useStore((s) => s.theme === "dark");
