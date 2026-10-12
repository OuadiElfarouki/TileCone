import { remapped, ViewCfg, viewCfgFits } from "../../view/tensor/tensor-view";
import { TILE_SCALE_MAX, TILE_SCALE_MIN } from "../../view/tensor/tiling";
import { Direction, idRecord, viewAnchorIndex } from "../../view/workspace";
import { revealHidden } from "../selection-edit";
import { GetState, SetState, State } from "../types";

/** View settings: directions, theme, per-tensor views, detail, snapping, the
 *  canvas tool, the inspector view, playback. */
export const viewActions = (set: SetState, get: GetState): Pick<
  State,
  "setDragging"
  | "toggleEntangled"
  | "setDirection"
  | "toggleDirection"
  | "setTheme"
  | "setCanvasTool"
  | "setViewCfg"
  | "setViewAxes"
  | "setSnapToGrid"
  | "setAxisMode"
  | "setInspectorTab"
  | "setReplayPlayback"
  | "setReplayDepth"
  | "updateReplayPlayback"
  | "setTileScale"
  | "setFocusNode"
  | "setSelectedOp"
> => ({
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

  setCanvasTool: (tool) => set({ canvasTool: tool }),

  setViewCfg: (tensorId, cfg) => {
    const state = get();
    const shape = state.resolved?.tensors[tensorId]?.resolved;
    const next = { ...state.viewCfgs[tensorId], ...cfg };
    if (!shape || !viewCfgFits(shape, next)) return;
    const stored: ViewCfg = { ...next, sliders: next.sliders.slice() };
    if (next.tile) stored.tile = next.tile.slice();
    else delete stored.tile;
    if (next.axes && remapped(shape, next)) stored.axes = [next.axes[0], next.axes[1]];
    else delete stored.axes;
    set({
      viewCfgs: idRecord({ ...state.viewCfgs, [tensorId]: stored }),
      preview: null,
    });
  },

  setViewAxes: (tensorId, axes) => {
    const state = get();
    const shape = state.resolved?.tensors[tensorId]?.resolved;
    if (!shape) return;
    const cfg = state.viewCfgs[tensorId];
    // The default pair is stored as absent, so "is this card remapped" has one
    // answer however the pair was reached.
    const chosen = axes && remapped(shape, { axes }) ? axes : undefined;
    const next: ViewCfg = { ...cfg, sliders: cfg.sliders.slice() };
    if (chosen) next.axes = [chosen[0], chosen[1]];
    else delete next.axes;
    if (!viewCfgFits(shape, next)) return;
    set({ viewCfgs: idRecord({ ...state.viewCfgs, [tensorId]: next }), preview: null });
    const parts = state.selection?.parts ?? [];
    const anchor = viewAnchorIndex(parts, tensorId, state.focusedBox);
    if (anchor !== null) revealHidden(get, tensorId, parts[anchor].box);
  },

  setSnapToGrid: (v) => set({ snapToGrid: v }),

  setAxisMode: (v) => set({ axisMode: v }),

  setInspectorTab: (tab) => set({ inspectorTab: tab }),

  setReplayPlayback: (replayPlayback) => set({ replayPlayback }),

  setReplayDepth: (tensorId, depth) =>
    set((state) => ({
      replayDepth: idRecord({ ...state.replayDepth, [tensorId]: Math.max(1, Math.floor(depth)) }),
    })),

  updateReplayPlayback: (patch) => set((state) => ({
    replayPlayback: state.replayPlayback
      ? { ...state.replayPlayback, ...patch }
      : null,
  })),

  setTileScale: (v) =>
    set({ tileScale: Math.max(TILE_SCALE_MIN, Math.min(TILE_SCALE_MAX, Math.round(v))) }),

  setFocusNode: (node) => set({ focusNode: node }),

  setSelectedOp: (nodeId) => set({ selectedOp: nodeId }),
});
