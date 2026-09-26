import { TensorOffset } from "../../view/graph/tensor-layout";
import { idRecord, PANEL_COLLAPSE_AT, PANEL_MAX, PANEL_MIN } from "../../view/workspace";
import { appendWorkspaceHistory, planEditOf } from "../history";
import { GetState, SetState, State } from "../types";

/** Panel geometry and tensor-card placement. */
export const layoutActions = (set: SetState, get: GetState): Pick<
  State,
  "setPanelWidth"
  | "finishPanelResize"
  | "togglePanel"
  | "setTensorOffset"
  | "commitTensorMove"
  | "resetTensorLayout"
> => ({
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
    const tensorOffsets = idRecord(get().tensorOffsets);
    if (Math.abs(offset.dx) < 1e-6 && Math.abs(offset.dy) < 1e-6) delete tensorOffsets[tensorId];
    else tensorOffsets[tensorId] = offset;
    set({ tensorOffsets });
  },

  commitTensorMove: (tensorId, before) => {
    const { selection, tensorOffsets, workspaceHistory } = get();
    const after = tensorOffsets[tensorId] ?? { dx: 0, dy: 0 };
    if (Math.abs(after.dx - before.dx) < 1e-6 && Math.abs(after.dy - before.dy) < 1e-6) return;
    const previousOffsets = idRecord(tensorOffsets);
    if (Math.abs(before.dx) < 1e-6 && Math.abs(before.dy) < 1e-6) delete previousOffsets[tensorId];
    else previousOffsets[tensorId] = before;
    set({
      workspaceHistory: appendWorkspaceHistory(workspaceHistory, {
        selection,
        tensorOffsets: previousOffsets,
        plan: planEditOf(get()),
      }),
    });
  },

  resetTensorLayout: () => {
    const { selection, tensorOffsets, workspaceHistory } = get();
    if (!Object.keys(tensorOffsets).length) return;
    set({
      tensorOffsets: idRecord<TensorOffset>(),
      workspaceHistory: appendWorkspaceHistory(workspaceHistory, { selection, tensorOffsets, plan: planEditOf(get()) }),
    });
  },
});
