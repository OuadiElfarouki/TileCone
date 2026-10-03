import { NodeOffset } from "../../view/graph/node-layout";
import { idRecord, PANEL_COLLAPSE_AT, PANEL_MAX, PANEL_MIN } from "../../view/workspace";
import { recordWorkspace } from "../history";
import { GetState, SetState, State } from "../types";

/** Panel geometry and node placement. */
export const layoutActions = (set: SetState, get: GetState): Pick<
  State,
  "setPanelWidth"
  | "finishPanelResize"
  | "togglePanel"
  | "setNodeOffset"
  | "commitNodeMove"
  | "resetNodeLayout"
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

  setNodeOffset: (key, offset) => {
    const nodeOffsets = idRecord(get().nodeOffsets);
    if (Math.abs(offset.dx) < 1e-6 && Math.abs(offset.dy) < 1e-6) delete nodeOffsets[key];
    else nodeOffsets[key] = offset;
    set({ nodeOffsets });
  },

  commitNodeMove: (key, before) => {
    const { nodeOffsets } = get();
    const after = nodeOffsets[key] ?? { dx: 0, dy: 0 };
    if (Math.abs(after.dx - before.dx) < 1e-6 && Math.abs(after.dy - before.dy) < 1e-6) return;
    const previousOffsets = idRecord(nodeOffsets);
    if (Math.abs(before.dx) < 1e-6 && Math.abs(before.dy) < 1e-6) delete previousOffsets[key];
    else previousOffsets[key] = before;
    set({
      workspaceHistory: recordWorkspace(get(), { nodeOffsets: previousOffsets }),
    });
  },

  resetNodeLayout: () => {
    const { nodeOffsets } = get();
    if (!Object.keys(nodeOffsets).length) return;
    set({
      nodeOffsets: idRecord<NodeOffset>(),
      workspaceHistory: recordWorkspace(get()),
    });
  },
});
