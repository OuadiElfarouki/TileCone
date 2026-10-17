import { nodeKey } from "../../view/graph/graph-scene";
import { NodeOffset, NodeOffsets } from "../../view/graph/node-layout";
import type { BaseGraphLayout } from "../../view/graph/graph-scene";
import { graphIsInteractive, idRecord, layoutMovable, MAX_NODE_OFFSET, PANEL_COLLAPSE_AT, PANEL_MAX, PANEL_MIN } from "../../view/workspace";
import { cardCanResize, tensorCardSize } from "../../view/tensor/card-size";
import { roomForNode } from "../../view/graph/carry-placement";
import { MAX_CARD_SCALE, MIN_CARD_SCALE } from "../../view/tensor/tensor-view";
import { cardScalesOf, recordWorkspace } from "../history";
import { GetState, SetState, State } from "../types";

/** Every tensor card's footprint under `viewCfgs`, by node key. */
function cardSizes(state: State, viewCfgs: State["viewCfgs"]): Record<string, { w: number; h: number }> {
  const tensors = state.resolved ? Object.values(state.resolved.tensors) : [];
  return Object.fromEntries(tensors.map((tensor) =>
    [nodeKey("tensor", tensor.id), tensorCardSize(tensor, state.graphPx, viewCfgs[tensor.id])]));
}

const sameOffsets = (a: Readonly<NodeOffsets>, b: Readonly<NodeOffsets>): boolean =>
  [...new Set([...Object.keys(a), ...Object.keys(b)])].every((key) =>
    (a[key]?.dx ?? 0) === (b[key]?.dx ?? 0) && (a[key]?.dy ?? 0) === (b[key]?.dy ?? 0));

const offsetsInBounds = (offsets: Readonly<NodeOffsets>): boolean =>
  Object.values(offsets).every(({ dx, dy }) =>
    Number.isFinite(dx) && Number.isFinite(dy) && Math.abs(dx) <= MAX_NODE_OFFSET && Math.abs(dy) <= MAX_NODE_OFFSET);

/** Validate a resize before producing one complete store update. */
function cardGeometry(
  state: State,
  id: string,
  scale: number,
  offset: NodeOffset,
  record: boolean,
  room?: { layout: BaseGraphLayout; from: Readonly<NodeOffsets> }
): Partial<State> | null {
  const cfg = state.viewCfgs[id], shape = state.resolved?.tensors[id]?.resolved;
  if (!cfg || !shape || !Number.isFinite(scale) || scale < MIN_CARD_SCALE || scale > MAX_CARD_SCALE ||
      !Number.isFinite(offset.dx) || !Number.isFinite(offset.dy) ||
      Math.abs(offset.dx) > MAX_NODE_OFFSET || Math.abs(offset.dy) > MAX_NODE_OFFSET) return null;
  const nextScale = Math.abs(scale - 1) < 1e-8 ? 1 : scale;
  // A reset can still clear a legacy scale on a now-fixed view.
  if (nextScale !== 1 && !cardCanResize(shape, cfg, state.graphPx)) return null;
  const key = nodeKey("tensor", id), before = state.nodeOffsets[key] ?? { dx: 0, dy: 0 };
  const dx = Math.abs(offset.dx) < 1e-6 ? 0 : offset.dx;
  const dy = Math.abs(offset.dy) < 1e-6 ? 0 : offset.dy;
  const scaleChanged = (cfg.scale ?? 1) !== nextScale, moved = before.dx !== dx || before.dy !== dy;
  if (!scaleChanged && !moved && !room) return null;
  let viewCfgs = state.viewCfgs, nodeOffsets = state.nodeOffsets;
  if (scaleChanged) {
    const { scale: _scale, ...rest } = cfg;
    viewCfgs = idRecord({ ...viewCfgs, [id]: nextScale === 1 ? rest : { ...rest, scale: nextScale } });
  }
  if (moved || room) {
    nodeOffsets = idRecord(room ? room.from : nodeOffsets);
    if (dx === 0 && dy === 0) delete nodeOffsets[key]; else nodeOffsets[key] = { dx, dy };
  }
  if (room) {
    // Placed again from where the gesture began, not from the last frame, so
    // the neighbours follow the card's size both ways rather than drifting.
    const placed = idRecord(roomForNode(room.layout.nodes, nodeOffsets, key, cardSizes(state, viewCfgs)));
    if (offsetsInBounds(placed)) nodeOffsets = placed;
    if (!scaleChanged && sameOffsets(nodeOffsets, state.nodeOffsets)) return null;
  }
  return { viewCfgs, nodeOffsets, preview: null,
    ...(record ? { workspaceHistory: recordWorkspace(state) } : {}) };
}

/** Panel geometry and node placement. */
export const layoutActions = (set: SetState, get: GetState): Pick<
  State,
  "setPanelWidth"
  | "finishPanelResize"
  | "togglePanel"
  | "setNodeOffset"
  | "commitNodeMove"
  | "resetNodeLayout"
  | "carryPlacement"
  | "setResizingTensor"
  | "setCardScale"
  | "commitCardResize"
  | "setCardGeometry"
  | "restoreCardGeometry"
  | "makeRoomForCard"
> => ({
  setResizingTensor: (id) => {
    const state = get();
    const shape = id === null ? undefined : state.resolved?.tensors[id]?.resolved;
    if (id !== null && (!graphIsInteractive(state) || !layoutMovable(state.inspectorTab) || !shape ||
      !cardCanResize(shape, state.viewCfgs[id], state.graphPx))) return;
    set({ resizingTensor: id, ...(id !== null ? { canvasTool: "move" as const, preview: null } : {}) });
  },

  setCardScale: (id, scale, record = true) => {
    const state = get();
    const next = cardGeometry(state, id, scale, state.nodeOffsets[nodeKey("tensor", id)] ?? { dx: 0, dy: 0 }, record);
    if (next) set(next);
  },

  setCardGeometry: (id, scale, offset, record = true, room) => {
    const next = cardGeometry(get(), id, scale, offset, record, room);
    if (next) set(next);
  },

  restoreCardGeometry: (id, scale, offsets) => {
    const state = get(), cfg = state.viewCfgs[id];
    if (!cfg) return;
    const { scale: _scale, ...rest } = cfg;
    set({
      viewCfgs: idRecord({ ...state.viewCfgs, [id]: scale === 1 ? rest : { ...rest, scale } }),
      nodeOffsets: idRecord(offsets),
    });
  },

  makeRoomForCard: (id, layout) => {
    const state = get(), resolved = state.resolved;
    if (!resolved?.tensors[id] || !graphIsInteractive(state) || !layoutMovable(state.inspectorTab) || state.dragging) return;
    const nodeOffsets = roomForNode(layout.nodes, state.nodeOffsets, nodeKey("tensor", id), cardSizes(state, state.viewCfgs));
    if (!offsetsInBounds(nodeOffsets) || sameOffsets(nodeOffsets, state.nodeOffsets)) return;
    set({ nodeOffsets, workspaceHistory: recordWorkspace(state), preview: null, resizingTensor: null });
  },

  commitCardResize: (id, beforeScale, beforeOffsets) => {
    const state = get();
    if ((state.viewCfgs[id]?.scale ?? 1) === beforeScale && sameOffsets(state.nodeOffsets, beforeOffsets)) return;
    const cardScales = cardScalesOf(state);
    if (beforeScale === 1) delete cardScales[id]; else cardScales[id] = beforeScale;
    set({ workspaceHistory: recordWorkspace(state, { cardScales, nodeOffsets: idRecord(beforeOffsets) }) });
  },
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

  carryPlacement: (offsets) => set({ nodeOffsets: idRecord(offsets), placementCarry: false }),

  resetNodeLayout: () => {
    const { nodeOffsets } = get();
    if (!Object.keys(nodeOffsets).length) return;
    set({
      nodeOffsets: idRecord<NodeOffset>(),
      workspaceHistory: recordWorkspace(get()),
    });
  },
});
