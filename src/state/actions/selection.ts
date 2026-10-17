import { executeQuery } from "../../core/executor";
import { hasUnitStride } from "../../core/coordinates";
import { addPart, Box, Region, sameBox, subtractFromParts, translatePart } from "../../core/region";
import { nudgeDelta, nudgeUnit } from "../../view/tensor/grid";
import { activeGrowth, growthDelta, stepEdge } from "../../view/growth";
import { displayTile } from "../../view/tensor/tiling";
import { nudgeBoxes } from "../../view/tensor/movement";
import { viewAxes } from "../../view/tensor/tensor-view";
import { gestureTile, tileFits, tileSpanAt } from "../../view/tensor/tile-spec";
import { keyboardTile, MAX_PREVIEW_NODES, operationForTensor, partsOn, SelPart } from "../../view/workspace";
import { recompute } from "../analysis";
import { recordWorkspace } from "../history";
import { editSelection, revealHidden } from "../selection-edit";
import { GetState, SetState, State } from "../types";

/** The keyboard's tile, metadata, and hidden-axis following. With no keyboard
 * tile (`keyboardTile`) nothing moves. */
function moveParts(
  get: GetState,
  set: SetState,
  record: boolean,
  move: (box: Box, shape: number[], state: State) => Box
): void {
  const state = get();
  const parts = state.selection?.parts;
  const index = keyboardTile(state);
  if (!parts || index === null) return;
  const anchor = parts[index].tensorId;
  const shape = state.resolved?.tensors[anchor]?.resolved;
  if (!shape) return;
  const oldBox = parts[index].box;
  const movedBox = move(oldBox, shape, state);
  if (movedBox === oldBox) return;
  const next = parts.slice();
  next[index] = { tensorId: anchor, box: movedBox };
  editSelection(get, set, () => next, true, record);
  const { rowAxis, colAxis } = viewAxes(shape, get().viewCfgs[anchor]);
  const hiddenMoved = oldBox?.some((interval, axis) => axis !== rowAxis && axis !== colAxis && interval.lo !== movedBox[axis].lo);
  if (hiddenMoved) revealHidden(get, anchor, movedBox);
}

/** Drawing, editing, focusing and enabling tiles, and the hover preview. */
export const selectionActions = (set: SetState, get: GetState): Pick<
  State,
  "setSelection"
  | "clearSelection"
  | "moveSelection"
  | "nudgeSelection"
  | "replaceBox"
  | "deleteBox"
  | "hoverBox"
  | "togglePinBox"
  | "clearFocus"
  | "releaseEditBox"
  | "growSelection"
  | "selectAnalysisGroup"
  | "toggleBoxHidden"
  | "setTensorTile"
  | "setPreviewBox"
> => ({
  setSelection: (tensorId, region, compose) => {
    const {
      selection,
      resolved,
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
    // The tile just drawn is the one the keyboard moves next. Drawing a box
    // that is already a tile adds nothing, and selects that tile instead.
    const last = drawn[drawn.length - 1];
    const editBox = mode === "subtract" || !last
      ? null
      : parts.findIndex((part) => part.tensorId === tensorId && sameBox(part.box, last));
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
      workspaceHistory: recordWorkspace(get()),
      // Drawing releases the pin, and the analysis follows the pointer to this
      // tensor. Remapping it was never able to keep a pin on the tensor being
      // drawn on - those parts are rebuilt, so their identity is gone - and
      // kept one on any *other* tensor, which is exactly backwards: it left the
      // panel describing the tile the reader had just left while the tile they
      // drew sat dimmed in another group.
      focusedBox: null,
      pinnedBox: null,
      editBox: editBox === null || editBox < 0 ? null : editBox,
      hiddenBoxes: nextHidden,
      // A subtract gesture can empty the tensor it was aimed at; the store then
      // holds no group rather than one nothing is drawn on.
      analysisGroup: parts.some((part) => part.tensorId === tensorId) ? tensorId : null,
      preview: null,
      ...recompute(resolved, sel, { selection, perBox, entangled }),
    });
  },

  clearSelection: () => {
    const { selection, workspaceHistory } = get();
    set({
      selection: null,
      selectedOp: null,
      workspaceHistory: selection
        ? recordWorkspace(get())
        : workspaceHistory,
      backwardRes: null,
      byTensorRes: null,
      forwardRes: null,
      perBox: null,
      entangled: null,
      focusedBox: null,
      pinnedBox: null,
      editBox: null,
      growth: null,
      hiddenBoxes: new Set<number>(),
      analysisGroup: null,
      preview: null,
    });
  },

  /**
   * Moves the keyboard's tile (`keyboardTile`), and nothing when there is
   * none. `axis` is an index into that tile's tensor's shape.
   */
  moveSelection: (axis, delta, record = true) => {
    moveParts(get, set, record, (box, shape) => {
      if (!Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length || !Number.isSafeInteger(delta)) return box;
      return translatePart([box], 0, axis, delta, shape)[0];
    });
  },

  nudgeSelection: (axis, sign, multiplier = 1, record = true) => {
    moveParts(get, set, record, (box, shape, state) => {
      if (!Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length ||
          (sign !== -1 && sign !== 1) || !Number.isSafeInteger(multiplier) || multiplier < 1) return box;
      const cfg = state.viewCfgs[state.selection!.parts[keyboardTile(state)!].tensorId];
      const units = state.snapToGrid
        ? gestureTile(shape, cfg, displayTile(shape, state.tileScale))
        : shape.map(() => 1);
      // Non-contiguous axes keep the existing clamped translation policy.
      if (!hasUnitStride(shape, axis)) {
        const delta = nudgeDelta(box[axis], sign, units[axis], state.snapToGrid, multiplier);
        return translatePart([box], 0, axis, delta, shape)[0];
      }
      const moving = [box];
      const moved = nudgeBoxes(moving, shape, units, axis, sign, multiplier);
      return moved === moving ? box : moved[0];
    });
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
    // Either way the click was on this tile, so it is the one the keyboard edits.
    set({ pinnedBox: pinned, focusedBox: pinned, analysisGroup: tensorId, editBox: index });
  },

  clearFocus: () => set({ pinnedBox: null, focusedBox: null, editBox: null }),

  releaseEditBox: () => {
    if (get().editBox !== null) set({ editBox: null });
  },

  growSelection: (axis, grow, record = true) => {
    const state = get();
    const parts = state.selection?.parts;
    const index = keyboardTile(state);
    if (!parts || index === null || !state.resolved) return;
    const part = parts[index];
    const shape = state.resolved.tensors[part.tensorId].resolved!;
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length) return;
    // The step is the arrows' unit: a tile while snapping, an element when not.
    const unit = nudgeUnit(shape, state.viewCfgs[part.tensorId], state.tileScale, state.snapToGrid, axis);
    const edge = stepEdge(part.box[axis], unit, shape[axis], grow);
    if (!edge) return;
    const box = part.box.map((interval, a) => (a === axis ? edge : interval));
    // A step continues the growth the last one left; anything else - a move,
    // an edit, an undo - has changed the box, and this step starts afresh.
    const baseBox = activeGrowth(state.growth, parts, index)?.baseBox ?? part.box;
    editSelection(get, set, (current) => current.map((p, i) => (i === index ? { tensorId: p.tensorId, box } : p)), true, record);
    const installed = get().selection!.parts[index];
    set({
      growth: { index, tensorId: part.tensorId, baseBox, box, part: installed, delta: growthDelta(state.resolved, part.tensorId, baseBox, box) },
    });
  },

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

  setTensorTile: (tensorId, tile, refit) => {
    const state = get();
    const shape = state.resolved?.tensors[tensorId]?.resolved;
    if (!shape || (tile && !tileFits(shape, tile))) return;
    state.setViewCfg(tensorId, { tile: tile ?? undefined });
    const part = refit === undefined ? undefined : get().selection?.parts[refit];
    if (!tile || !part || part.tensorId !== tensorId) return;
    const box = part.box.map((interval, axis) => tileSpanAt(interval.lo, tile[axis], shape[axis]));
    if (sameBox(box, part.box)) return;
    get().replaceBox(refit!, box);
    revealHidden(get, tensorId, box);
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
});
