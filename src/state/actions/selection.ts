import { executeQuery } from "../../core/executor";
import { hasUnitStride } from "../../core/coordinates";
import { addPart, Box, Region, sameBox, subtractFromParts, translateAllParts, translatePart } from "../../core/region";
import { nudgeDelta, tileOf } from "../../view/tensor/grid";
import { nudgeBoxes } from "../../view/tensor/movement";
import { viewAxes } from "../../view/tensor/tensor-view";
import { gestureTile, tileFits, tileSpanAt } from "../../view/tensor/tile-spec";
import { analysisTarget, MAX_PREVIEW_NODES, operationForTensor, partsOn, SelPart } from "../../view/workspace";
import { recompute } from "../analysis";
import { recordWorkspace } from "../history";
import { editSelection, revealHidden } from "../selection-edit";
import { GetState, SetState, State } from "../types";

/** Share scope, rigid group translation, metadata, and hidden-axis following. */
function moveParts(
  get: GetState,
  set: SetState,
  record: boolean,
  move: (boxes: Box[], focused: number, shape: number[], state: State) => Box[]
): void {
  const state = get();
  const parts = state.selection?.parts;
  const target = analysisTarget(parts ?? [], state.analysisGroup, state.perBox ? state.focusedBox : null);
  const anchor = target.tensorId;
  const shape = anchor && state.resolved?.tensors[anchor]?.resolved;
  if (!parts || !anchor || !shape) return;
  const local = parts.flatMap((part, i) => part.tensorId === anchor ? [i] : []);
  const boxes = local.map((i) => parts[i].box);
  const focused = target.focusedBox === null ? -1 : local.indexOf(target.focusedBox);
  const moved = move(boxes, focused, shape, state);
  if (moved === boxes) return;
  const next = parts.slice();
  local.forEach((i, j) => { if (moved[j] !== boxes[j]) next[i] = { tensorId: anchor, box: moved[j] }; });
  editSelection(get, set, () => next, true, record);
  const oldBox = parts[target.index]?.box;
  const movedBox = next[target.index]?.box;
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
      hiddenBoxes: new Set<number>(),
      analysisGroup: null,
      preview: null,
    });
  },

  /**
   * Moves the focused part when one is focused, otherwise every part on the
   * anchor tensor. `axis` is an index into one tensor's shape, so it cannot be
   * applied across tensors of different rank -- parts elsewhere hold still.
   */
  moveSelection: (axis, delta, record = true) => {
    moveParts(get, set, record, (boxes, focused, shape) => {
      if (!Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length || !Number.isSafeInteger(delta)) return boxes;
      return focused >= 0 ? translatePart(boxes, focused, axis, delta, shape) : translateAllParts(boxes, axis, delta, shape);
    });
  },

  nudgeSelection: (axis, sign, multiplier = 1, record = true) => {
    moveParts(get, set, record, (boxes, focused, shape, state) => {
      if (!Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length ||
          (sign !== -1 && sign !== 1) || !Number.isSafeInteger(multiplier) || multiplier < 1) return boxes;
      const target = analysisTarget(state.selection!.parts, state.analysisGroup, state.perBox ? state.focusedBox : null);
      const cfg = state.viewCfgs[target.tensorId!];
      const units = state.snapToGrid
        ? gestureTile(shape, cfg, tileOf(shape, state.tileScale, state.graphPx, cfg))
        : shape.map(() => 1);
      // Non-contiguous axes keep the existing clamped translation policy.
      if (!hasUnitStride(shape, axis)) {
        const interval = state.selection!.parts[target.index].box[axis];
        const delta = nudgeDelta(interval, sign, units[axis], state.snapToGrid, multiplier);
        return focused >= 0 ? translatePart(boxes, focused, axis, delta, shape) : translateAllParts(boxes, axis, delta, shape);
      }
      const moving = focused >= 0 ? [boxes[focused]] : boxes;
      const moved = nudgeBoxes(moving, shape, units, axis, sign, multiplier);
      if (moved === moving) return boxes;
      return focused >= 0 ? boxes.map((box, i) => i === focused ? moved[0] : box) : moved;
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
