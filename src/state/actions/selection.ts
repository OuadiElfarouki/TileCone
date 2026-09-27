import { executeQuery } from "../../core/executor";
import { addPart, Box, Region, subtractFromParts, translateAllParts, translatePart } from "../../core/region";
import { nudgeDelta, tileOf } from "../../view/tensor/grid";
import { viewAxes } from "../../view/tensor/tensor-view";
import { gestureTile, tileFits, tilePosition, tileSpanAt } from "../../view/tensor/tile-spec";
import { analysisTarget, MAX_PREVIEW_NODES, operationForTensor, partsOn, SelPart } from "../../view/workspace";
import { recompute } from "../analysis";
import { appendWorkspaceHistory, planEditOf } from "../history";
import { editSelection, revealHidden } from "../selection-edit";
import { GetState, SetState, State } from "../types";

/** Drawing, editing, focusing and enabling tiles, and the hover preview. */
export const selectionActions = (set: SetState, get: GetState): Pick<
  State,
  "setSelection"
  | "clearSelection"
  | "moveSelection"
  | "replaceBox"
  | "deleteBox"
  | "hoverBox"
  | "togglePinBox"
  | "clearFocus"
  | "selectAnalysisGroup"
  | "toggleBoxHidden"
  | "stepTile"
  | "setTensorTile"
  | "setPreviewBox"
> => ({
  setSelection: (tensorId, region, compose) => {
    const {
      selection,
      resolved,
      workspaceHistory,
      nodeOffsets,
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
      workspaceHistory: appendWorkspaceHistory(workspaceHistory, { selection, nodeOffsets, plan: planEditOf(get()) }),
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
    const { selection, workspaceHistory, nodeOffsets } = get();
    set({
      selection: null,
      selectedOp: null,
      workspaceHistory: selection
        ? appendWorkspaceHistory(workspaceHistory, { selection, nodeOffsets, plan: planEditOf(get()) })
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
    const state = get();
    const target = analysisTarget(state.selection?.parts ?? [], state.analysisGroup,
      state.perBox ? state.focusedBox : null);
    const focused = target.focusedBox;
    editSelection(
      get,
      set,
      (parts, shapeOf) => {
        const anchor = target.tensorId;
        if (!anchor) return parts;
        const shape = shapeOf(anchor);
        const local: number[] = [];
        const boxes: Box[] = [];
        parts.forEach((p, i) => {
          if (p.tensorId === anchor) {
            local.push(i);
            boxes.push(p.box);
          }
        });
        const at = focused !== null ? local.indexOf(focused) : -1;
        const moved =
          at >= 0
            ? translatePart(boxes, at, axis, delta, shape)
            : translateAllParts(boxes, axis, delta, shape);
        if (moved === boxes) return parts;
        const next = parts.slice();
        local.forEach((globalIndex, j) => {
          next[globalIndex] = { tensorId: anchor, box: moved[j] };
        });
        return next;
      },
      true,
      record
    );
    // A move along a hidden axis would otherwise carry the tile out of the
    // slice on screen; the view follows the tile the reader is studying.
    const moved = target.tensorId && get().selection?.parts[target.index];
    if (!moved) return;
    const { rowAxis, colAxis } = viewAxes(moved.box.map((interval) => interval.hi), get().viewCfgs[moved.tensorId]);
    if (axis !== rowAxis && axis !== colAxis) revealHidden(get, moved.tensorId, moved.box);
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

  stepTile: (index, axis, steps) => {
    const state = get();
    const part = state.selection?.parts[index];
    const shape = part && state.resolved?.tensors[part.tensorId]?.resolved;
    if (!part || !shape || axis < 0 || axis >= shape.length || steps === 0) return;
    const cfg = state.viewCfgs[part.tensorId];
    const unit = gestureTile(shape, cfg, tileOf(shape, state.tileScale, state.graphPx, cfg))[axis];
    const at = tilePosition(part.box[axis], unit, shape[axis]);
    let moved: Box;
    if (at.aligned) {
      // A tile moves to another tile, so the shortened last one is reachable
      // and a step back from it lands on the full tile before it.
      const coord = Math.max(0, Math.min(at.count - 1, at.coord + steps));
      if (coord === at.coord) return;
      moved = part.box.map((interval, ax) =>
        ax === axis ? tileSpanAt(coord * unit, unit, shape[axis]) : interval);
    } else {
      // Anything else keeps its extent and first lands an edge on the lattice.
      const delta = nudgeDelta(part.box[axis], steps > 0 ? 1 : -1, unit, true, Math.abs(steps));
      [moved] = translatePart([part.box], 0, axis, delta, shape);
      if (moved === part.box) return;
    }
    editSelection(
      get,
      set,
      (parts) => parts.map((p, i) => (i === index ? { ...p, box: moved } : p)),
      true
    );
    revealHidden(get, part.tensorId, moved);
  },

  setTensorTile: (tensorId, tile, refit) => {
    const state = get();
    const shape = state.resolved?.tensors[tensorId]?.resolved;
    if (!shape || (tile && !tileFits(shape, tile))) return;
    state.setViewCfg(tensorId, { tile: tile ?? undefined });
    const part = refit === undefined ? undefined : get().selection?.parts[refit];
    if (!tile || !part || part.tensorId !== tensorId) return;
    const box = part.box.map((interval, axis) => tileSpanAt(interval.lo, tile[axis], shape[axis]));
    if (box.every((interval, axis) =>
      interval.lo === part.box[axis].lo && interval.hi === part.box[axis].hi)) return;
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
