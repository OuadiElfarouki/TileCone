import { Box } from "../core/region";
import { viewAxes } from "../view/tensor/tensor-view";
import { anchorTensorId, operationForTensor, SelPart } from "../view/workspace";
import { recompute } from "./analysis";
import { appendWorkspaceHistory, planEditOf } from "./history";
import { State } from "./types";

/**
 * Keep a box on screen after it moved along a hidden axis.
 *
 * In slice mode the card shows one index per hidden axis, and in projection a
 * tensor with a tile of its own reads the same position as "the tile a gesture
 * takes". Either way a position outside the box would leave the reader looking
 * at, or about to draw on, a different tile from the one just placed. A
 * position already inside the box is left alone.
 */
export function revealHidden(get: () => State, tensorId: string, box: Box): void {
  const state = get();
  const shape = state.resolved?.tensors[tensorId]?.resolved;
  const cfg = state.viewCfgs[tensorId];
  if (!shape || !cfg) return;
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  let changed = false;
  const sliders = cfg.sliders.slice();
  box.forEach((interval, axis) => {
    if (axis === rowAxis || axis === colAxis) return;
    const at = sliders[axis] ?? 0;
    if (at >= interval.lo && at < interval.hi) return;
    sliders[axis] = interval.lo;
    changed = true;
  });
  if (changed) state.setViewCfg(tensorId, { sliders });
}

/**
 * Apply a transform to the selection's parts and repropagate.
 * `keepFocus` holds the focused part across edits that preserve indices (a move);
 * edits that reorder or remove parts drop it so a stale index can never be used.
 */
export function editSelection(
  get: () => State,
  set: (partial: Partial<State>) => void,
  fn: (parts: SelPart[], shapeOf: (tensorId: string) => number[]) => SelPart[],
  keepFocus = false,
  record = true
): void {
  const {
    selection,
    resolved,
    workspaceHistory,
    tensorOffsets,
    focusedBox,
    perBox,
    entangled,
  } = get();
  if (!selection || !resolved) return;
  const shapeOf = (tensorId: string) => resolved.tensors[tensorId].resolved!;
  const parts = fn(selection.parts, shapeOf);
  const sel = parts.length === 0 ? null : { parts };
  const nextFocus =
    keepFocus && sel && focusedBox !== null && focusedBox < parts.length ? focusedBox : null;
  // Moving or editing a tile is working at its operation, so the list follows.
  // The anchor is the same tile the keyboard acts on, so the row that lights is
  // the one the reader is driving.
  const anchor = anchorTensorId(sel, nextFocus);
  set({
    selectedOp: operationForTensor(resolved, anchor),
    selection: sel,
    workspaceHistory: record
      ? appendWorkspaceHistory(workspaceHistory, { selection, tensorOffsets, plan: planEditOf(get()) })
      : workspaceHistory,
    focusedBox: nextFocus,
    pinnedBox: nextFocus === null ? null : get().pinnedBox,
    // `keepFocus` marks the edits that preserve part order and length (a move).
    // Anything else can renumber the parts, which would leave these indexes
    // pointing at the wrong cone.
    hiddenBoxes: keepFocus ? get().hiddenBoxes : new Set<number>(),
    // A group is named by tensor, not by index, so an edit that renumbers parts
    // leaves it intact. Deleting the last tile on it is what retires it.
    analysisGroup: parts.some((part) => part.tensorId === get().analysisGroup)
      ? get().analysisGroup
      : null,
    preview: null,
    ...recompute(resolved, sel, { selection, perBox, entangled }),
  });
}
