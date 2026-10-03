import { useEffect } from "react";
import { useStore } from "../../state/store";

/** Operation nodes and Ops rows are two surfaces for the same selection, and a
 * press anywhere else lets it go - except on the graph canvas outside a card.
 * A press on its background may start a pan and one on its controls a zoom,
 * which change where the reader is looking rather than what they chose, so the
 * canvas decides at release (`GraphView`). A press on a card still lets go. */
export function dismissOperationOnPointer(event: Pick<PointerEvent, "target">): void {
  const target = event.target as Element | null;
  if (target?.closest?.(".op-node, .operation-row")) return;
  if (target?.closest?.(".graph-canvas") && !target.closest(".card-slot")) return;
  const state = useStore.getState();
  if (state.selectedOp !== null) state.setSelectedOp(null);
}

export function dismissOperationOnKey(event: Pick<KeyboardEvent, "key">): void {
  if (event.key !== "Escape") return;
  const state = useStore.getState();
  if (state.selectedOp !== null) state.setSelectedOp(null);
}

/** Clear the visual selection without consuming another control's cancel or click. */
export function useOperationSelection(): void {
  useEffect(() => {
    // Capture also sees presses and Escape handled by fields, menus or drags.
    document.addEventListener("pointerdown", dismissOperationOnPointer, true);
    document.addEventListener("keydown", dismissOperationOnKey, true);
    return () => {
      document.removeEventListener("pointerdown", dismissOperationOnPointer, true);
      document.removeEventListener("keydown", dismissOperationOnKey, true);
    };
  }, []);
}
