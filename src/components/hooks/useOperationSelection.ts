import { useEffect } from "react";
import { useStore } from "../../state/store";

/** Operation nodes and Ops rows are two surfaces for the same selection. */
export function dismissOperationOnPointer(event: Pick<PointerEvent, "target">): void {
  const target = event.target as Element | null;
  if (target?.closest?.(".op-node, .operation-row")) return;
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
