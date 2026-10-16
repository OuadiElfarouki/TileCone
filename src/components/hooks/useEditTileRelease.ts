import { useEffect } from "react";
import { useStore } from "../../state/store";

/**
 * The keyboard's tile (`editBox`) is let go by a press anywhere that is not
 * about it: another card, an operation, the source panel, the header. Two
 * places keep it. Its own card, where a press either draws the next tile -
 * which then takes over - or works the card's own controls; and the
 * inspector, which describes and edits the tile. The graph canvas outside a
 * card decides at release, as for operations (`GraphView`): a press there may
 * start a pan or work a zoom control, which move the view and choose nothing.
 */
export function releaseEditTileOnPointer(event: Pick<PointerEvent, "target">): void {
  const state = useStore.getState();
  if (state.editBox === null) return;
  const target = event.target as Element | null;
  if (target?.closest?.(".inspector")) return;
  const card = target?.closest?.(".card-slot");
  if (card && card.getAttribute("data-tensor-id") === state.selection?.parts[state.editBox]?.tensorId) return;
  if (!card && target?.closest?.(".graph-canvas") && !target.closest(".op-node")) return;
  state.releaseEditBox();
}

export function useEditTileRelease(): void {
  useEffect(() => {
    // Capture sees presses that cards, menus and drags handle themselves.
    document.addEventListener("pointerdown", releaseEditTileOnPointer, true);
    return () => document.removeEventListener("pointerdown", releaseEditTileOnPointer, true);
  }, []);
}
