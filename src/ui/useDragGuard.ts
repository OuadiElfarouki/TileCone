import { useEffect } from "react";
import { useStore } from "./store";

/**
 * Suppress text selection for the duration of any drag.
 *
 * Without this, rubber-banding on a card or panning the canvas highlights
 * unrelated text in the side panel and inspector, because pointer capture does
 * not stop the browser's own selection gesture.
 *
 * This lives in a hook rather than inside the `setDragging` action on purpose:
 * the store is imported by tests that run under vitest's `node` environment,
 * where `document` does not exist. Keeping DOM effects out of store actions is
 * what lets the store stay headless.
 */
export function useDragGuard(): void {
  const dragging = useStore((s) => s.dragging);
  useEffect(() => {
    if (!dragging) return;
    const previous = document.body.style.userSelect;
    document.body.style.userSelect = "none";
    // drop any range the gesture already started before we got here
    window.getSelection()?.removeAllRanges();
    return () => {
      document.body.style.userSelect = previous;
    };
  }, [dragging]);
}
