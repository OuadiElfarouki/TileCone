import React, { useLayoutEffect, useState } from "react";
import { createPortal } from "react-dom";

/** The element on the canvas bar that holds controls for what the canvas plays. */
export const CANVAS_SLOT_ID = "canvas-controls-slot";

/**
 * Render `children` on the canvas bar while they stay owned where they are
 * declared.
 *
 * A replay is watched on the cards, so its transport belongs beside them; but
 * its timers live with the inspector, which retires them when it unmounts. A
 * portal gives both: the controls show on the canvas and leave with their
 * owner. Where there is no canvas bar - a server render, or before the canvas
 * has mounted - they render in place instead.
 */
export function OnCanvas({ children }: { children: React.ReactNode }): React.ReactElement {
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  useLayoutEffect(() => {
    setSlot(document.getElementById(CANVAS_SLOT_ID));
  }, []);
  return slot ? createPortal(children, slot) : <>{children}</>;
}
