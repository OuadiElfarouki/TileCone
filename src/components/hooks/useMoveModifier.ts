import { useEffect, useState } from "react";

/**
 * Whether an event carries the modifier that borrows the move tool: Ctrl, or
 * Cmd, the same pair the primary shortcuts accept. A press decides from its own
 * event rather than from the held state below, because the event is always
 * right and the held state can miss a key pressed while the window was elsewhere.
 */
export const borrowsMoveTool = (event: { ctrlKey: boolean; metaKey: boolean }): boolean =>
  event.ctrlKey || event.metaKey;

/**
 * Whether Ctrl/Cmd is held, so the canvas can show the move tool before the
 * press that uses it, as design tools change the cursor while the key is down.
 *
 * Every key and pointer event reports the modifiers, so each one resyncs the
 * state rather than counting presses and releases. Losing window focus clears
 * it: the release then lands somewhere this cannot hear.
 */
export function useMoveModifier(): boolean {
  const [held, setHeld] = useState(false);
  useEffect(() => {
    const sync = (event: KeyboardEvent | PointerEvent) => setHeld(borrowsMoveTool(event));
    const release = () => setHeld(false);
    window.addEventListener("keydown", sync);
    window.addEventListener("keyup", sync);
    window.addEventListener("pointermove", sync);
    window.addEventListener("blur", release);
    return () => {
      window.removeEventListener("keydown", sync);
      window.removeEventListener("keyup", sync);
      window.removeEventListener("pointermove", sync);
      window.removeEventListener("blur", release);
    };
  }, []);
  return held;
}
