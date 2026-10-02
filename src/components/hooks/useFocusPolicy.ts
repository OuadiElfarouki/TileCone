import { useEffect } from "react";

/**
 * Keyboard focus follows what the reader is working on, never the path the
 * pointer took. The global bindings - the arrows that move a tile, the letters
 * that switch views - act whenever focus is not in something that owns those
 * keys, so whatever holds focus decides what a key does. Two rules, installed
 * once for the whole app, keep that holder where the reader expects it.
 *
 * 1. A press releases what it lands outside of. The browser does this itself,
 *    but the workspace's gestures - drawing on a card, dragging one, panning -
 *    cancel the press to keep text from being selected, and that cancels the
 *    focus change with it. The source editor then kept the keys after a tile
 *    was drawn, and a `d` meant to toggle the Forward Cone was typed into the
 *    graph source.
 *
 * 2. A pointer leaves no control holding the keys. A button or slider used
 *    with the pointer is released when the press ends, so the keys return to
 *    the workspace: the arrows move the tile just drawn, not the detail slider
 *    dragged a moment before, which held them with no sign that it did. A
 *    control reached with the keyboard keeps focus, and shows it.
 *
 * Text fields are exempt from the second rule, since typing is what they are
 * for. Menus, list boxes and dialogs are exempt from both: they move focus
 * within themselves.
 */

const MANAGES_FOCUS = '[role="menu"], [role="listbox"], [role="dialog"]';
/** Controls the pointer operates by pressing or dragging, with nothing to type. */
const POINTER_CONTROL =
  'button, input[type="range"], input[type="checkbox"], input[type="radio"], [role="slider"], [role="separator"]';

/** The part of an element these rules read; a DOM-free seam for tests. */
export type FocusNode = {
  closest(selector: string): unknown;
  matches(selector: string): boolean;
  contains(other: unknown): boolean;
};

/** @internal Rule 1: whether a press on `target` releases the focused `active`. */
export function pressReleases(active: FocusNode, target: FocusNode | null): boolean {
  if (active.closest(MANAGES_FOCUS)) return false;
  if (!target) return true;
  return !active.contains(target) && !target.closest(MANAGES_FOCUS);
}

/** @internal Rule 2: whether the focused `active` lets go once a pointer press ends. */
export function releasedAfterPointer(active: FocusNode): boolean {
  return active.matches(POINTER_CONTROL) && !active.closest(MANAGES_FOCUS);
}

export function useFocusPolicy(): void {
  useEffect(() => {
    const focused = () => {
      const active = document.activeElement;
      return active instanceof HTMLElement && active !== document.body ? active : null;
    };
    const onPointerDown = (e: PointerEvent) => {
      const active = focused();
      const target = e.target instanceof Element ? e.target : null;
      if (active && pressReleases(active, target)) active.blur();
    };
    const onPointerUp = () => {
      const active = focused();
      if (active && releasedAfterPointer(active)) active.blur();
    };
    // Capture phase: the policy runs before any handler that cancels the press.
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("pointerup", onPointerUp, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("pointerup", onPointerUp, true);
    };
  }, []);
}
