/**
 * The one rendering of a menu described as data (`view/menu.ts`): settings and
 * actions that are occasional rather than part of reading what they belong to.
 *
 * A menu opens from a right-click on its owner - or, from the keyboard, the
 * Menu key or Shift+F10, which raise the same event - and, where the owner has
 * room for one, from a ⋯ button. The owner stays a readout, and its settings
 * are one gesture away instead of permanently on screen.
 *
 * The menu renders into the document body at fixed coordinates, so neither a
 * scrolling panel nor the graph's transform can clip or scale it. It carries
 * `role="menu"`, which the global key bindings stand down for: the arrows walk
 * its entries rather than moving the inspected tile.
 */

import React, { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { MenuHandlers, MenuSpec } from "../../view/menu";

/** Where a menu opens: its top-left corner, or its top-right for `end`. */
type Anchor = { x: number; y: number; align: "start" | "end" };

export type OptionsMenuState = {
  anchor: Anchor | null;
  /** The ⋯ button, when the owner has one: a press on it is not a press outside. */
  triggerRef: React.RefObject<HTMLButtonElement>;
  /** Open below an element, right edges aligned: the ⋯ button's placement. */
  openBelow: (element: HTMLElement) => void;
  /**
   * The owner's context-menu handler. Opens at the pointer for a pointer, and
   * below the focused element for the Menu key or Shift+F10.
   */
  onContextMenu: (event: React.MouseEvent) => void;
  close: (refocus: boolean) => void;
};

export function useOptionsMenu(): OptionsMenuState {
  const [anchor, setAnchor] = useState<Anchor | null>(null);
  // Focus goes back where it came from, whichever way the menu was opened.
  const returnTo = useRef<HTMLElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const open = (next: Anchor) => {
    returnTo.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setAnchor(next);
  };
  return {
    anchor,
    triggerRef,
    openBelow: (element) => {
      const r = element.getBoundingClientRect();
      open({ x: r.right, y: r.bottom + EDGE, align: "end" });
    },
    onContextMenu: (event) => {
      // A text field keeps the browser's own menu: copy and paste live there.
      if ((event.target as HTMLElement).closest("input, textarea")) return;
      event.preventDefault();
      // A right-click reports the secondary button at the element under the
      // pointer. The Menu key and Shift+F10 raise the event on the focused
      // element without it, at a position browsers choose differently - Chrome
      // puts it inside the element, where the menu would cover what it is for.
      const target = event.target as HTMLElement;
      const fromKeyboard = event.button !== 2 && target === document.activeElement;
      if (fromKeyboard) {
        const r = target.getBoundingClientRect();
        open({ x: r.left, y: r.bottom + EDGE, align: "start" });
      } else open({ x: event.clientX, y: event.clientY, align: "start" });
    },
    close: (refocus) => {
      setAnchor(null);
      const target = returnTo.current;
      returnTo.current = null;
      if (refocus && target?.isConnected) target.focus();
    },
  };
}

const EDGE = 4;
const ENTRY = "[role=menuitem], [role=menuitemradio]";

/**
 * The menu is a surface of its own: nothing done in it means anything to what
 * lies behind it. It renders through a portal, and React bubbles a portal's
 * events to the owner and the owner's ancestors - without this, a press in a
 * card's menu would start a pan of the graph, a wheel would zoom it, and a
 * right-click would reopen the menu through its owner.
 */
const contain = (e: React.SyntheticEvent) => e.stopPropagation();

/** The ⋯ button that opens a menu, for an owner with room for one. */
export function OptionsMenuButton({
  menu,
  label,
}: {
  menu: OptionsMenuState;
  label: string;
}): React.ReactElement {
  return (
    <button
      ref={menu.triggerRef}
      className={`mini options-trigger${menu.anchor ? " on" : ""}`}
      aria-label={label}
      aria-haspopup="menu"
      aria-expanded={!!menu.anchor}
      title={`${label} (or right-click)`}
      onClick={(e) => (menu.anchor ? menu.close(false) : menu.openBelow(e.currentTarget))}
    >
      ⋯
    </button>
  );
}

export function OptionsMenu<A extends string, C extends string = never>({
  menu,
  label,
  spec,
  handlers,
}: {
  menu: OptionsMenuState;
  /** Names the menu for assistive technology, e.g. "axis options". */
  label: string;
  spec: MenuSpec<A, C>;
  handlers: MenuHandlers<A, C>;
}): React.ReactElement | null {
  const { anchor, close, triggerRef } = menu;
  const listRef = useRef<HTMLUListElement>(null);
  const [place, setPlace] = useState<{ left: number; top: number } | null>(null);
  const groupId = useId();

  // Keep the whole menu on screen: flip above the anchor, or pull it left,
  // when it would run past the viewport's edge.
  useLayoutEffect(() => {
    const list = listRef.current;
    if (!anchor || !list) {
      setPlace(null);
      return;
    }
    const { width, height } = list.getBoundingClientRect();
    let left = anchor.align === "end" ? anchor.x - width : anchor.x;
    let top = anchor.y;
    if (left + width > window.innerWidth - EDGE) left = window.innerWidth - EDGE - width;
    if (top + height > window.innerHeight - EDGE) top = Math.max(EDGE, anchor.y - height);
    setPlace({ left: Math.max(EDGE, left), top });
  }, [anchor]);

  // Focus moves in once the menu is placed: until then it is hidden while it
  // is measured, and a hidden element cannot take focus. Left on the owner,
  // the arrows would reach the global bindings and move the tile.
  const placed = place !== null;
  useEffect(() => {
    if (placed) listRef.current?.querySelector<HTMLElement>(ENTRY)?.focus();
  }, [placed]);

  // A press anywhere else closes the menu, and so does anything that moves the
  // page under a fixed menu.
  useEffect(() => {
    if (!anchor) return;
    // The menu's own trigger toggles it on click; counting its press as
    // outside would close the menu only for the click to reopen it.
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (listRef.current?.contains(target) || triggerRef.current?.contains(target)) return;
      close(false);
    };
    const onMove = () => close(false);
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [anchor]);

  if (!anchor || !spec.length) return null;

  const onKeyDown = (e: React.KeyboardEvent) => {
    const entries = [...(listRef.current?.querySelectorAll<HTMLElement>(ENTRY) ?? [])];
    const at = entries.indexOf(document.activeElement as HTMLElement);
    const go = (i: number) => {
      e.preventDefault();
      entries[(i + entries.length) % entries.length]?.focus();
    };
    // Entries are walked in reading order, a choice's options included, so
    // either pair of arrows moves through a row of options or down the list.
    if (e.key === "ArrowDown" || e.key === "ArrowRight") go(at + 1);
    else if (e.key === "ArrowUp" || e.key === "ArrowLeft") go(at - 1);
    else if (e.key === "Home") go(0);
    else if (e.key === "End") go(entries.length - 1);
    else if (e.key === "Escape") {
      e.preventDefault();
      close(true);
    } else if (e.key === "Tab") close(false);
  };

  return createPortal(
    <ul
      ref={listRef}
      className="options-menu"
      role="menu"
      aria-label={label}
      style={place ?? { left: anchor.x, top: anchor.y, visibility: "hidden" }}
      onKeyDown={(e) => {
        contain(e);
        onKeyDown(e);
      }}
      onContextMenu={(e) => {
        contain(e);
        e.preventDefault();
      }}
      onPointerDown={contain}
      onPointerMove={contain}
      onPointerUp={contain}
      onClick={contain}
      onWheel={contain}
    >
      {spec.map((entry, i) => {
        if (entry.kind === "separator") return <li key={i} role="separator" className="options-separator" />;
        if (entry.kind === "action")
          return (
            <li key={entry.id} role="none">
              <button
                role="menuitem"
                className="options-item"
                aria-disabled={entry.disabled || undefined}
                title={entry.title}
                onClick={() => {
                  if (entry.disabled) return;
                  close(true);
                  handlers.action[entry.id]();
                }}
              >
                {entry.label}
              </button>
            </li>
          );
        const labelId = `${groupId}-${entry.id}`;
        return (
          <li key={entry.id} role="none" className="options-choice">
            <span id={labelId} className="options-choice-label">{entry.label}</span>
            <span role="group" aria-labelledby={labelId} className="options-choice-values">
              {entry.options.map((option) => (
                <button
                  key={option.value}
                  role="menuitemradio"
                  className={`options-value${option.checked ? " on" : ""}`}
                  aria-checked={option.checked}
                  title={option.title}
                  onClick={() => {
                    if (!option.checked) handlers.choice[entry.id](option.value);
                  }}
                >
                  {option.label}
                </button>
              ))}
            </span>
          </li>
        );
      })}
    </ul>,
    document.body
  );
}
