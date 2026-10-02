/**
 * A menu for actions that are occasional rather than part of reading a
 * section: opened from a ⋯ button, or by right-clicking the section. The
 * section stays a readout, and the actions are one gesture away instead of
 * permanently on screen.
 *
 * The menu renders into the document body at fixed coordinates, so a
 * scrolling panel cannot clip it, and it carries `role="menu"`, which the
 * global key bindings stand down for: the arrows walk the items rather than
 * moving the inspected tile.
 */

import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export type MenuItem = {
  label: string;
  /** What the item does, or why it is unavailable. */
  title: string;
  disabled?: boolean;
  onSelect: () => void;
};

/** Where the menu opens: at a pointer, or below the ⋯ button's right edge. */
type Anchor = { x: number; y: number; align: "left" | "right" };

export type OptionsMenuHandle = {
  anchor: Anchor | null;
  /** Open at a pointer position, as a context menu does. */
  openAt: (x: number, y: number) => void;
  /** Opens the menu on a right-click, except over a text field. */
  onContextMenu: (event: React.MouseEvent) => void;
  set: (anchor: Anchor | null) => void;
};

export function useOptionsMenu(): OptionsMenuHandle {
  const [anchor, set] = useState<Anchor | null>(null);
  const openAt = (x: number, y: number) => set({ x, y, align: "left" });
  return {
    anchor,
    openAt,
    set,
    onContextMenu: (event) => {
      // A text field keeps the browser's own menu: copy and paste live there.
      if ((event.target as HTMLElement).closest("input, textarea")) return;
      event.preventDefault();
      openAt(event.clientX, event.clientY);
    },
  };
}

const EDGE = 4;

export function OptionsMenu({
  menu,
  label,
  items,
}: {
  menu: OptionsMenuHandle;
  /** Names the menu for assistive technology, e.g. "axis options". */
  label: string;
  items: MenuItem[];
}): React.ReactElement {
  const { anchor, set } = menu;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const [place, setPlace] = useState<{ left: number; top: number } | null>(null);

  const close = (refocus: boolean) => {
    set(null);
    setPlace(null);
    if (refocus) triggerRef.current?.focus();
  };

  // Keep the whole menu on screen: flip above or to the left of the anchor
  // when it would run past the viewport's edge.
  useLayoutEffect(() => {
    const list = listRef.current;
    if (!anchor || !list) return;
    const { width, height } = list.getBoundingClientRect();
    let left = anchor.align === "right" ? anchor.x - width : anchor.x;
    let top = anchor.y;
    if (left + width > window.innerWidth - EDGE) left = window.innerWidth - EDGE - width;
    if (top + height > window.innerHeight - EDGE) top = Math.max(EDGE, anchor.y - height);
    setPlace({ left: Math.max(EDGE, left), top });
  }, [anchor]);

  // Focus moves in once the menu is placed: until then it is hidden while it
  // is measured, and a hidden element cannot take focus. With focus left on
  // the trigger, the arrows would reach the global bindings and move the tile.
  const placed = place !== null;
  useEffect(() => {
    if (placed) (listRef.current?.querySelector("[role=menuitem]") as HTMLElement | null)?.focus();
  }, [placed]);

  // Anything that moves the page under a fixed menu closes it, as does a
  // press anywhere else.
  useEffect(() => {
    if (!anchor) return;
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (!listRef.current?.contains(target) && !triggerRef.current?.contains(target)) close(false);
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

  const onKeyDown = (e: React.KeyboardEvent) => {
    const entries = [...(listRef.current?.querySelectorAll<HTMLElement>("[role=menuitem]") ?? [])];
    const at = entries.indexOf(document.activeElement as HTMLElement);
    const go = (i: number) => {
      e.preventDefault();
      entries[(i + entries.length) % entries.length]?.focus();
    };
    if (e.key === "ArrowDown") go(at + 1);
    else if (e.key === "ArrowUp") go(at - 1);
    else if (e.key === "Home") go(0);
    else if (e.key === "End") go(entries.length - 1);
    else if (e.key === "Escape") {
      e.preventDefault();
      close(true);
    } else if (e.key === "Tab") close(false);
  };

  return (
    <>
      <button
        ref={triggerRef}
        className={`mini options-trigger${anchor ? " on" : ""}`}
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={!!anchor}
        title={`${label} (or right-click)`}
        onClick={() => {
          if (anchor) return close(false);
          const r = triggerRef.current!.getBoundingClientRect();
          set({ x: r.right, y: r.bottom + EDGE, align: "right" });
        }}
        onKeyDown={(e) => {
          if (anchor && e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            close(true);
          }
        }}
      >
        ⋯
      </button>
      {anchor &&
        createPortal(
          <ul
            ref={listRef}
            className="options-menu"
            role="menu"
            aria-label={label}
            style={place ?? { left: anchor.x, top: anchor.y, visibility: "hidden" }}
            onKeyDown={onKeyDown}
            onContextMenu={(e) => e.preventDefault()}
          >
            {items.map((item) => (
              <li key={item.label} role="none">
                <button
                  role="menuitem"
                  className="options-item"
                  aria-disabled={item.disabled || undefined}
                  title={item.title}
                  onClick={() => {
                    if (item.disabled) return;
                    close(true);
                    item.onSelect();
                  }}
                >
                  {item.label}
                </button>
              </li>
            ))}
          </ul>,
          document.body
        )}
    </>
  );
}
