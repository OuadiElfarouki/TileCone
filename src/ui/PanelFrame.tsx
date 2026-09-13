import React, { useLayoutEffect, useRef, useState } from "react";
import {
  PANEL_COLLAPSE_AT,
  PANEL_MAX,
  PANEL_MIN,
  PANEL_RAIL,
  PanelSide,
  useStore,
} from "./store";

/**
 * Width, collapse-to-rail, and the drag strip shared by both side panels.
 *
 * The strip always sits on the panel's *inner* edge : the one facing the canvas
 * : so the gesture reads as pushing the canvas boundary rather than dragging the
 * window frame. Dragging far enough inward collapses instead of clamping, which
 * is how VS Code behaves and means one gesture both resizes and closes.
 */
export function PanelFrame({
  side,
  label,
  children,
}: {
  side: PanelSide;
  label: string;
  children: React.ReactNode;
}): React.ReactElement {
  const width = useStore((s) => s.panelW[side]);
  const collapsed = useStore((s) => s.panelCollapsed[side]);
  const setPanelWidth = useStore((s) => s.setPanelWidth);
  const finishPanelResize = useStore((s) => s.finishPanelResize);
  const togglePanel = useStore((s) => s.togglePanel);
  const setDragging = useStore((s) => s.setDragging);
  const frameRef = useRef<HTMLDivElement>(null);
  const railRef = useRef<HTMLButtonElement>(null);
  const collapseRef = useRef<HTMLButtonElement>(null);
  const wasCollapsedRef = useRef(collapsed);
  const dragRef = useRef(false);
  const rawWidthRef = useRef(width);
  const startWidthRef = useRef(width);
  /**
   * Whether releasing now would collapse the panel.
   *
   * The preview clamps at `PANEL_MIN` while the commit tests the raw width
   * against `PANEL_COLLAPSE_AT`, so between the two the panel has stopped
   * moving while the pointer keeps travelling — and 64px later, letting go
   * makes it vanish. Nothing on screen distinguished "will spring back" from
   * "will collapse", and the only signal was a title on a 4px strip, which does
   * not appear mid-drag. This state is what makes the gesture read as pushing
   * the panel off the edge, which is what it is.
   */
  const [willCollapse, setWillCollapse] = useState(false);

  /* The control that caused a collapse/expand disappears in the next render.
     Put focus on its replacement instead of dropping keyboard users onto the
     document body. Initial collapsed state is not a transition and gets no
     unsolicited focus. */
  useLayoutEffect(() => {
    if (wasCollapsedRef.current === collapsed) return;
    (collapsed ? railRef.current : collapseRef.current)?.focus();
    wasCollapsedRef.current = collapsed;
  }, [collapsed]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = true;
    setWillCollapse(false);
    rawWidthRef.current = width;
    startWidthRef.current = width;
    setDragging(true); // suppresses text selection for the duration
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const rect = frameRef.current!.getBoundingClientRect();
    // measure from the panel's outer edge, so the pointer tracks the strip
    const raw = side === "left" ? e.clientX - rect.left : rect.right - e.clientX;
    rawWidthRef.current = raw;
    setWillCollapse(raw < PANEL_COLLAPSE_AT);
    // Width previews stay open and usable. Crossing the collapse threshold is
    // committed only on pointerup, so the captured element cannot disappear
    // before it has a chance to end the drag.
    setPanelWidth(side, raw);
  };
  const endDrag = (commit: boolean) => {
    if (!dragRef.current) return;
    dragRef.current = false;
    setWillCollapse(false);
    if (commit) finishPanelResize(side, rawWidthRef.current);
    else setPanelWidth(side, startWidthRef.current);
    setDragging(false);
  };

  return (
    <div
      ref={frameRef}
      className={`panel-frame ${side}${collapsed ? " collapsed" : ""}${
        willCollapse ? " will-collapse" : ""
      }`}
      style={{ width: collapsed ? PANEL_RAIL : width }}
    >
      {/* Draft source and selection state live in the store. Unmount transient
          panel UI while collapsed so a hidden field cannot retain focus and an
          open picker cannot reappear later in stale local state. */}
      {!collapsed && <div className="panel-content">{children}</div>}
      {collapsed ? (
        <button
          ref={railRef}
          className={`panel-rail ${side}`}
          onClick={() => togglePanel(side)}
          title={`show ${label} (alt+${side === "left" ? "1" : "2"})`}
        >
          <span>{label}</span>
        </button>
      ) : (
        <>
          <button
            ref={collapseRef}
            className={`panel-collapse ${side}`}
            onClick={() => togglePanel(side)}
            title={`collapse ${label} (alt+${side === "left" ? "1" : "2"})`}
            aria-label={`collapse ${label}`}
          >
            {side === "left" ? "‹" : "›"}
          </button>
          <div
            className="panel-resize"
            role="separator"
            tabIndex={0}
            aria-label={`${label} panel width`}
            aria-orientation="vertical"
            aria-valuenow={width}
            aria-valuemin={PANEL_MIN}
            aria-valuemax={PANEL_MAX}
            aria-valuetext={`${width} pixels`}
            title={`drag or use arrow keys to resize · release below ${PANEL_COLLAPSE_AT}px to collapse · double-click to collapse`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={() => endDrag(true)}
            onPointerCancel={() => endDrag(false)}
            onLostPointerCapture={() => endDrag(false)}
            onDoubleClick={() => togglePanel(side)}
            onKeyDown={(event) => {
              const movement = event.key === "ArrowLeft" ? -16
                : event.key === "ArrowRight" ? 16
                  : 0;
              if (!movement) return;
              event.preventDefault();
              event.stopPropagation();
              // Move the separator in the direction of the key. The right
              // panel grows when its inner edge moves left, hence the reversal.
              setPanelWidth(side, width + (side === "left" ? movement : -movement));
            }}
          />
        </>
      )}
    </div>
  );
}
