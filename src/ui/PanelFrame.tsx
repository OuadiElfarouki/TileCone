import React, { useRef } from "react";
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
 * The strip always sits on the panel's *inner* edge — the one facing the canvas
 * — so the gesture reads as pushing the canvas boundary rather than dragging the
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
  const dragRef = useRef(false);
  const rawWidthRef = useRef(width);
  const startWidthRef = useRef(width);

  const onPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = true;
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
    // Width previews stay open and usable. Crossing the collapse threshold is
    // committed only on pointerup, so the captured element cannot disappear
    // before it has a chance to end the drag.
    setPanelWidth(side, raw);
  };
  const endDrag = (commit: boolean) => {
    if (!dragRef.current) return;
    dragRef.current = false;
    if (commit) finishPanelResize(side, rawWidthRef.current);
    else setPanelWidth(side, startWidthRef.current);
    setDragging(false);
  };

  return (
    <div
      ref={frameRef}
      className={`panel-frame ${side}${collapsed ? " collapsed" : ""}`}
      style={{ width: collapsed ? PANEL_RAIL : width }}
    >
      {/* Keep children mounted while collapsed. The source editor owns its draft
          locally, so unmounting it here would silently discard unrun work. */}
      <div className="panel-content" aria-hidden={collapsed || undefined}>
        {children}
      </div>
      {collapsed ? (
        <button
          className={`panel-rail ${side}`}
          onClick={() => togglePanel(side)}
          title={`show ${label} (alt+${side === "left" ? "1" : "2"})`}
        >
          <span>{label}</span>
        </button>
      ) : (
        <>
          <button
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
            aria-orientation="vertical"
            aria-valuenow={width}
            aria-valuemin={PANEL_MIN}
            aria-valuemax={PANEL_MAX}
            title={`drag to resize · release below ${PANEL_COLLAPSE_AT}px to collapse`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={() => endDrag(true)}
            onPointerCancel={() => endDrag(false)}
            onLostPointerCapture={() => endDrag(false)}
            onDoubleClick={() => togglePanel(side)}
          />
        </>
      )}
    </div>
  );
}
