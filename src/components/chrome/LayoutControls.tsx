import React from "react";
import { useStore } from "../../state/store";

/**
 * The layout group: the tool that moves nodes, and the way back.
 *
 * The select tool is the default because reading is: a press on a card draws a
 * tile and a press on an operation picks it, and the generated rank is what
 * makes a chain legible. The move tool turns every node, card or operation, into
 * furniture dragged from anywhere on it, for arranging a graph to be looked at
 * or shown. Holding Ctrl/Cmd borrows it for one drag without switching, so a
 * single card can be moved aside in the middle of reading.
 *
 * `reset` belongs here rather than beside the zoom buttons: it restores
 * generated placement, which is this group's subject, while zoom and fit change
 * only where the viewport is pointed and leave every node where it was. The
 * group rides the canvas for the same reason the grid settings do - the canvas
 * is what it acts on, and unlike either side panel it never collapses out from
 * under a live gesture.
 *
 * Resetting also re-fits the view, which is viewport state the canvas owns, so
 * the action arrives as a callback rather than being read from the store here.
 */
export function LayoutControls({
  onResetLayout,
}: {
  onResetLayout: () => void;
}): React.ReactElement {
  const moveTool = useStore((s) => s.canvasTool === "move");
  const setCanvasTool = useStore((s) => s.setCanvasTool);
  const moved = useStore((s) => Object.keys(s.nodeOffsets).length > 0);

  return (
    <div className="layout-controls">
      <span className="setup-kicker">layout</span>
      <button
        className={`mini toggle${moveTool ? " on" : ""}`}
        aria-pressed={moveTool}
        onClick={() => setCanvasTool(moveTool ? "select" : "move")}
        title="drag cards and operations from anywhere on them; Esc or off returns to drawing tiles and selecting operations. Hold Ctrl/Cmd to move one without switching"
      >
        move
      </button>
      <button
        className="mini"
        onClick={onResetLayout}
        disabled={!moved}
        title="restore the generated layout, cards and operations alike (undoable)"
      >
        reset
      </button>
    </div>
  );
}
