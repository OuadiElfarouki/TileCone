import React from "react";
import { useStore } from "../../state/store";

/**
 * The layout group: what a drag is allowed to move, and the way back.
 *
 * Cards move by default: they are the subject, and reading a cone means putting
 * two of them where they can be compared. Operation nodes are the scaffolding
 * dagre put between them, and their generated rank is what makes a chain
 * legible, so they stay pinned until someone asks otherwise - and then the whole
 * scene is furniture, for arranging a graph to be looked at or shown.
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
  const moveOps = useStore((s) => s.moveOps);
  const setMoveOps = useStore((s) => s.setMoveOps);
  const moved = useStore((s) => Object.keys(s.nodeOffsets).length > 0);

  return (
    <div className="layout-controls">
      <span className="setup-kicker">layout</span>
      <button
        className={`mini toggle${moveOps ? " on" : ""}`}
        aria-pressed={moveOps}
        onClick={() => setMoveOps(!moveOps)}
        title="drag operation nodes as well as tensor cards; off locks operations in place"
      >
        move ops
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
