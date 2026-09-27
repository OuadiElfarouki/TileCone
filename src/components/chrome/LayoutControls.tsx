import React from "react";
import { useStore } from "../../state/store";

/**
 * What a drag is allowed to move.
 *
 * Cards move by default: they are the subject, and reading a cone means putting
 * two of them where they can be compared. Operation nodes are the scaffolding
 * dagre put between them, and their generated rank is what makes a chain
 * legible, so they stay pinned until someone asks otherwise - and then the whole
 * scene is furniture, for arranging a graph to be looked at or shown.
 *
 * It rides the canvas beside the zoom controls for the same reason the grid
 * settings do: the canvas is what it acts on, and unlike either side panel the
 * canvas never collapses out from under a live gesture.
 */
export function LayoutControls(): React.ReactElement {
  const moveOps = useStore((s) => s.moveOps);
  const setMoveOps = useStore((s) => s.setMoveOps);

  return (
    <div className="layout-controls">
      <span className="setup-kicker">move</span>
      <button
        className={`mini toggle${moveOps ? " on" : ""}`}
        aria-pressed={moveOps}
        onClick={() => setMoveOps(!moveOps)}
        title="drag operation nodes as well as tensor cards; off locks operations in place"
      >
        ops
      </button>
    </div>
  );
}
