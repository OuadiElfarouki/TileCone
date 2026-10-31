import React from "react";
import { useStore } from "../../state/store";
import { canvasHint, InspectorTab } from "../../view/workspace";
import { CANVAS_SLOT_ID } from "./OnCanvas";

export const VIEWS = [
  {
    id: "dependencies",
    label: "Dependencies",
    hint: "what the drawn tiles need, feed and share: exact, or bounded with its reason named",
  },
  /* A second class, and a second promise. A plan figure rests on a declared
     tiling rather than on the drawn region, and it is exact for that tiling or
     bounded with its reason. Fusion and materialization are choices the plan
     states, so nothing here is modelled behind the reader's back. */
  {
    id: "plan",
    label: "Plan",
    hint: "how a declared tiling divides the work: exact for that tiling, or bounded",
  },
] as const satisfies readonly { id: InspectorTab; label: string; hint: string }[];

/**
 * The Dependencies / Plan switch, on the canvas it changes.
 *
 * The two views give one gesture two meanings: a drag draws a tile in one and
 * divides a tensor into tasks in the other, and the canvas paints different
 * things for each. So the switch sits where the gesture is made, beside a line
 * saying what a press does now, rather than in the inspector, which may be
 * collapsed while the canvas never is.
 *
 * Plain buttons rather than the ARIA tab pattern: tabs would promise roving
 * focus and arrow navigation, while arrows already move tiles and tasks. Both
 * buttons remain ordinary tab stops and expose their state through
 * `aria-pressed`.
 */
export function ViewModeBar(): React.ReactElement {
  const view = useStore((s) => s.inspectorTab);
  const setView = useStore((s) => s.setInspectorTab);
  const hint = useStore((s) =>
    canvasHint({
      inspectorTab: s.inspectorTab,
      canvasTool: s.canvasTool,
      planTiled: Object.keys(s.planTiles).length > 0,
    })
  );
  return (
    <div className="view-mode-bar">
      <div className="view-mode-switch" role="group" aria-label="view (P)">
        {VIEWS.map(({ id, label, hint: title }) => (
          <button
            key={id}
            id={`view-${id}`}
            aria-pressed={view === id}
            aria-controls={`ins-panel-${id}`}
            title={`${title} (P switches)`}
            onClick={() => setView(id)}
          >
            {label}
          </button>
        ))}
      </div>
      <span className="view-mode-hint" role="status">
        {hint}
      </span>
      {/* Controls for what plays on the cards, placed here by their owner. */}
      <div id={CANVAS_SLOT_ID} className="canvas-controls-slot" />
    </div>
  );
}
