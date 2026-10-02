/** The Dependencies / Cost model / Plan switch, and the panel shown with nothing drawn. */

import React, { useMemo } from "react";
import { fromBox } from "../../core/region";
import { startingTiles } from "../../view/tensor/seeds";
import { InspectorTab } from "../../view/workspace";
import { useStore } from "../../state/store";

export const TABS = [
  {
    id: "dependencies",
    label: "Dependencies",
    hint: "what the drawn tiles need, feed and share: exact, or bounded with its reason named",
  },
  {
    id: "execution",
    label: "Cost model",
    hint: "what an assumed execution would do with them: modelled, not bounded",
  },
  /* A third class, and a third promise. A plan figure rests on a declared
     tiling rather than on the drawn region, and it is exact for that tiling or
     bounded with its reason - so it belongs with neither the figures that need
     no plan nor the ones that are modelled. */
  {
    id: "plan",
    label: "Plan",
    hint: "how a declared tiling divides the work: exact for that tiling, or bounded",
  },
] as const satisfies readonly { id: InspectorTab; label: string; hint: string }[];

/**
 * A view switch of plain buttons, deliberately not the ARIA tab pattern. Tabs would
 * promise roving focus and arrow navigation, while arrows already move the
 * tile this panel describes. Both buttons remain ordinary tab stops and expose
 * their current state through `aria-pressed`.
 */
export function InspectorTabs(): React.ReactElement {
  const tab = useStore((s) => s.inspectorTab);
  const setTab = useStore((s) => s.setInspectorTab);
  return (
    <div className="ins-tabs" role="group" aria-label="analysis class">
      {TABS.map(({ id, label, hint }) => (
        <button
          key={id}
          id={`ins-tab-${id}`}
          className="ins-tab"
          aria-pressed={tab === id}
          aria-controls={`ins-panel-${id}`}
          title={hint}
          onClick={() => setTab(id)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

/** Nothing drawn yet: teach the three relations instead of rendering empty tables. */
export function EmptyPanel(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const tileScale = useStore((s) => s.tileScale);
  const viewCfgs = useStore((s) => s.viewCfgs);
  const graphPx = useStore((s) => s.graphPx);
  const setSelection = useStore((s) => s.setSelection);
  const starts = useMemo(
    () => startingTiles(resolved, tileScale, graphPx, viewCfgs),
    [resolved, tileScale, graphPx, viewCfgs]
  );

  return (
    <div className="empty-panel">
      <h2 className="panel-title">No tile drawn</h2>
      <p className="hint">
        Drag a rectangle on any tensor to cut a tile. Shift adds another, Alt subtracts. This
        panel then answers three questions about it.
      </p>
      <dl className="empty-questions">
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ↑
            </span>
            Backward Cone
          </dt>
          <dd>Everything upstream the tile requires.</dd>
        </div>
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ↓
            </span>
            Forward Cone
          </dt>
          <dd>Everything downstream the tile affects.</dd>
        </div>
        <div>
          <dt>
            <span className="cone-arrow" aria-hidden>
              ×
            </span>
            Co-access Surface
          </dt>
          <dd>
            Everything on the same level as the tile that is co-accessed with it.
            Not a hop along the graph; press <kbd>e</kbd> to show it.
          </dd>
        </div>
      </dl>
      {starts.length > 0 && (
        <>
          <div className="setup-kicker">or start from</div>
          <div className="empty-starts">
            {starts.map((start) => (
              <button
                key={start.tensorId}
                className="mini"
                onClick={() => setSelection(start.tensorId, fromBox(start.box), "replace")}
              >
                {resolved.tensors[start.tensorId].name} - {start.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
