/** The Dependencies panel shown with nothing drawn. */

import React, { useMemo } from "react";
import { fromBox } from "../../core/region";
import { startingTiles } from "../../view/tensor/seeds";
import { useStore } from "../../state/store";

/** Nothing drawn yet: teach the three relations instead of rendering empty tables. */
export function EmptyPanel(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const tileScale = useStore((s) => s.tileScale);
  const viewCfgs = useStore((s) => s.viewCfgs);
  const graphPx = useStore((s) => s.graphPx);
  const setSelection = useStore((s) => s.setSelection);
  const starts = useMemo(
    () => startingTiles(resolved, tileScale, viewCfgs),
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
