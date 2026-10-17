import React, { useMemo } from "react";
import { hasSymbolicShape } from "../../view/tensor/shape-label";
import { useStore } from "../../state/store";
import {
  effectiveTileScaleIndex,
  effectiveTileScaleStops,
  tileScaleLabel,
  TILE_SCALE_NONE,
} from "../../view/tensor/tiling";
import { planesOf } from "../../view/tensor/seeds";

/**
 * The lattice, the snap, and how axes read - the three settings that change the
 * canvas rather than the readout.
 *
 * They sit on the canvas because that is the only thing they act on: the grid
 * a card is ruled with, what a drag is cut to, and how a card names its axes.
 * Both side panels collapse; the canvas does not, and snap in particular has
 * to stay reachable while a tile is being drawn.
 */
export function GridControls(): React.ReactElement | null {
  const resolved = useStore((s) => s.resolved);
  const tileScale = useStore((s) => s.tileScale);
  const setTileScale = useStore((s) => s.setTileScale);
  const snapToGrid = useStore((s) => s.snapToGrid);
  /* A plan tile is a quantity, so a plan drag always snaps. The toggle has
     nothing to act on here and says so instead of appearing to work. */
  const planView = useStore((s) => s.inspectorTab === "plan");
  const setSnapToGrid = useStore((s) => s.setSnapToGrid);
  const axisMode = useStore((s) => s.axisMode);
  const setAxisMode = useStore((s) => s.setAxisMode);

  /* The setting reads as what it asks of every axis - about how many tiles
     cross it - which is one number for the whole graph. The tile each card
     settles on is in the card's own header, since axes of different lengths
     take different tiles. */
  const detail = useMemo(() => {
    if (!resolved) return null;
    const planes = planesOf(resolved);
    const stops = effectiveTileScaleStops(planes);
    const index = effectiveTileScaleIndex(planes, stops, tileScale);
    return { stops, index, labels: stops.map(tileScaleLabel) };
  }, [resolved, tileScale]);
  if (!resolved || !detail) return null;

  const perElement = detail.stops[detail.index] === TILE_SCALE_NONE;
  const hasSemanticLabels = Object.values(resolved.tensors).some(hasSymbolicShape);

  return (
    <div className="grid-controls">
      <span className="setup-kicker">grid</span>
      <span
        className="tile-settled"
        title={
          perElement
            ? "one logical tile per element, on every tensor; boundaries are omitted where they are too dense to draw"
            : "about this many tiles across every axis, each a power of two: an axis of one length is gridded alike on every card; each card's header shows its tile"
        }
      >
        {detail.labels[detail.index]}
      </span>
      <input
        type="range"
        min={0}
        max={Math.max(0, detail.stops.length - 1)}
        step={1}
        value={detail.index}
        onChange={(event) => setTileScale(detail.stops[Number(event.target.value)])}
        aria-label="tile grid detail"
        aria-valuetext={detail.labels[detail.index]}
        title={`tile detail, all tensors - ${detail.labels.join(" · ")}`}
      />
      <button
        className={`mini toggle${snapToGrid && !planView ? " on" : ""}`}
        aria-pressed={planView ? true : snapToGrid}
        disabled={planView}
        onClick={() => setSnapToGrid(!snapToGrid)}
        title={
          planView
            ? "the Plan view always snaps: a tile is a quantity, not a highlight"
            : "snap a drawn box out to whole tiles; off cuts an exact element range"
        }
      >
        snap
      </button>
      <button
        className={`mini toggle${axisMode === "symbolic" ? " on" : ""}`}
        aria-pressed={axisMode === "symbolic"}
        onClick={() => setAxisMode(axisMode === "symbolic" ? "numeric" : "symbolic")}
        disabled={!hasSemanticLabels}
        title="read compact shapes as semantic labels or numeric extents; hover a tensor name for all readings"
      >
        {axisMode === "symbolic" ? "labels" : "extents"}
      </button>
    </div>
  );
}
