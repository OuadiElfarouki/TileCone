import React, { useMemo } from "react";
import { hasSymbolicShape } from "./shape-label";
import { planesOf, useStore } from "./store";
import {
  effectiveTileScaleIndex,
  effectiveTileScaleStops,
  settledTiles,
  TILE_SCALE_NONE,
} from "./tiling";

/** What the graph actually settled on, which the request only asks for. */
function settledLabel(min: number, max: number): string {
  return min === max ? `${min} × ${min}` : `${min} × ${min} – ${max} × ${max}`;
}

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
  const graphPx = useStore((s) => s.graphPx);
  const tileScale = useStore((s) => s.tileScale);
  const setTileScale = useStore((s) => s.setTileScale);
  const snapToGrid = useStore((s) => s.snapToGrid);
  const setSnapToGrid = useStore((s) => s.setSnapToGrid);
  const axisMode = useStore((s) => s.axisMode);
  const setAxisMode = useStore((s) => s.setAxisMode);

  const detail = useMemo(() => {
    if (!resolved) return null;
    const planes = planesOf(resolved);
    const stops = effectiveTileScaleStops(planes, graphPx);
    const index = effectiveTileScaleIndex(planes, graphPx, stops, tileScale);
    const settled = stops.map((scale) => settledTiles(planes, scale, graphPx));
    return {
      stops,
      index,
      settled: settled[index],
      labels: stops.map((scale, i) =>
        scale === TILE_SCALE_NONE
          ? `none → ${settledLabel(settled[i].min, settled[i].max)}`
          : settledLabel(settled[i].min, settled[i].max)
      ),
    };
  }, [resolved, graphPx, tileScale]);
  if (!resolved || !detail) return null;

  const { min, max } = detail.settled;
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
            : min === max
              ? "the tile every tensor settles on"
              : "tiles differ per tensor: the fit rule coarsens the largest ones"
        }
      >
        {settledLabel(min, max)}
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
        className={`mini toggle${snapToGrid ? " on" : ""}`}
        aria-pressed={snapToGrid}
        onClick={() => setSnapToGrid(!snapToGrid)}
        title="snap a drawn box out to whole tiles; off cuts an exact element range"
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
