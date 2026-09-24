import { cardPx, planeExtents } from "./tiling";
import { viewAxes } from "./tensor-view";

/** Layout footprint of a tensor card at the graph's scale `px`.
 * Independent of React so a Worker can perform structural layout. */
export function cardSize(
  shape: number[],
  px: number,
  name = "",
  alternateShapeLabels: string[] = []
): { w: number; h: number } {
  const rank = shape.length;
  const { rowAxis, colAxis } = viewAxes(shape);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  const canvas = cardPx(rows, cols, px);
  const hiddenAxes = Math.max(0, rank - (rowAxis >= 0 ? 1 : 0) - (colAxis >= 0 ? 1 : 0));
  const h = 24 + (rank > 2 ? 24 : 0) + hiddenAxes * 20 + canvas.h;
  const numericShapeLabel = `[${shape.join(" × ")}]`;
  const widestShapeLabel = [numericShapeLabel, ...alternateShapeLabels]
    .reduce((longest, label) => label.length > longest.length ? label : longest);
  const widestTileLabel = `⊞ ${rows}×${cols}`;
  const labelW = name.length * 9 +
    (widestShapeLabel.length + widestTileLabel.length) * 6.5 + 22;
  return { w: Math.max(canvas.w, labelW, 120), h };
}
