import { cardPx, MAX_GRAPH_H, MAX_GRAPH_W, planeExtents } from "./tiling";
import { remapped, viewAxes, type ViewCfg } from "./tensor-view";

/**
 * Px per element for one card.
 *
 * Every card draws at the graph's scale, so a dimension two tensors share has
 * one length everywhere. A card drawing a chosen pair of axes is the one
 * exception: `[B, S, H, D]` shown as `S x D` can be many times the plane the
 * graph was scaled for, so it shrinks to the size budget instead of growing
 * past it. Its grid is then denser than its neighbours', which is the true
 * reading - its scale differs - and the card says so.
 */
export function cardScaleFor(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes"> | undefined,
  graphPx: number
): number {
  if (!remapped(shape, cfg)) return graphPx;
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const { rows, cols } = planeExtents([...shape], rowAxis, colAxis);
  return Math.min(graphPx, MAX_GRAPH_W / Math.max(1, cols), MAX_GRAPH_H / Math.max(1, rows));
}

/** Layout footprint of a tensor card at the graph's scale `px`.
 * Independent of React so a Worker can perform structural layout. */
export function cardSize(
  shape: number[],
  px: number,
  name = "",
  alternateShapeLabels: string[] = [],
  cfg?: Pick<ViewCfg, "axes">,
  axisNames: readonly (string | undefined)[] = []
): { w: number; h: number } {
  const rank = shape.length;
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  const canvas = cardPx(rows, cols, cardScaleFor(shape, cfg, px));
  const hiddenAxes = Math.max(0, rank - (rowAxis >= 0 ? 1 : 0) - (colAxis >= 0 ? 1 : 0));
  const h = 24 + (rank > 2 || remapped(shape, cfg) ? 24 : 0) + hiddenAxes * 20 + canvas.h;
  const numericShapeLabel = `[${shape.join(" × ")}]`;
  const widestShapeLabel = [numericShapeLabel, ...alternateShapeLabels]
    .reduce((longest, label) => label.length > longest.length ? label : longest);
  const widestTileLabel = `⊞ ${rows}×${cols}`;
  // A chosen pair is printed beside the proj/slice toggle, which has room to
  // spare; only a long pair of names needs the card wider.
  const name_ = (axis: number) => axisNames[axis] ?? `ax${axis}`;
  const planeLabelW = remapped(shape, cfg)
    ? (`rows ${name_(rowAxis)} · cols ${name_(colAxis)} · scale ÷00`.length) * 6.5 + 60
    : 0;
  const labelW = name.length * 9 +
    (widestShapeLabel.length + widestTileLabel.length) * 6.5 + 22;
  return { w: Math.max(canvas.w, labelW, planeLabelW, 120), h };
}
