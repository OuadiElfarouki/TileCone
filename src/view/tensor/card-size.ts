import { cardPx, MAX_GRAPH_H, MAX_GRAPH_W, planeExtents } from "./tiling";
import { hiddenAxes, hiddenMode, projectedTileAxes, remapped, UNFOLD_MARK, viewAxes, type ViewCfg } from "./tensor-view";
import { foldGeometry, foldScale } from "./fold";
import { DECK_ROOM_PX } from "./deck";
import type { Tensor } from "../../core/graph";
import { shapeLabel, symbolicExtentLabel } from "./shape-label";

/** What of a tensor's view changes the size of its card. An absent
 * `projection` is the default, projected. */
export type CardShapeView = Partial<Pick<ViewCfg, "axes" | "projection" | "unfold" | "tile" | "scale">> & {
  /** Only the presence of projected tile controls affects placement. */
  tilePositions?: number[];
};

/**
 * The part of a view that sizes its card, or null when the card has the
 * default footprint: projected, on the default plane. Graph layout needs only
 * these, so changing a position or tile extent never re-runs placement unless
 * it adds or removes a projected tile control.
 */
export function cardShapeView(shape: readonly number[], cfg: CardShapeView | undefined): CardShapeView | null {
  if (!cfg) return null;
  const mode = shape.length > 2 ? hiddenMode(cfg) : "project";
  const axes = remapped(shape, cfg) ? cfg.axes : undefined;
  const tilePositions = cfg.tilePositions ?? projectedTileAxes(shape, cfg);
  if (mode === "project" && !axes && !tilePositions.length) return null;
  return {
    projection: mode === "project",
    ...(mode === "unfold" ? { unfold: cfg.unfold } : {}),
    ...(axes ? { axes: [axes[0], axes[1]] as [number, number] } : {}),
    ...(tilePositions.length ? { tilePositions: tilePositions.slice() } : {}),
  };
}

/**
 * The card's view line: how it reads the axes it does not draw, and which
 * pair it draws when that is not the default. Each is null when there is
 * nothing to say. Shared by the card and its measurement, so the line never
 * outgrows the width reserved for it.
 */
export function viewTags(
  shape: readonly number[],
  cfg: CardShapeView | undefined,
  nameOf: (axis: number) => string
): { hidden: string | null; plane: string | null } {
  const names = hiddenAxes(shape, cfg).map(nameOf).join("·");
  const mode = hiddenMode(cfg);
  const hidden = shape.length <= 2 || !names
    ? null
    : mode === "unfold"
      ? `unfold ${names} ${UNFOLD_MARK[cfg!.unfold!]}`
      : mode === "slice" ? `slice ${names}` : `∪ ${names}`;
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const plane = remapped(shape, cfg) ? `rows ${nameOf(rowAxis)} · cols ${nameOf(colAxis)}` : null;
  return { hidden, plane };
}

/**
 * Px per element for one card.
 *
 * Every card draws at the graph's scale, so a dimension two tensors share has
 * one length everywhere. Two kinds of card are the exception. One drawing a
 * chosen pair of axes - `[B, S, H, D]` shown as `S x D` can be many times the
 * plane the graph was scaled for - and an unfolded one, which draws a plane per
 * hidden index. Both shrink to a size budget. Manual resizing multiplies that
 * fitted scale. A card off the graph's scale states its ratio in its header.
 */
export function cardScaleFor(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes" | "unfold" | "scale"> | undefined,
  graphPx: number
): number {
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const { rows, cols } = planeExtents([...shape], rowAxis, colAxis);
  const plane = remapped(shape, cfg)
    ? Math.min(graphPx, MAX_GRAPH_W / Math.max(1, cols), MAX_GRAPH_H / Math.max(1, rows))
    : graphPx;
  return (cfg?.unfold ? foldScale(shape, cfg, rows, cols, plane) : plane) * (cfg?.scale ?? 1);
}

/** `×1.5`, `÷4`: how a card's scale compares with the graph's, to three
 *  significant figures, or null where it draws at the graph's scale. */
export function scaleRatioLabel(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes" | "unfold" | "scale"> | undefined,
  graphPx: number
): string | null {
  const ratio = cardScaleFor(shape, cfg, graphPx) / graphPx;
  const shown = Number((ratio >= 1 ? ratio : 1 / ratio).toPrecision(3));
  return shown === 1 ? null : `${ratio >= 1 ? "×" : "÷"}${shown}`;
}

/** The header's scale note at its longest, for the width the card reserves. */
const LONGEST_SCALE_NOTE = "scale ÷0000";

/** The canvas a card draws, in card px: the plane, or every panel of an
 *  unfolded card with the gaps between them. */
export function canvasSize(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes" | "unfold" | "scale"> | undefined,
  px: number
): { w: number; h: number } {
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const { rows, cols } = planeExtents([...shape], rowAxis, colAxis);
  const scale = cardScaleFor(shape, cfg, px);
  const fold = foldGeometry(shape, cfg, rows, cols, scale);
  return fold ? { w: fold.across.total, h: fold.down.total } : cardPx(rows, cols, scale);
}

/** Layout footprint of a tensor card at the graph's scale `px`.
 * Independent of React so a Worker can perform structural layout. */
export function cardSize(
  shape: number[],
  px: number,
  name = "",
  alternateShapeLabels: string[] = [],
  cfg?: CardShapeView,
  axisNames: readonly (string | undefined)[] = []
): { w: number; h: number } {
  const rank = shape.length;
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
  const canvas = canvasSize(shape, cfg, px);
  const hiddenCount = hiddenAxes(shape, cfg).length;
  // Slice sliders and projected tile-position controls each need one row.
  // The deck keeps room above and beside the canvas whichever slice is shown.
  const slicing = hiddenCount > 0 && hiddenMode(cfg) === "slice";
  const positions = cfg?.tilePositions ?? projectedTileAxes(shape, cfg);
  const deck = slicing ? DECK_ROOM_PX : 0;
  const h = 24 + (rank > 2 || remapped(shape, cfg) ? 24 : 0) + (slicing ? hiddenCount : positions.length) * 20 + canvas.h + deck;
  const numericShapeLabel = `[${shape.join(" × ")}]`;
  const widestShapeLabel = [numericShapeLabel, ...alternateShapeLabels]
    .reduce((longest, label) => label.length > longest.length ? label : longest);
  const widestTileLabel = `⊞ ${rows}×${cols}`;
  // The view line is short beside a plane of any size; only long axis names
  // need the card wider.
  const tags = viewTags(shape, cfg, (axis) => axisNames[axis] ?? `ax${axis}`);
  const viewLine = [tags.hidden, tags.plane].filter(Boolean).join(" · ");
  const viewLineW = viewLine ? viewLine.length * 6.5 + 20 : 0;
  // A card that can draw off the graph's scale reserves its longest scale note,
  // so the card's width does not follow the ratio. A scale of exactly 1 is the
  // fitted card that placement measured.
  const offScale = remapped(shape, cfg) || !!cfg?.unfold || (cfg?.scale !== undefined && cfg.scale !== 1);
  const scaleNoteW = offScale ? LONGEST_SCALE_NOTE.length * 6.5 + 6 : 0;
  const labelW = name.length * 9 +
    (widestShapeLabel.length + widestTileLabel.length) * 6.5 + 22 + scaleNoteW;
  return { w: Math.max(canvas.w + deck, labelW, viewLineW, 120), h };
}

/** A graph tensor's card, wide enough for any of its shape labels. */
export function tensorCardSize(tensor: Tensor, px: number, cfg?: CardShapeView): { w: number; h: number } {
  return cardSize(tensor.resolved!, px, tensor.name,
    [shapeLabel(tensor, "symbolic"), symbolicExtentLabel(tensor)], cfg, tensor.axisNames);
}
