import type { PlacedGraphNode } from "./graph-scene";
import type { Rect } from "./graph-geometry";

export const OVERVIEW_SCALE = 0.75;
/**
 * The lowest view scale overview labels are counter-scaled for. Below it a
 * label keeps the world size it had there and shrinks with the graph, so it
 * never grows past a fixed proportion of the nodes it names.
 */
export const LABEL_FLOOR_SCALE = 0.2;
/** Text smaller than this on screen is not drawn: it reads as a smudge. */
export const MIN_LEGIBLE_PX = 5;
/** Edge operand labels (`arg0`, `arg1`) are sized in the world, like the
 * connectors they annotate, and never counter-scaled. */
export const OPERAND_FONT_PX = 9;
const LINE_PX = 12;
const FONT_PX = 10;
/** Screen-px budget per character, generous enough to cover the widest glyphs
 * at the 10px counter-scaled size both label kinds render at. */
const CHAR_PX = 10;
const MAX_LABEL_PX = 180;

/** The scale an overview label is counter-scaled by at view scale `scale`. */
export const overviewTextScale = (scale: number): number => Math.max(scale, LABEL_FLOOR_SCALE);

/** Whether edge operand labels are large enough on screen to draw. */
export const operandLabelsLegible = (scale: number): boolean =>
  OPERAND_FONT_PX * scale >= MIN_LEGIBLE_PX;

/** Where an operation's label sits, in CSS px: its width, and its offset from
 * the node's own top edge. Unlike a tensor name, it may be wider and taller
 * than the box it belongs to, so it carries a position rather than a width. */
export type OpLabelPlacement = { w: number; dy: number };

export type OverviewLabels = {
  /** CSS-px width for an enlarged tensor name, keyed by tensor id. */
  tensors: Map<string, number>;
  /** Placement for an enlarged operation label, keyed by node id. */
  ops: Map<string, OpLabelPlacement>;
};

/**
 * Counter-scaled labels for a zoomed-out view, placed so none overlaps a node
 * or another label. Presentation only, with no Dagre relayout.
 *
 * Tensor names borrow the free space above their own header and never exceed
 * their card. Operation labels have no such space: a node box is sized for its
 * label at scale 1, so a label counter-scaled to stay legible at 20% needs
 * roughly five times the width the box reserves. They are therefore allowed to
 * extend past their box into the gaps between ranks, and are dropped when
 * nothing free is within reach - the same trade tensor names already make.
 *
 * Tensors are placed first, because a card is what the reader inspects and an
 * operation label that displaces one would cost more than it gives.
 *
 * Below `LABEL_FLOOR_SCALE` every screen-px budget shrinks with the text, and
 * once the text is below `MIN_LEGIBLE_PX` no label is placed at all.
 */
export function overviewLabels(
  nodes: PlacedGraphNode[],
  scale: number,
  names: Record<string, string>
): OverviewLabels {
  const tensors = new Map<string, number>();
  const ops = new Map<string, OpLabelPlacement>();
  if (scale >= OVERVIEW_SCALE) return { tensors, ops };
  const shrink = scale / overviewTextScale(scale);
  if (FONT_PX * shrink < MIN_LEGIBLE_PX) return { tensors, ops };
  const line = LINE_PX * shrink;
  const char = CHAR_PX * shrink;
  const maxLabel = MAX_LABEL_PX * shrink;
  const occupied: Rect[] = [];
  const intersects = (a: Rect, b: Rect) =>
    a.x < b.x + b.w + 2 && a.x + a.w + 2 > b.x &&
    a.y < b.y + b.h + 2 && a.y + a.h + 2 > b.y;
  const projected = nodes.map((node) => ({
    x: node.x * scale, y: node.y * scale, w: node.w * scale, h: node.h * scale,
  }));
  const free = (label: Rect, index: number) =>
    !projected.some((other, i) => i !== index && intersects(label, other)) &&
    !occupied.some((other) => intersects(label, other));

  nodes.forEach((node, index) => {
    if (node.kind !== "tensor") return;
    const card = projected[index];
    const w = Math.min(maxLabel, card.w, Math.max(line, (names[node.id]?.length ?? 1) * char + 2));
    if (w < FONT_PX * shrink) return;
    const label = { x: card.x + (card.w - w) / 2, y: card.y + 17 * scale - line, w, h: line };
    if (!free(label, index)) return;
    occupied.push(label);
    tensors.set(node.id, w / scale);
  });

  nodes.forEach((node, index) => {
    if (node.kind === "tensor") return;
    const text = names[node.id];
    if (!text) return;
    const box = projected[index];
    const w = Math.min(maxLabel, Math.max(line, text.length * char + 2));
    const x = box.x + (box.w - w) / 2;
    // Across the node first, so a label stays on the thing it names; the gaps
    // above and below are the fallback when a neighbour is in the way.
    for (const y of [
      box.y + (box.h - line) / 2,
      box.y - line - 2,
      box.y + box.h + 2,
    ]) {
      const label = { x, y, w, h: line };
      if (!free(label, index)) continue;
      occupied.push(label);
      ops.set(node.id, { w: w / scale, dy: (y - box.y) / scale });
      return;
    }
  });

  return { tensors, ops };
}
