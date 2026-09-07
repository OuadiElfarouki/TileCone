import type { PlacedGraphNode } from "./graph-scene";
import type { Rect } from "./graph-geometry";

export const OVERVIEW_SCALE = 0.75;
const LINE_PX = 12;
/** Screen-px budget per character, generous enough to cover the widest glyphs
 * at the 10px counter-scaled size both label kinds render at. */
const CHAR_PX = 10;
const MAX_LABEL_PX = 180;

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
 */
export function overviewLabels(
  nodes: PlacedGraphNode[],
  scale: number,
  names: Record<string, string>
): OverviewLabels {
  const tensors = new Map<string, number>();
  const ops = new Map<string, OpLabelPlacement>();
  if (scale >= OVERVIEW_SCALE) return { tensors, ops };
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
    const w = Math.min(MAX_LABEL_PX, card.w, Math.max(12, (names[node.id]?.length ?? 1) * CHAR_PX + 2));
    if (w < 10) return;
    const label = { x: card.x + (card.w - w) / 2, y: card.y + 17 * scale - LINE_PX, w, h: LINE_PX };
    if (!free(label, index)) return;
    occupied.push(label);
    tensors.set(node.id, w / scale);
  });

  nodes.forEach((node, index) => {
    if (node.kind === "tensor") return;
    const text = names[node.id];
    if (!text) return;
    const box = projected[index];
    const w = Math.min(MAX_LABEL_PX, Math.max(12, text.length * CHAR_PX + 2));
    const x = box.x + (box.w - w) / 2;
    // Across the node first, so a label stays on the thing it names; the gaps
    // above and below are the fallback when a neighbour is in the way.
    for (const y of [
      box.y + (box.h - LINE_PX) / 2,
      box.y - LINE_PX - 2,
      box.y + box.h + 2,
    ]) {
      const label = { x, y, w, h: LINE_PX };
      if (!free(label, index)) continue;
      occupied.push(label);
      ops.set(node.id, { w: w / scale, dy: (y - box.y) / scale });
      return;
    }
  });

  return { tensors, ops };
}
