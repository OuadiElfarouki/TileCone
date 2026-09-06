import type { PlacedGraphNode } from "./graph-scene";
import type { Rect } from "./graph-geometry";

export const OVERVIEW_SCALE = 0.75;
const LINE_PX = 12;

/** Names borrow the free space above their own header, never another node's
 * footprint. Widths are capped and ellipsized; unavailable space falls back to
 * the ordinary small name. This is presentation only, with no Dagre relayout. */
export function overviewLabelWidths(
  nodes: PlacedGraphNode[],
  scale: number,
  names: Record<string, string>
): Map<string, number> {
  const widths = new Map<string, number>();
  if (scale >= OVERVIEW_SCALE) return widths;
  const occupied: Rect[] = [];
  const intersects = (a: Rect, b: Rect) =>
    a.x < b.x + b.w + 2 && a.x + a.w + 2 > b.x &&
    a.y < b.y + b.h + 2 && a.y + a.h + 2 > b.y;
  const projected = nodes.map((node) => ({
    x: node.x * scale, y: node.y * scale, w: node.w * scale, h: node.h * scale,
  }));
  nodes.forEach((node, index) => {
    if (node.kind !== "tensor") return;
    const card = projected[index];
    const w = Math.min(180, card.w, Math.max(12, (names[node.id]?.length ?? 1) * 10 + 2));
    if (w < 10) return;
    const label = { x: card.x + (card.w - w) / 2, y: card.y + 17 * scale - LINE_PX, w, h: LINE_PX };
    if (projected.some((other, i) => i !== index && intersects(label, other)) ||
        occupied.some((other) => intersects(label, other))) return;
    occupied.push(label);
    widths.set(node.id, w / scale);
  });
  return widths;
}
