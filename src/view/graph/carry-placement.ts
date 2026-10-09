import { NODE_GAP, Rect, rectsOverlap } from "./graph-geometry";
import { nodeKey, PlacedGraphNode } from "./graph-scene";
import type { NodeOffsets } from "./node-layout";

/** Moves a node may take to clear its neighbours before it is sent past them. */
const MAX_NUDGES = 32;

/**
 * Offsets that keep every node of a size-only rebuild where it was on screen.
 *
 * `previous` is where each node was drawn before the rebuild, keyed by scene
 * node key; `base` is the new generated layout. Each node keeps its previous
 * centre, so a card that changed size grows or shrinks about the point the
 * reader left it at rather than jumping to wherever the new layout put it.
 *
 * Nodes that did not grow are placed first and exactly: shrinking about the
 * same centre cannot bring two of them closer than they were. A node that grew
 * may then overlap a neighbour, so it alone is nudged - by the shortest move
 * that clears the neighbour it meets, repeated until it is clear, and past
 * everything to its right if that does not settle. Nodes absent from
 * `previous` stay at their generated position.
 */
export function carriedOffsets(
  previous: ReadonlyMap<string, Rect>,
  base: readonly PlacedGraphNode[]
): NodeOffsets {
  const entries = base.map((node) => {
    const key = nodeKey(node.kind, node.id);
    const before = previous.get(key);
    const rect: Rect = before
      ? {
          x: before.x + (before.w - node.w) / 2,
          y: before.y + (before.h - node.h) / 2,
          w: node.w,
          h: node.h,
        }
      : { x: node.x, y: node.y, w: node.w, h: node.h };
    const grew = !before || node.w > before.w || node.h > before.h;
    return { key, node, rect, grew };
  });

  // Stable nodes first, then grown ones left to right, as the graph reads.
  const order = [
    ...entries.filter((entry) => !entry.grew),
    ...entries.filter((entry) => entry.grew).sort((a, b) => a.rect.x - b.rect.x),
  ];

  const placed: Rect[] = [];
  const offsets: NodeOffsets = {};
  for (const entry of order) {
    const rect = entry.grew ? clear(entry.rect, placed) : entry.rect;
    placed.push(rect);
    const dx = rect.x - entry.node.x;
    const dy = rect.y - entry.node.y;
    if (Math.abs(dx) > 1e-6 || Math.abs(dy) > 1e-6) offsets[entry.key] = { dx, dy };
  }
  return offsets;
}

/** `rect` moved clear of every rect in `placed`, by short moves where possible. */
function clear(start: Rect, placed: readonly Rect[]): Rect {
  let rect = start;
  for (let step = 0; step < MAX_NUDGES; step++) {
    const hit = placed.find((other) => rectsOverlap(rect, other));
    if (!hit) return rect;
    const moves = [
      { x: hit.x + hit.w + NODE_GAP - rect.x, y: 0 },
      { x: hit.x - NODE_GAP - rect.w - rect.x, y: 0 },
      { x: 0, y: hit.y + hit.h + NODE_GAP - rect.y },
      { x: 0, y: hit.y - NODE_GAP - rect.h - rect.y },
    ];
    const move = moves.reduce((best, m) =>
      Math.abs(m.x) + Math.abs(m.y) < Math.abs(best.x) + Math.abs(best.y) ? m : best
    );
    rect = { ...rect, x: rect.x + move.x, y: rect.y + move.y };
  }
  // Short moves can trade one neighbour for another indefinitely. Past the
  // rightmost node there is nothing left to meet.
  if (!placed.some((other) => rectsOverlap(rect, other))) return rect;
  const right = Math.max(...placed.map((other) => other.x + other.w));
  return { ...rect, x: right + NODE_GAP };
}
