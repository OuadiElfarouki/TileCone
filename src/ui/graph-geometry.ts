export type Point = { x: number; y: number };
export type Rect = Point & { w: number; h: number };

/** Air between every solid graph node. Wide enough for the 15px grab handle to
 * remain usable without sitting over a neighbouring node. */
export const NODE_GAP = 18;
export const WORLD_MARGIN = 20;

export function rectsOverlap(a: Rect, b: Rect, gap = NODE_GAP): boolean {
  return (
    a.x < b.x + b.w + gap &&
    a.x + a.w + gap > b.x &&
    a.y < b.y + b.h + gap &&
    a.y + a.h + gap > b.y
  );
}

function verticalConflict(a: Rect, b: Rect, gap: number): boolean {
  return a.y < b.y + b.h + gap && a.y + a.h + gap > b.y;
}

function horizontalConflict(a: Rect, b: Rect, gap: number): boolean {
  return a.x < b.x + b.w + gap && a.x + a.w + gap > b.x;
}

function moveX(rect: Rect, requested: number, blockers: Rect[], gap: number): Rect {
  let dx = requested;
  for (const block of blockers) {
    if (!verticalConflict(rect, block, gap)) continue;
    if (requested > 0) {
      const limit = block.x - gap - (rect.x + rect.w);
      if (limit >= 0 && limit < dx) dx = limit;
    } else if (requested < 0) {
      const limit = block.x + block.w + gap - rect.x;
      if (limit <= 0 && limit > dx) dx = limit;
    }
  }
  return { ...rect, x: rect.x + dx };
}

function moveY(rect: Rect, requested: number, blockers: Rect[], gap: number): Rect {
  let dy = requested;
  for (const block of blockers) {
    if (!horizontalConflict(rect, block, gap)) continue;
    if (requested > 0) {
      const limit = block.y - gap - (rect.y + rect.h);
      if (limit >= 0 && limit < dy) dy = limit;
    } else if (requested < 0) {
      const limit = block.y + block.h + gap - rect.y;
      if (limit <= 0 && limit > dy) dy = limit;
    }
  }
  return { ...rect, y: rect.y + dy };
}

/**
 * Move one solid rectangle without ever crossing another solid rectangle.
 *
 * Each axis is swept over the whole requested distance, so a sparse stream of
 * pointer events cannot tunnel through a node. Resolving the dominant axis
 * first gives the other axis a chance to slide along the boundary instead of
 * making diagonal collisions feel sticky.
 */
export function constrainRectMotion(
  rect: Rect,
  delta: Point,
  blockers: Rect[],
  gap = NODE_GAP,
  min: Point = { x: WORLD_MARGIN, y: WORLD_MARGIN }
): Rect {
  const dx = Math.max(delta.x, min.x - rect.x);
  const dy = Math.max(delta.y, min.y - rect.y);
  let moved = rect;
  if (Math.abs(dx) >= Math.abs(dy)) {
    moved = moveX(moved, dx, blockers, gap);
    moved = moveY(moved, dy, blockers, gap);
  } else {
    moved = moveY(moved, dy, blockers, gap);
    moved = moveX(moved, dx, blockers, gap);
  }
  // A valid graph starts separated. This guard makes the invariant fail safe
  // if a future layout policy supplies an already-overlapping blocker set.
  return blockers.some((block) => rectsOverlap(moved, block, gap)) ? rect : moved;
}

const n = (value: number) => Math.round(value * 10) / 10;

/** Cubic connector anchored on the facing sides of two solid graph nodes. */
export function curvedEdgePath(from: Rect, to: Rect): string {
  const fc = { x: from.x + from.w / 2, y: from.y + from.h / 2 };
  const tc = { x: to.x + to.w / 2, y: to.y + to.h / 2 };
  const dx = tc.x - fc.x;
  const dy = tc.y - fc.y;

  if (Math.abs(dx) >= Math.abs(dy) * 0.55) {
    const sign = dx >= 0 ? 1 : -1;
    const start = { x: sign > 0 ? from.x + from.w : from.x, y: fc.y };
    const end = { x: sign > 0 ? to.x : to.x + to.w, y: tc.y };
    const bend = Math.max(32, Math.abs(end.x - start.x) * 0.42);
    return `M${n(start.x)},${n(start.y)} C${n(start.x + sign * bend)},${n(start.y)} ${n(end.x - sign * bend)},${n(end.y)} ${n(end.x)},${n(end.y)}`;
  }

  const sign = dy >= 0 ? 1 : -1;
  const start = { x: fc.x, y: sign > 0 ? from.y + from.h : from.y };
  const end = { x: tc.x, y: sign > 0 ? to.y : to.y + to.h };
  const bend = Math.max(32, Math.abs(end.y - start.y) * 0.42);
  return `M${n(start.x)},${n(start.y)} C${n(start.x)},${n(start.y + sign * bend)} ${n(end.x)},${n(end.y - sign * bend)} ${n(end.x)},${n(end.y)}`;
}
