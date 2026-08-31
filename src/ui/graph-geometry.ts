export type Point = { x: number; y: number };
export type Rect = Point & { w: number; h: number };

/** Air between every solid graph node. Wide enough for the 15px grab handle to
 * remain usable without sitting over a neighbouring node. */
export const NODE_GAP = 18;
export const WORLD_MARGIN = 20;

/** @internal Pure collision predicate exported for boundary invariant tests. */
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

/** An edge's cubic, in draw order: start, its handle, the end's handle, end. */
export type Cubic = [Point, Point, Point, Point];

/** Widest a fan will spread its anchors, however much room the sides offer. */
export const EDGE_FAN_MAX_PX = 14;

/** True when the connector leaves and arrives on vertical (left/right) sides. */
function edgeIsHorizontal(from: Rect, to: Rect): boolean {
  const dx = to.x + to.w / 2 - (from.x + from.w / 2);
  const dy = to.y + to.h / 2 - (from.y + from.h / 2);
  return Math.abs(dx) >= Math.abs(dy) * 0.55;
}

/** Half the travel an anchor has along its facing side, minus a corner margin. */
function fanRoom(from: Rect, to: Rect): number {
  const side = edgeIsHorizontal(from, to)
    ? Math.min(from.h, to.h)
    : Math.min(from.w, to.w);
  return Math.max(0, side / 2 - 3);
}

/**
 * Anchor spacing for `count` connectors sharing this pair: as wide as the
 * facing sides allow, up to a cap.
 *
 * A fixed spacing has to assume the tightest case — an op node is only 30px
 * tall, so a horizontal bundle of three has barely any room — and then spends
 * that same cramped gap on vertical bundles, where the op node's *width* is the
 * limit and there is several times more of it. Deriving the spacing lets each
 * bundle be as legible as its own geometry permits.
 */
export function fanSpacing(from: Rect, to: Rect, count: number): number {
  if (count <= 1) return 0;
  return Math.min(EDGE_FAN_MAX_PX, (2 * fanRoom(from, to)) / (count - 1));
}

/** Keep a fanned anchor on the side it belongs to, never past its corners. */
function clampLateral(lateral: number, from: number, to: number): number {
  const room = Math.min(from, to) / 2 - 3;
  if (room <= 0) return 0;
  return Math.max(-room, Math.min(lateral, room));
}

/**
 * Cubic connector anchored on the facing sides of two solid graph nodes.
 *
 * `lateral` slides both anchors along their facing side, which is how several
 * connectors between the *same* pair of nodes stay distinguishable. An op may
 * legitimately take one tensor in more than one operand slot — `matmul(X, X)`
 * — and drawing that as a single line claims an arity the op does not have.
 * Fanning is a lateral offset rather than a difference in bend so the lines
 * stay parallel and read as a bundle, in operand order.
 */
export function curvedEdge(from: Rect, to: Rect, lateral = 0): Cubic {
  const fc = { x: from.x + from.w / 2, y: from.y + from.h / 2 };
  const tc = { x: to.x + to.w / 2, y: to.y + to.h / 2 };
  const dx = tc.x - fc.x;
  const dy = tc.y - fc.y;

  if (Math.abs(dx) >= Math.abs(dy) * 0.55) {
    const sign = dx >= 0 ? 1 : -1;
    const off = clampLateral(lateral, from.h, to.h);
    const start = { x: sign > 0 ? from.x + from.w : from.x, y: fc.y + off };
    const end = { x: sign > 0 ? to.x : to.x + to.w, y: tc.y + off };
    const bend = Math.max(32, Math.abs(end.x - start.x) * 0.42);
    return [
      start,
      { x: start.x + sign * bend, y: start.y },
      { x: end.x - sign * bend, y: end.y },
      end,
    ];
  }

  const sign = dy >= 0 ? 1 : -1;
  const off = clampLateral(lateral, from.w, to.w);
  const start = { x: fc.x + off, y: sign > 0 ? from.y + from.h : from.y };
  const end = { x: tc.x + off, y: sign > 0 ? to.y : to.y + to.h };
  const bend = Math.max(32, Math.abs(end.y - start.y) * 0.42);
  return [
    start,
    { x: start.x, y: start.y + sign * bend },
    { x: end.x, y: end.y - sign * bend },
    end,
  ];
}

export function cubicPath([p0, p1, p2, p3]: Cubic): string {
  return `M${n(p0.x)},${n(p0.y)} C${n(p1.x)},${n(p1.y)} ${n(p2.x)},${n(p2.y)} ${n(p3.x)},${n(p3.y)}`;
}

/** Half-width of the flow chevron, in world px. */
export const FLOW_MARK_PX = 4.5;

/**
 * A chevron at the curve's midpoint, pointing the way values travel.
 *
 * Direction was already in the geometry — every connector runs producer to
 * consumer — but nothing on screen said so, and an upstream cone and a
 * downstream one drawn on the same edge look identical without it.
 *
 * It is a stroked chevron rather than a filled arrowhead for two reasons: the
 * rest of this UI is drawn rather than filled, and a solid triangle at the node
 * boundary competes with the card outlines it sits between. Mid-curve is also
 * the one place on the path that no node can cover.
 */
export function flowMarkPath(c: Cubic, size = FLOW_MARK_PX, t = 0.5): string {
  const [p0, p1, p2, p3] = c;
  // B(t) and B'(t) for a cubic Bezier.
  const u = 1 - t;
  const mid = {
    x: u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
    y: u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y,
  };
  let tx = 3 * u * u * (p1.x - p0.x) + 6 * u * t * (p2.x - p1.x) + 3 * t * t * (p3.x - p2.x);
  let ty = 3 * u * u * (p1.y - p0.y) + 6 * u * t * (p2.y - p1.y) + 3 * t * t * (p3.y - p2.y);
  let len = Math.hypot(tx, ty);
  if (len < 1e-6) {
    // A degenerate tangent means the handles cancel; the chord still has the
    // direction, and only its direction is wanted here.
    tx = p3.x - p0.x;
    ty = p3.y - p0.y;
    len = Math.hypot(tx, ty);
    if (len < 1e-6) return "";
  }
  const ux = tx / len;
  const uy = ty / len;
  // Perpendicular, for the two wings.
  const px = -uy;
  const py = ux;
  const tip = { x: mid.x + ux * size * 0.6, y: mid.y + uy * size * 0.6 };
  const wing = (s: number) => ({
    x: tip.x - ux * size + px * size * s,
    y: tip.y - uy * size + py * size * s,
  });
  const a = wing(1);
  const b = wing(-1);
  return `M${n(a.x)},${n(a.y)} L${n(tip.x)},${n(tip.y)} L${n(b.x)},${n(b.y)}`;
}

/**
 * The flow mark for one connector in a fanned bundle, staggered along its own
 * curve so the marks cascade instead of lining up.
 *
 * At the fan spacing the chevrons are exactly as tall as the gap between the
 * lines, so aligned marks abut and read as one continuous zigzag rather than as
 * N arrows. Staggering fixes that without widening the fan, which the op node's
 * own height caps anyway. Arc position is approximated from the chord — good
 * enough for spacing marks whenever the chord tracks the arc, which it does at
 * any separation dagre can produce (its ranksep exceeds the bend floor). The
 * range is clamped so marks stay off the endpoints; on a connector short enough
 * for the cubic to loop back on itself that clamp is in `t`, not in pixels, so
 * a mark may still crowd an end. Arc-length parameterisation would fix it and
 * is not worth its cost for a layout the graph does not generate.
 */
/** Where along a connector a bundle's marks are allowed to sit. */
const MARK_BAND: [number, number] = [0.26, 0.74];

export function fannedFlowMark(
  c: Cubic,
  slot: number,
  count: number,
  spacing: number
): string {
  if (count <= 1) return flowMarkPath(c);
  // A chevron spans 2*size across the line, so at the default size it is
  // exactly as wide as a default fan gap and the marks of neighbouring lines
  // touch. Keep each mark inside its own lane.
  const size = Math.min(FLOW_MARK_PX, spacing * 0.38);
  // Spread across a fixed band rather than a few percent either side of the
  // midpoint. Two connectors that share a pair are a translation of each other,
  // so where the tangent runs parallel to the offset — the middle of an S — the
  // lines have no perpendicular separation at all and marks clustered there
  // overlap however small they are. They are furthest apart near the bends.
  const t = MARK_BAND[0] + ((slot + 0.5) / count) * (MARK_BAND[1] - MARK_BAND[0]);
  return flowMarkPath(c, size, t);
}
