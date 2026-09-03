import { describe, expect, it } from "vitest";
import {
  constrainRectMotion,
  cubicPath,
  curvedEdge,
  EDGE_FAN_MAX_PX,
  fanSpacing,
  fannedFlowMark,
  flowMarkPath,
  FLOW_MARK_PX,
  type Cubic,
  NODE_GAP,
  rectsOverlap,
  Rect,
} from "../ui/graph-geometry";

const card: Rect = { x: 20, y: 40, w: 100, h: 80 };
const op: Rect = { x: 180, y: 60, w: 60, h: 30 };

describe("solid graph-node motion", () => {
  it("stops at the full-card boundary, including the safety gap", () => {
    const moved = constrainRectMotion(card, { x: 500, y: 0 }, [op]);
    expect(moved.x + moved.w + NODE_GAP).toBe(op.x);
    expect(rectsOverlap(moved, op)).toBe(false);
  });

  it("cannot tunnel through a blocker on a large pointer jump", () => {
    const moved = constrainRectMotion(card, { x: 10_000, y: 0 }, [op]);
    expect(moved.x).toBeLessThan(op.x);
    expect(rectsOverlap(moved, op)).toBe(false);
  });

  it("slides along a collision boundary on the free axis", () => {
    const moved = constrainRectMotion(card, { x: 100, y: 75 }, [op]);
    expect(moved.x + moved.w + NODE_GAP).toBe(op.x);
    expect(moved.y).toBe(115);
    expect(rectsOverlap(moved, op)).toBe(false);
  });

  it("allows movement into negative world coordinates", () => {
    const moved = constrainRectMotion(card, { x: -1000, y: -1000 }, []);
    expect(moved.x).toBe(-980);
    expect(moved.y).toBe(-960);
  });
});

describe("curved graph connectors", () => {
  it("uses a cubic path between facing horizontal edges", () => {
    expect(cubicPath(curvedEdge(card, op))).toBe("M120,80 C152,80 148,75 180,75");
  });

  it("switches to vertical anchors for a mostly vertical relationship", () => {
    const below = { x: 35, y: 220, w: 60, h: 30 };
    const path = cubicPath(curvedEdge(card, below));
    expect(path).toMatch(/^M70,120 C70,/);
    expect(path).toMatch(/ 65,220$/);
  });
});

describe("flow direction marks", () => {
  it("points along the curve, from producer to consumer", () => {
    // card is left of op, so the flow runs +x and the chevron tip leads it.
    const c = curvedEdge(card, op);
    const pts = flowMarkPath(c).match(/-?\d+(\.\d+)?/g)!.map(Number);
    const [ax, , tipX, , bx] = pts;
    expect(tipX).toBeGreaterThan(ax);
    expect(tipX).toBeGreaterThan(bx);
  });

  it("reverses when the consumer is on the other side", () => {
    const left = { x: 0, y: 70, w: 40, h: 20 };
    const forward = flowMarkPath(curvedEdge(left, op)).match(/-?\d+(\.\d+)?/g)!.map(Number);
    const backward = flowMarkPath(curvedEdge(op, left)).match(/-?\d+(\.\d+)?/g)!.map(Number);
    // tip x relative to the first wing flips sign with the flow
    expect(Math.sign(forward[2] - forward[0])).toBe(-Math.sign(backward[2] - backward[0]));
  });

  it("turns with a vertical connector rather than staying horizontal", () => {
    const below = { x: 35, y: 220, w: 60, h: 30 };
    const pts = flowMarkPath(curvedEdge(card, below)).match(/-?\d+(\.\d+)?/g)!.map(Number);
    const [, ay, , tipY] = pts;
    expect(tipY).toBeGreaterThan(ay); // tip leads downward
  });

  it("sits on the curve it marks", () => {
    const c = curvedEdge(card, op);
    const mid = {
      x: (c[0].x + 3 * c[1].x + 3 * c[2].x + c[3].x) / 8,
      y: (c[0].y + 3 * c[1].y + 3 * c[2].y + c[3].y) / 8,
    };
    const pts = flowMarkPath(c).match(/-?\d+(\.\d+)?/g)!.map(Number);
    const tip = { x: pts[2], y: pts[3] };
    expect(Math.hypot(tip.x - mid.x, tip.y - mid.y)).toBeLessThan(FLOW_MARK_PX);
  });

  it("falls back to the chord when the handles cancel", () => {
    const degenerate: Cubic = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: -10, y: 0 },
      { x: 20, y: 0 },
    ];
    expect(flowMarkPath(degenerate)).not.toBe("");
  });

  it("emits nothing for a zero-length connector rather than a NaN path", () => {
    const point = { x: 5, y: 5 };
    expect(flowMarkPath([point, point, point, point])).toBe("");
  });
});

describe("fanned connectors", () => {
  it("anchors at the facing side's centre with no offset", () => {
    expect(curvedEdge(card, op, 0)).toEqual(curvedEdge(card, op));
  });

  it("slides both anchors along the side, keeping the pair parallel", () => {
    const [s0, , , e0] = curvedEdge(card, op, 0);
    const [s1, , , e1] = curvedEdge(card, op, 4);
    expect(s1.y - s0.y).toBe(4);
    expect(e1.y - e0.y).toBe(4);
    expect(s1.x).toBe(s0.x); // still on the same facing side
    expect(e1.x).toBe(e0.x);
  });

  it("offsets across the other axis for a vertical connector", () => {
    const below = { x: 35, y: 220, w: 60, h: 30 };
    const [s0] = curvedEdge(card, below, 0);
    const [s1] = curvedEdge(card, below, 5);
    expect(s1.x - s0.x).toBe(5);
    expect(s1.y).toBe(s0.y);
  });

  it("clamps the offset so an anchor never leaves its side", () => {
    // op is 30 tall, so anchors may travel at most 15 - 3 = 12 from centre.
    const [, , , far] = curvedEdge(card, op, 500);
    const [, , , limit] = curvedEdge(card, op, 12);
    expect(far).toEqual(limit);
  });

  it("gives distinct marks to two connectors between the same pair", () => {
    const s = fanSpacing(card, op, 2);
    const a = flowMarkPath(curvedEdge(card, op, -s / 2));
    const b = flowMarkPath(curvedEdge(card, op, s / 2));
    expect(a).not.toBe(b);
  });

  it("spreads a bundle as wide as the facing sides allow, up to the cap", () => {
    // op is 30 tall, so a horizontal pair may sit at most 2*(15-3) = 24 apart
    expect(fanSpacing(card, op, 2)).toBe(EDGE_FAN_MAX_PX);
    // three lines have to share that room, so they land inside the cap
    expect(fanSpacing(card, op, 3)).toBe(12);
    expect(fanSpacing(card, op, 3)).toBeLessThan(EDGE_FAN_MAX_PX);
  });

  it("never spreads a bundle past its anchors' side", () => {
    for (const count of [2, 3, 4, 6]) {
      const spacing = fanSpacing(card, op, count);
      expect((count - 1) * spacing).toBeLessThanOrEqual(op.h - 6);
    }
  });

  it("uses the wider of the two axes for a vertical connector", () => {
    const below = { x: 35, y: 220, w: 60, h: 30 };
    // limited by width (60) here rather than the 30 that caps a horizontal pair
    expect(fanSpacing(card, below, 3)).toBe(EDGE_FAN_MAX_PX);
  });
});

describe("staggered marks on a fanned bundle", () => {
  const bundle = (count: number) =>
    Array.from({ length: count }, (_, i) => {
      const spacing = fanSpacing(card, op, count);
      const c = curvedEdge(card, op, (i - (count - 1) / 2) * spacing);
      const pts = fannedFlowMark(c, i, count, spacing).match(/-?\d+(\.\d+)?/g)!.map(Number);
      return { tip: { x: pts[2], y: pts[3] }, spacing };
    });

  it("leaves a lone connector's mark at the midpoint", () => {
    const c = curvedEdge(card, op);
    expect(fannedFlowMark(c, 0, 1, 0)).toBe(flowMarkPath(c));
  });

  it("keeps each mark inside its own lane", () => {
    for (const count of [2, 3, 4, 6]) {
      const { spacing } = bundle(count)[0];
      const c = curvedEdge(card, op);
      const pts = fannedFlowMark(c, 0, count, spacing).match(/-?\d+(\.\d+)?/g)!.map(Number);
      // wing-to-wing span, measured across the two outer points
      const span = Math.hypot(pts[4] - pts[0], pts[5] - pts[1]);
      expect(span).toBeLessThan(spacing);
    }
  });

  it("separates neighbouring marks rather than merging them into one zigzag", () => {
    for (const count of [2, 3, 4]) {
      const b = bundle(count);
      for (let i = 1; i < b.length; i++) {
        const d = Math.hypot(b[i].tip.x - b[i - 1].tip.x, b[i].tip.y - b[i - 1].tip.y);
        expect(d).toBeGreaterThan(Math.min(FLOW_MARK_PX, b[i].spacing * 0.38) * 2);
      }
    }
  });

  it("keeps every mark clear of both endpoints on a short connector", () => {
    // As close as dagre's ranksep allows; nearer than this the cubic loops back
    // on itself and arc length stops tracking the chord.
    const near = { x: 166, y: 60, w: 40, h: 30 };
    const c = curvedEdge(card, near);
    const spacing = fanSpacing(card, near, 4);
    for (let i = 0; i < 4; i++) {
      const pts = fannedFlowMark(c, i, 4, spacing).match(/-?\d+(\.\d+)?/g)!.map(Number);
      const tip = { x: pts[2], y: pts[3] };
      expect(Math.hypot(tip.x - c[0].x, tip.y - c[0].y)).toBeGreaterThan(FLOW_MARK_PX);
      expect(Math.hypot(tip.x - c[3].x, tip.y - c[3].y)).toBeGreaterThan(FLOW_MARK_PX);
    }
  });
});
