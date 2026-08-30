import { describe, expect, it } from "vitest";
import {
  constrainRectMotion,
  curvedEdgePath,
  NODE_GAP,
  rectsOverlap,
  Rect,
  WORLD_MARGIN,
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

  it("enforces the top-left world boundary", () => {
    const moved = constrainRectMotion(card, { x: -1000, y: -1000 }, []);
    expect(moved.x).toBe(WORLD_MARGIN);
    expect(moved.y).toBe(WORLD_MARGIN);
  });
});

describe("curved graph connectors", () => {
  it("uses a cubic path between facing horizontal edges", () => {
    expect(curvedEdgePath(card, op)).toBe("M120,80 C152,80 148,75 180,75");
  });

  it("switches to vertical anchors for a mostly vertical relationship", () => {
    const below = { x: 35, y: 220, w: 60, h: 30 };
    const path = curvedEdgePath(card, below);
    expect(path).toMatch(/^M70,120 C70,/);
    expect(path).toMatch(/ 65,220$/);
  });
});
