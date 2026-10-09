import { describe, expect, it } from "vitest";
import { carriedOffsets } from "../../../src/view/graph/carry-placement";
import { NODE_GAP, rectsOverlap } from "../../../src/view/graph/graph-geometry";
import type { PlacedGraphNode } from "../../../src/view/graph/graph-scene";

const node = (id: string, x: number, y: number, w: number, h: number, kind: "tensor" | "op" = "tensor"): PlacedGraphNode =>
  ({ id, kind, x, y, w, h });
const drawn = (entries: [string, number, number, number, number][]) =>
  new Map(entries.map(([key, x, y, w, h]) => [key, { x, y, w, h }]));
const placed = (base: PlacedGraphNode[], offsets: Record<string, { dx: number; dy: number }>) =>
  base.map((n) => {
    const o = offsets[`${n.kind === "tensor" ? "t" : "n"}:${n.id}`] ?? { dx: 0, dy: 0 };
    return { x: n.x + o.dx, y: n.y + o.dy, w: n.w, h: n.h };
  });

describe("keeping nodes in place across a size-only run", () => {
  it("holds an unchanged node where it was drawn, wherever the new layout put it", () => {
    const base = [node("A", 0, 0, 100, 100)];
    const offsets = carriedOffsets(drawn([["t:A", 300, 40, 100, 100]]), base);
    expect(offsets["t:A"]).toEqual({ dx: 300, dy: 40 });
  });

  it("shrinks a card about the centre it was left at", () => {
    const base = [node("A", 0, 0, 50, 50)];
    const [rect] = placed(base, carriedOffsets(drawn([["t:A", 100, 100, 100, 100]]), base));
    expect(rect.x + rect.w / 2).toBe(150);
    expect(rect.y + rect.h / 2).toBe(150);
  });

  it("nudges a grown card clear of a neighbour that kept its size", () => {
    const base = [node("A", 0, 0, 300, 100), node("B", 400, 0, 100, 100)];
    const before = drawn([["t:A", 0, 0, 100, 100], ["t:B", 130, 0, 100, 100]]);
    const offsets = carriedOffsets(before, base);
    const [a, b] = placed(base, offsets);
    // The neighbour did not grow, so it stays exactly where it was.
    expect(b).toEqual({ x: 130, y: 0, w: 100, h: 100 });
    expect(rectsOverlap(a, b)).toBe(false);
    expect(rectsOverlap(a, b, NODE_GAP - 1e-6)).toBe(false);
  });

  it("leaves a node it has no record of at its generated position", () => {
    const base = [node("A", 10, 20, 50, 50), node("m", 90, 20, 30, 20, "op")];
    expect(carriedOffsets(drawn([["t:A", 10, 20, 50, 50]]), base)).toEqual({});
  });
});
