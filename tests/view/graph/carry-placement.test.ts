import { describe, expect, it } from "vitest";
import { carriedOffsets, roomForNode } from "../../../src/view/graph/carry-placement";
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

describe("making room for requested card sizes", () => {
  it("anchors the chosen card, clears its neighbours and preserves a distant node", () => {
    const base = [node("A", 100, 100, 100, 100), node("op", 240, 120, 64, 30, "op"), node("B", 350, 100, 100, 100), node("far", 1200, 1200, 100, 100)];
    const offsets = roomForNode(base, { "t:A": { dx: 30, dy: -40 } }, "t:A", { "t:A": { w: 400, h: 300 } });
    expect(offsets["t:A"]).toEqual({ dx: 30, dy: -40 });
    expect(offsets["t:far"]).toBeUndefined();
    const rects = placed(base.map((n) => n.id === "A" ? { ...n, w: 400, h: 300 } : n), offsets);
    for (const [i, a] of rects.entries()) for (const b of rects.slice(i + 1)) expect(rectsOverlap(a, b)).toBe(false);
  });
  it("moves only nodes that overlap, even where a displaced node lands beside one that does not", () => {
    // A's requested card covers N; N's shortest way out is up, onto M, which overlaps nothing.
    const base = [node("A", 0, 0, 100, 100), node("N", 150, 0, 100, 100), node("M", 150, -200, 100, 100)];
    const offsets = roomForNode(base, {}, "t:A", { "t:A": { w: 300, h: 400 } });
    expect(offsets["t:M"]).toBeUndefined();
    expect(offsets["t:N"]).toBeDefined();
    const rects = placed(base.map((n) => n.id === "A" ? { ...n, w: 300, h: 400 } : n), offsets);
    for (const [i, a] of rects.entries()) for (const b of rects.slice(i + 1)) expect(rectsOverlap(a, b)).toBe(false);
  });
  it("settles a dense graph at all requested sizes without moving the anchor", () => {
    const base = Array.from({ length: 80 }, (_, i) => node(String(i), i * 150, 0, 100, 100));
    const sizes = Object.fromEntries(base.map((n) => [`t:${n.id}`, { w: n.id === "0" ? 1800 : 130, h: n.id === "0" ? 1000 : 130 }]));
    const offsets = roomForNode(base, {}, "t:0", sizes);
    expect(offsets["t:0"]).toBeUndefined();
    const rects = placed(base.map((n) => ({ ...n, ...sizes[`t:${n.id}`] })), offsets);
    for (const [i, a] of rects.entries()) for (const b of rects.slice(i + 1)) expect(rectsOverlap(a, b)).toBe(false);
  });
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
