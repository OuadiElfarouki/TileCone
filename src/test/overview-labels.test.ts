import { describe, expect, it } from "vitest";
import { overviewLabels } from "../ui/overview-labels";
import type { PlacedGraphNode } from "../ui/graph-scene";

const card = (id: string, x: number, y: number, w = 300): PlacedGraphNode =>
  ({ id, x, y, w, h: 200, kind: "tensor" });

describe("overview names", () => {
  it("keeps short names readable and caps long names inside their card width", () => {
    const nodes = [card("A", 0, 0), card("B", 400, 0)];
    const widths = overviewLabels(nodes, 0.3, { A: "A", B: "a_very_long_tensor_name" }).tensors;
    expect(widths.get("A")! * 0.3).toBeGreaterThanOrEqual(12);
    expect(widths.get("B")! * 0.3).toBeLessThanOrEqual(90);
    expect(overviewLabels(nodes, 1, { A: "A" }).tensors.size).toBe(0);
  });

  it("does not enlarge a name into a neighbouring tensor or operation", () => {
    for (const kind of ["tensor", "op"] as const) {
      const nodes = [card("A", 0, 205), { ...card("B", 0, 0), kind }];
      expect(overviewLabels(nodes, 0.1, { A: "A", B: "B" }).tensors.has("A")).toBe(false);
    }
  });

  it("leaves names small when the whole card is narrower than a readable label", () => {
    const nodes = [card("A", 0, 0), card("B", 300, 0)];
    const widths = overviewLabels(nodes, 0.03, { A: "long_name", B: "long_name" }).tensors;
    // Neither 9px-wide card can support a readable label.
    expect(widths.size).toBe(0);
  });
});

describe("overview operation labels", () => {
  const op = (id: string, x: number, y: number): PlacedGraphNode =>
    ({ id, x, y, w: 70, h: 30, kind: "op" });

  /* The point of placing these separately from tensor names: a node box is
     sized for its label at scale 1, so a counter-scaled label cannot fit it. */
  it("lets an operation label extend past the node box it names", () => {
    const placed = overviewLabels([op("n", 0, 0)], 0.2, { n: "matmul" }).ops.get("n");
    expect(placed).toBeDefined();
    expect(placed!.w).toBeGreaterThan(70);
    // Centred across the node, so it still reads as belonging to it.
    expect(placed!.dy).toBeLessThan(30);
    expect(placed!.dy + 12 / 0.2).toBeGreaterThan(0);
  });

  it("falls back to the gap above the node when a neighbour blocks the middle", () => {
    const nodes = [op("n", 0, 0), card("T", 0, 20, 200)];
    const placed = overviewLabels(nodes, 0.5, { n: "add", T: "T" }).ops.get("n");
    expect(placed).toBeDefined();
    // Above the node's own top edge rather than across its middle.
    expect(placed!.dy).toBeLessThan(0);
  });

  it("drops the label rather than covering a neighbour", () => {
    // Cards on both sides and above and below leave nothing free at this scale.
    const nodes = [
      op("n", 0, 0),
      card("L", -260, -100, 250), card("R", 90, -100, 250),
      card("U", -100, -220, 250), card("D", -100, 40, 250),
    ];
    expect(overviewLabels(nodes, 0.2, {
      n: "einsum", L: "L", R: "R", U: "U", D: "D",
    }).ops.has("n")).toBe(false);
  });
});
