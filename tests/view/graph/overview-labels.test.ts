import { describe, expect, it } from "vitest";
import {
  LABEL_FLOOR_SCALE,
  operandLabelsLegible,
  overviewLabels,
  overviewTextScale,
} from "../../../src/view/graph/overview-labels";
import type { PlacedGraphNode } from "../../../src/view/graph/graph-scene";

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

/* Counter-scaling keeps a label readable while zooming out, but only down to a
   floor: past it the label shrinks with the graph, so it never grows without
   bound against the nodes it names, and it is dropped once it would be a smudge. */
describe("bounded label growth", () => {
  it("counter-scales down to the floor and no further", () => {
    expect(overviewTextScale(0.5)).toBe(0.5);
    expect(overviewTextScale(LABEL_FLOOR_SCALE / 2)).toBe(LABEL_FLOOR_SCALE);
  });

  it("keeps a name's world size below the floor instead of growing it", () => {
    const nodes = [card("A", 0, 0, 3000)];
    const atFloor = overviewLabels(nodes, LABEL_FLOOR_SCALE, { A: "Scores" }).tensors.get("A")!;
    const below = overviewLabels(nodes, LABEL_FLOOR_SCALE * 0.6, { A: "Scores" }).tensors.get("A")!;
    expect(below).toBeLessThan(atFloor * 1.1);
  });

  it("places no label once the text would be too small to read", () => {
    const nodes = [card("A", 0, 0, 3000), { ...card("n", 4000, 0, 70), kind: "op" as const }];
    const labels = overviewLabels(nodes, LABEL_FLOOR_SCALE * 0.4, { A: "A", n: "matmul" });
    expect(labels.tensors.size).toBe(0);
    expect(labels.ops.size).toBe(0);
  });

  it("draws operand labels only while they are legible", () => {
    expect(operandLabelsLegible(1)).toBe(true);
    expect(operandLabelsLegible(0.6)).toBe(true);
    expect(operandLabelsLegible(0.3)).toBe(false);
  });
});
