import { describe, expect, it } from "vitest";
import { overviewLabelWidths } from "../ui/overview-labels";
import type { PlacedGraphNode } from "../ui/graph-scene";

const card = (id: string, x: number, y: number, w = 300): PlacedGraphNode =>
  ({ id, x, y, w, h: 200, kind: "tensor" });

describe("overview names", () => {
  it("keeps short names readable and caps long names inside their card width", () => {
    const nodes = [card("A", 0, 0), card("B", 400, 0)];
    const widths = overviewLabelWidths(nodes, 0.3, { A: "A", B: "a_very_long_tensor_name" });
    expect(widths.get("A")! * 0.3).toBeGreaterThanOrEqual(12);
    expect(widths.get("B")! * 0.3).toBeLessThanOrEqual(90);
    expect(overviewLabelWidths(nodes, 1, { A: "A" }).size).toBe(0);
  });

  it("does not enlarge a name into a neighbouring tensor or operation", () => {
    for (const kind of ["tensor", "op"] as const) {
      const nodes = [card("A", 0, 205), { ...card("B", 0, 0), kind }];
      expect(overviewLabelWidths(nodes, 0.1, { A: "A", B: "B" }).has("A")).toBe(false);
    }
  });

  it("leaves names small when the whole card is narrower than a readable label", () => {
    const nodes = [card("A", 0, 0), card("B", 300, 0)];
    const widths = overviewLabelWidths(nodes, 0.03, { A: "long_name", B: "long_name" });
    // Neither 9px-wide card can support a readable label.
    expect(widths.size).toBe(0);
  });
});
