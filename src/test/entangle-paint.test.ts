/**
 * Entanglement's paint encoding.
 *
 * The rules worth pinning are the ones a reader would misread if they broke:
 * that the texture is not the downstream ruling, that the toggle is orthogonal
 * to direction rather than a fourth value of it, and that hiding a tile hides
 * this too - one visibility control, not two.
 */

import { describe, expect, it } from "vitest";
import { buildLayers } from "../ui/TensorCard";
import { box, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";

const base = {
  tensorId: "B", dark: false, direction: "backward" as const, isSelected: false,
  parts: [], partCount: 1, perBox: null, hiddenBoxes: new Set<number>(),
  focusedBox: null, dragRegion: null,
};

describe("entanglement paint", () => {
  const p = compileDSL("A = Tensor(8, 8)\nB = Tensor(8, 8)\nC = matmul(A, B)\n");
  const ent = p.executor.entangled("A", fromBox(box([0, 4], [0, 4])));
  const rows = ent.map((e) => ({ index: 0, region: e.region }));

  it("paints nothing when the toggle is off", () => {
    const l = buildLayers({ ...base, entangled: rows, showEntangled: false });
    expect(l.filter((x) => x.pattern?.kind === "stipple")).toHaveLength(0);
  });

  it("paints a stipple layer when on", () => {
    const l = buildLayers({ ...base, entangled: rows, showEntangled: true });
    const stipple = l.filter((x) => x.pattern?.kind === "stipple");
    expect(stipple).toHaveLength(1);
    expect(stipple[0].region.boxes).toEqual([box([0, 4], [0, 8])]);
  });

  it("paints regardless of direction, including none", () => {
    for (const direction of ["none", "backward", "forward", "both"] as const) {
      const l = buildLayers({ ...base, direction, entangled: rows, showEntangled: true });
      expect(l.some((x) => x.pattern?.kind === "stipple"), direction).toBe(true);
    }
  });

  it("respects the per-tile hide toggle", () => {
    const l = buildLayers({
      ...base, entangled: rows, showEntangled: true, hiddenBoxes: new Set([0]),
    });
    expect(l.filter((x) => x.pattern?.kind === "stipple")).toHaveLength(0);
  });

  it("uses a texture distinct from the downstream ruling", () => {
    const l = buildLayers({ ...base, entangled: rows, showEntangled: true });
    const kinds = new Set(l.map((x) => x.pattern?.kind).filter(Boolean));
    expect(kinds.has("stipple")).toBe(true);
    expect(kinds.has("stripe")).toBe(false);
  });
});
