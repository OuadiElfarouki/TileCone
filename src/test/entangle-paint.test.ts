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
import { box, count, fromBox } from "../core/region";
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

/**
 * D76/D77: the stipple must survive the two configurations where it would
 * otherwise disappear into another relation's mark — on a solid fill of its own
 * hue, and at an extent too small to hold a dot lattice. Both are the default
 * case rather than the edge one, which is what makes them worth pinning.
 */
describe("the stipple cannot collapse into another relation", () => {
  /* X is both upstream of Y and entangled with it: Y reads X through the first
     matmul, and the second matmul multiplies Y against X. So X carries a needs
     fill and a combined-with stipple in the same hue at once, which is the case
     D76 names — dots in a hue on a near-opaque ground of that hue are nothing. */
  const p = compileDSL(
    "X = Tensor(8, 8)\nW = Parameter(8, 8)\nY = matmul(X, W)\nZ = matmul(Y, X)\n"
  );
  const sel = box([0, 4], [0, 8]);
  const ent = p.executor.entangled("Y", fromBox(sel));
  const perBox = [
    {
      backward: p.executor.upstream("Y", fromBox(sel)),
      forward: p.executor.downstream("Y", fromBox(sel)),
    },
  ];
  const onX = {
    ...base,
    tensorId: "X",
    isSelected: false,
    parts: [],
    perBox,
    entangled: ent
      .filter((e) => e.tensorId === "X")
      .map((e) => ({ index: 0, region: e.region })),
    showEntangled: true,
  };

  it("has a case where both marks land on the same tensor", () => {
    expect(onX.entangled.length).toBeGreaterThan(0);
    expect(perBox[0].backward.tensors.has("X")).toBe(true);
  });

  it("knocks out where it lands on the needs fill of its own hue", () => {
    const stipple = buildLayers(onX).filter((l) => l.pattern?.kind === "stipple");
    expect(stipple.length).toBeGreaterThan(0);
    expect(stipple.some((l) => l.knockout)).toBe(true);
  });

  it("stays in the tile's hue when the needs fill is not painted", () => {
    // Direction "none" paints no cone, so there is no ground to knock out of.
    const stipple = buildLayers({ ...onX, direction: "none" }).filter(
      (l) => l.pattern?.kind === "stipple"
    );
    expect(stipple.length).toBeGreaterThan(0);
    expect(stipple.every((l) => !l.knockout)).toBe(true);
  });

  it("cuts the region against its ground rather than classifying it whole", () => {
    const stipple = buildLayers(onX).filter((l) => l.pattern?.kind === "stipple");
    for (const l of stipple) expect(l.region.boxes.length).toBeGreaterThan(0);
    // Every element of the original region is still covered by one treatment
    // or the other: splitting must not drop paint.
    const total = stipple.reduce((a, l) => a + count(l.region), 0);
    expect(total).toBeGreaterThanOrEqual(count(onX.entangled[0].region));
  });
});
