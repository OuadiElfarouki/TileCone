import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { box, fromBox, iv } from "../../src/core/region";
import { compileDSL } from "../../src/parse/compiler";
import { useStore } from "../../src/state/store";
import { GrowthSummary } from "../../src/components/inspector/growth";
import { activeGrowth, growthDelta, stepEdge } from "../../src/view/growth";
import { buildLayers } from "../../src/view/tensor/layers";
import { boxColor } from "../../src/view/palette";
import { keyboardTile } from "../../src/view/workspace";

vi.mock("../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const S = () => useStore.getState();
const GEMM = "A = Tensor(16, 32, dtype=fp16)\nB = Tensor(32, 16, dtype=fp16)\nC = matmul(A, B)\n";

describe("one grid step of the far edge", () => {
  it("grows to the next boundary, from on or off the lattice, and stops at the end", () => {
    expect(stepEdge(iv(0, 4), 4, 16, true)).toEqual(iv(0, 8));
    expect(stepEdge(iv(0, 5), 4, 16, true)).toEqual(iv(0, 8));
    expect(stepEdge(iv(0, 14), 4, 15, true)).toEqual(iv(0, 15)); // a short last tile
    expect(stepEdge(iv(0, 16), 4, 16, true)).toBeNull();
  });

  it("shrinks back to the previous boundary, never below one step", () => {
    expect(stepEdge(iv(0, 8), 4, 16, false)).toEqual(iv(0, 4));
    expect(stepEdge(iv(0, 15), 4, 15, false)).toEqual(iv(0, 12));
    expect(stepEdge(iv(4, 8), 4, 16, false)).toBeNull();
    expect(stepEdge(iv(0, 3), 1, 16, false)).toEqual(iv(0, 2));
  });
});

describe("what growth adds to a tile's needs", () => {
  /* The example the lens exists for: one more block of C's columns needs no
     more of A, and a whole new band of B. */
  it("grows a GEMM tile: no new A, a new band of B, new outputs", () => {
    const { resolved } = compileDSL(GEMM);
    const delta = growthDelta(resolved, "C", box([0, 4], [0, 4]), box([0, 4], [0, 8]));
    expect(delta.outputs).toBe(16);
    expect(delta.exact).toBe(true);
    expect(delta.tensors.map((t) => [t.tensorId, t.elements])).toEqual([["A", 0], ["B", 32 * 4]]);
    expect(delta.tensors[1].added).toMatchObject({ boxes: [box([0, 32], [4, 8])] });
  });

  it("adds nothing when grown and shrunk back to where it started", () => {
    const { resolved } = compileDSL(GEMM);
    const delta = growthDelta(resolved, "C", box([0, 4], [0, 4]), box([0, 4], [0, 4]));
    expect([delta.outputs, ...delta.tensors.map((t) => t.elements)]).toEqual([0, 0, 0]);
  });
});

describe("growing the keyboard tile", () => {
  beforeEach(() => {
    S().applyDSL(GEMM);
    S().setInspectorTab("dependencies");
    S().setSnapToGrid(false);
    S().setSelection("C", fromBox(box([0, 4], [0, 4])), "replace");
  });

  it("keeps the starting box across steps and reports from it", () => {
    S().growSelection(1, true);
    S().growSelection(1, true);
    const growth = activeGrowth(S().growth, S().selection?.parts, keyboardTile(S()))!;
    expect(growth.baseBox).toEqual(box([0, 4], [0, 4]));
    expect(growth.box).toEqual(box([0, 4], [0, 6]));
    expect(growth.delta.tensors.find((t) => t.tensorId === "B")!.elements).toBe(32 * 2);
    S().growSelection(1, false);
    S().growSelection(1, false);
    expect(activeGrowth(S().growth, S().selection?.parts, keyboardTile(S()))!.delta.outputs).toBe(0);
  });

  it("retires the comparison when the tile is moved, undone or released, and starts over after a move", () => {
    S().growSelection(1, true);
    S().moveSelection(0, 1);
    expect(activeGrowth(S().growth, S().selection?.parts, keyboardTile(S()))).toBeNull();
    S().growSelection(1, true);
    expect(S().growth!.baseBox).toEqual(box([1, 5], [0, 5]));
    S().undoWorkspace();
    expect(activeGrowth(S().growth, S().selection?.parts, keyboardTile(S()))).toBeNull();
    S().growSelection(1, true);
    S().releaseEditBox();
    expect(activeGrowth(S().growth, S().selection?.parts, keyboardTile(S()))).toBeNull();
  });

  it("does nothing without a keyboard tile, and is one undo step per step", () => {
    const depth = S().workspaceHistory.length;
    S().growSelection(1, true);
    expect(S().workspaceHistory.length).toBe(depth + 1);
    S().growSelection(1, true, false);
    expect(S().workspaceHistory.length).toBe(depth + 1);
    S().releaseEditBox();
    const selection = S().selection;
    S().growSelection(1, true);
    expect(S().selection).toBe(selection);
  });

  it("summarises the growth under the tile, naming tensors with nothing new", () => {
    S().growSelection(1, true);
    const html = renderToStaticMarkup(createElement(GrowthSummary));
    expect(html).toContain("C[0:4, 0:4]</code> → <code>C[0:4, 0:5]");
    expect(html).toContain("+4 outputs");
    expect(html).toMatch(/<code>A<\/code><span class="reads-figures muted">no new elements/);
    expect(html).toContain("+32 elements · 64 B");
  });
});

describe("the growth on a card", () => {
  it("is drawn over the cones in the tile's hue, outlined", () => {
    const region = fromBox(box([0, 32], [4, 8]));
    const layers = buildLayers({
      tensorId: "B", dark: false, direction: "backward", isSelected: false, parts: [], partCount: 1,
      perBox: null, hiddenBoxes: new Set(), focusedBox: null, back: undefined, fwd: undefined,
      prev: undefined, prevForward: undefined, dragRegion: null, entangled: undefined, showEntangled: false,
      grown: { index: 0, region },
    });
    expect(layers).toEqual([
      { region, color: boxColor(0, false), alpha: 0.55, hatch: false, outline: true, lineWidth: 2.7 },
    ]);
  });
});
