import { beforeEach, describe, expect, it } from "vitest";
import { box } from "../../../src/core/region";
import { cardScaleFor, cardSize } from "../../../src/view/tensor/card-size";
import { gridGeometry } from "../../../src/view/tensor/grid";
import { decodeWorkspace, encodeWorkspace } from "../../../src/state/share";
import { useStore } from "../../../src/state/store";
import { axesWith, remapped, viewAxes, viewCfgFits } from "../../../src/view/tensor/tensor-view";
import { MAX_GRAPH_H, MAX_GRAPH_W } from "../../../src/view/tensor/tiling";
import { selectionBoxFromDrag } from "../../../src/view/tensor/gesture";

const S = () => useStore.getState();

/* [B, S, H, D] after the head split: the default plane is H x D, and the one
   worth studying is S x D. */
const DSL = "B = 1\nL = 128\nH = 4\nD = 32\nX = Tensor(B, L, H, D, dtype=fp16)\nY = relu(X)\n";
const SHAPE = [1, 128, 4, 32];

describe("choosing the axes a card draws", () => {
  it("draws the chosen pair, and the default pair when none is chosen or the choice is invalid", () => {
    expect(viewAxes(SHAPE)).toEqual({ rowAxis: 2, colAxis: 3 });
    expect(viewAxes(SHAPE, { axes: [1, 3] })).toEqual({ rowAxis: 1, colAxis: 3 });
    expect(viewAxes(SHAPE, { axes: [3, 1] })).toEqual({ rowAxis: 3, colAxis: 1 });
    expect(viewAxes(SHAPE, { axes: [1, 1] })).toEqual({ rowAxis: 2, colAxis: 3 });
    expect(remapped(SHAPE, { axes: [2, 3] })).toBe(false);
    expect(viewCfgFits(SHAPE, { projection: true, sliders: [0, 0, 0, 0], axes: [1, 4] })).toBe(false);
  });

  it("keeps the graph's scale when it fits and shrinks to the budget when it does not", () => {
    expect(cardScaleFor(SHAPE, undefined, 5)).toBe(5);
    expect(cardScaleFor(SHAPE, { axes: [1, 3] }, 2)).toBe(2);
    const shrunk = cardScaleFor(SHAPE, { axes: [1, 3] }, 10);
    expect(shrunk).toBeCloseTo(Math.min(MAX_GRAPH_H / 128, MAX_GRAPH_W / 32));
    const size = cardSize(SHAPE, 10, "X", [], { axes: [1, 3] });
    expect(size.h).toBeLessThanOrEqual(24 + 24 + 2 * 20 + MAX_GRAPH_H + 1);
  });

  it("gives an axis a role, swapping when it holds the other one", () => {
    // Default plane: rows H (2), columns D (3).
    expect(axesWith(SHAPE, undefined, 1, "rows")).toEqual([1, 3]);
    expect(axesWith(SHAPE, undefined, 3, "rows")).toEqual([3, 2]);
    expect(axesWith(SHAPE, undefined, 0, "cols")).toEqual([2, 0]);
    expect(axesWith(SHAPE, { axes: [1, 3] }, 1, "cols")).toEqual([3, 1]);
  });

  it("selects on the chosen axes and takes the hidden positions on the others", () => {
    const cfg = { projection: false, sliders: [0, 0, 3, 0], axes: [1, 3] as [number, number] };
    const geom = gridGeometry(SHAPE, cfg, 0, 2);
    expect([geom.rows, geom.cols]).toEqual([128, 32]);
    const picked = selectionBoxFromDrag(SHAPE, cfg, geom, { r0: 64, c0: 0, r1: 127, c1: 31 }, false);
    expect(picked).toEqual(box([0, 1], [64, 128], [3, 4], [0, 32]));
  });
});

describe("axes in the workspace", () => {
  beforeEach(() => {
    S().applyDSL(DSL);
    S().setViewCfg("X", { projection: false });
    S().setSelection("X", { boxes: [box([0, 1], [64, 128], [2, 3], [0, 32])], exact: true, reasons: [] }, "replace");
  });

  it("keeps the studied tile on screen when its axes become hidden", () => {
    S().setViewAxes("X", [1, 3]);
    expect(S().viewCfgs.X.axes).toEqual([1, 3]);
    // H is hidden now; the slice moves onto the tile's head.
    expect(S().viewCfgs.X.sliders[2]).toBe(2);
    // The selection itself is untouched: this is presentation.
    expect(S().selection!.parts[0].box).toEqual(box([0, 1], [64, 128], [2, 3], [0, 32]));
  });

  it("keeps the focused tile on screen, else the last one drawn on the tensor", () => {
    const tile = (head: number) => box([0, 1], [64, 128], [head, head + 1], [0, 32]);
    S().setSelection("X", { boxes: [tile(2)], exact: true, reasons: [] }, "replace");
    S().setSelection("X", { boxes: [tile(3)], exact: true, reasons: [] }, "union");
    expect(S().selection!.parts).toHaveLength(2);

    S().setViewAxes("X", [1, 3]);
    expect(S().viewCfgs.X.sliders[2]).toBe(3); // the last one drawn

    S().setViewAxes("X", null);
    S().hoverBox(0);
    S().togglePinBox(0);
    S().setViewAxes("X", [1, 3]);
    expect(S().viewCfgs.X.sliders[2]).toBe(2); // the focused one
  });

  it("leaves the positions alone on a tensor with no tile", () => {
    S().setSelection("Y", { boxes: [box([0, 1], [0, 8], [0, 1], [0, 32])], exact: true, reasons: [] }, "replace");
    const sliders = S().viewCfgs.X.sliders.slice();
    S().setViewAxes("X", [1, 3]);
    expect(S().viewCfgs.X.sliders).toEqual(sliders);
  });

  it("stores the default pair as absent", () => {
    S().setViewAxes("X", [1, 3]);
    S().setViewAxes("X", [2, 3]);
    expect(S().viewCfgs.X.axes).toBeUndefined();
  });

  it("round trips through a share link", () => {
    S().setViewAxes("X", [1, 3]);
    const views = { X: S().viewCfgs.X };
    const decoded = decodeWorkspace(`#s=${encodeWorkspace({ dsl: DSL, dir: "both", tile: 0, sel: null, views })}`)!;
    expect(S().restoreWorkspace({
      dsl: DSL, direction: "both", tileScale: 0, snapToGrid: true, axisMode: "symbolic",
      parts: null, viewCfgs: decoded.views,
    })).toBe(true);
    expect(S().viewCfgs.X.axes).toEqual([1, 3]);
  });
});
