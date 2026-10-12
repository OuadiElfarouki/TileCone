import { beforeEach, describe, expect, it } from "vitest";
import { box } from "../../../src/core/region";
import { canvasSize, cardShapeView, cardSize, viewTags } from "../../../src/view/tensor/card-size";
import { DECK_ROOM_PX, DECK_SHEETS, deckSheets } from "../../../src/view/tensor/deck";
import { foldAxes, foldGeometry, MAX_UNFOLD_H, MAX_UNFOLD_W, runLocate, runSpans, UNFOLD_GAP_PX } from "../../../src/view/tensor/fold";
import { dragAt, dragMoved, planGesture, selectionBoxFromDrag } from "../../../src/view/tensor/gesture";
import { cellAt, gridGeometry, panelOrigins, regionRects } from "../../../src/view/tensor/grid";
import { cardViewMenu, HIDDEN_VIEWS, hiddenViewCfg } from "../../../src/view/tensor/menus";
import {
  hiddenAxisPositioned,
  hiddenMode,
  MAX_UNFOLD_PANELS,
  sliceOrdinal,
  unfoldRefusal,
  type ViewCfg,
  viewCfgFits,
} from "../../../src/view/tensor/tensor-view";
import { carriedViewCfg } from "../../../src/state/load";
import { useStore } from "../../../src/state/store";
import type { MenuChoice } from "../../../src/view/menu";

/* [B, H, M, N]: the default plane is M x N, with batch and head hidden. */
const SHAPE = [2, 4, 8, 6];
const name = (axis: number) => ["batch", "head", "m", "n"][axis];
const unfolded = (unfold: ViewCfg["unfold"]): ViewCfg => ({ projection: false, sliders: [0, 0, 0, 0], unfold });

describe("the hidden-axis reading", () => {
  it("is projection by default, slice without it, and unfold when set", () => {
    expect(hiddenMode(undefined)).toBe("project");
    expect(hiddenMode({ projection: true })).toBe("project");
    expect(hiddenMode({ projection: false })).toBe("slice");
    expect(hiddenMode({ projection: false, unfold: "down" })).toBe("unfold");
  });

  it("is chosen from the card's menu, with grid offered only for two hidden axes", () => {
    const hidden = cardViewMenu(SHAPE, { projection: true }, name)[0] as MenuChoice<string>;
    expect(hidden.id).toBe("hidden");
    expect(hidden.options.map((o) => o.label)).toEqual(["project", "slice", "unfold →", "unfold ↓", "unfold ▦"]);
    expect(hidden.options.find((o) => o.checked)?.label).toBe("project");
    const cube = cardViewMenu([3, 8, 6], undefined, (a) => `ax${a}`)[0] as MenuChoice<string>;
    expect(cube.options.map((o) => o.label)).toEqual(["project", "slice", "unfold →", "unfold ↓"]);
    // A matrix has nothing hidden, so its menu is only the plane.
    expect(cardViewMenu([8, 6], undefined, (a) => `ax${a}`)[0]).toMatchObject({ id: "rows" });
  });

  it("maps each menu value to the settings that select it", () => {
    expect(HIDDEN_VIEWS.map((_, v) => hiddenMode(hiddenViewCfg(v)))).toEqual(
      ["project", "slice", "unfold", "unfold", "unfold"]);
    expect(hiddenViewCfg(3)).toEqual({ projection: false, unfold: "down" });
  });

  it("refuses an unfold with too many panels, and says why in the menu", () => {
    const big = [MAX_UNFOLD_PANELS + 1, 8, 6];
    expect(viewCfgFits(big, { projection: false, sliders: [0, 0, 0], unfold: "across" })).toBe(false);
    const hidden = cardViewMenu(big, undefined, (a) => `ax${a}`)[0] as MenuChoice<string>;
    const unfold = hidden.options.filter((o) => o.label.startsWith("unfold"));
    expect(unfold.every((o) => o.disabled && o.title.startsWith("cannot unfold"))).toBe(true);
    // An unfold that projects is not a reading at all.
    expect(viewCfgFits(SHAPE, { projection: true, sliders: [0, 0, 0, 0], unfold: "across" })).toBe(false);
  });

  it("has no position to hold when unfolded, so nothing scrubs it", () => {
    expect(hiddenAxisPositioned(SHAPE, unfolded("across"), 0)).toBe(false);
  });

  it("keeps the rest of a carried view, sliced, when its new shape is too large to unfold", () => {
    const carried = carriedViewCfg([MAX_UNFOLD_PANELS + 1, 8, 6], { projection: false, sliders: [1, 0, 0], unfold: "down" });
    expect(carried).toEqual({ projection: false, sliders: [1, 0, 0] });
  });
});

describe("unfolding", () => {
  it("lays hidden axes out in stride order: slowest outermost", () => {
    expect(foldAxes(SHAPE, { unfold: "across" })).toEqual({ across: [0, 1], down: [] });
    expect(foldAxes(SHAPE, { unfold: "down" })).toEqual({ across: [], down: [0, 1] });
    expect(foldAxes(SHAPE, { unfold: "grid" })).toEqual({ across: [1], down: [0] });
    // A chosen plane leaves different axes hidden, still in axis order.
    expect(foldAxes(SHAPE, { unfold: "across", axes: [1, 3] })).toEqual({ across: [0, 2], down: [] });
  });

  it("separates outer blocks by wider gaps than inner ones", () => {
    const fold = foldGeometry(SHAPE, { unfold: "across" }, 8, 6, 10)!;
    const { panelW } = fold;
    const starts = runSpans(fold.across);
    // batch 0 holds heads 0..3, then a wider gap before batch 1.
    expect(starts.slice(0, 4)).toEqual([0, 1, 2, 3].map((h) => h * (panelW + UNFOLD_GAP_PX)));
    const batchPitch = 4 * panelW + 3 * UNFOLD_GAP_PX + 2 * UNFOLD_GAP_PX;
    expect(starts[4]).toBe(batchPitch);
    expect(fold.across.total).toBe(batchPitch + 4 * panelW + 3 * UNFOLD_GAP_PX);
    expect(fold.down.total).toBe(fold.panelH);
  });

  it("finds the panel under a position, giving a gap to the nearer side", () => {
    const fold = foldGeometry(SHAPE, { unfold: "across" }, 8, 6, 10)!;
    const second = runSpans(fold.across)[5]; // batch 1, head 1
    expect(runLocate(fold.across, second + 5)).toEqual({ digits: [1, 1], local: 5 });
    expect(runLocate(fold.across, fold.panelW + 1).digits).toEqual([0, 0]);
    expect(runLocate(fold.across, fold.panelW + UNFOLD_GAP_PX - 0.5).digits).toEqual([0, 1]);
  });

  it("draws a box once in every panel its hidden intervals reach", () => {
    const cfg = unfolded("grid");
    const geom = gridGeometry(SHAPE, cfg, 0, 10);
    expect(panelOrigins(geom)).toHaveLength(8);
    const rects = regionRects({ boxes: [box([1, 2], [1, 3], [0, 4], [0, 6])], exact: true, reasons: [] }, SHAPE, cfg, geom);
    expect(rects).toHaveLength(2);
    expect(rects.every((r) => r.alpha === 1 && r.w === geom.panelW && r.h === geom.panelH / 2)).toBe(true);
    // Batch 1 is the second row of panels, heads 1 and 2 the middle columns.
    expect(rects.map((r) => [r.x, r.y])).toEqual(
      [1, 2].map((h) => [h * (geom.panelW + UNFOLD_GAP_PX), geom.panelH + UNFOLD_GAP_PX]));
  });

  it("names the element and its panel under a point, and selects across panels by corners", () => {
    const cfg = unfolded("grid");
    const geom = gridGeometry(SHAPE, cfg, 0, 10);
    const pitchX = geom.panelW + UNFOLD_GAP_PX;
    const pitchY = geom.panelH + UNFOLD_GAP_PX;
    const start = cellAt(geom, pitchX + 1, 1)!; // batch 0, head 1, top-left element
    const end = cellAt(geom, 2 * pitchX + geom.panelW - 1, pitchY + geom.panelH - 1)!; // batch 1, head 2, last element
    expect(start).toEqual({ row: 0, col: 0, panel: [0, 1, 0, 0] });
    expect(end).toEqual({ row: 7, col: 5, panel: [1, 2, 0, 0] });
    const drag = { ...dragAt(start), r1: end.row, c1: end.col, p1: end.panel };
    expect(dragMoved(drag)).toBe(true);
    expect(dragMoved({ ...dragAt(start), p1: [1, 1, 0, 0] })).toBe(true);
    expect(selectionBoxFromDrag(SHAPE, cfg, geom, drag, false)).toEqual(box([0, 2], [1, 3], [0, 8], [0, 6]));
    // A press selects the one element in its own panel.
    expect(selectionBoxFromDrag(SHAPE, cfg, geom, dragAt(end), false)).toEqual(box([1, 2], [2, 3], [7, 8], [5, 6]));
  });

  it("shrinks to its budget and says so, and only changes the footprint when the reading does", () => {
    const wide = [4, 8, 64, 64];
    expect(cardShapeView(wide, { projection: true, sliders: [3, 0, 0, 0] } as ViewCfg)).toBeNull();
    expect(cardShapeView(wide, unfolded("across"))).toEqual({ projection: false, unfold: "across" });
    const size = cardSize(wide, 10, "X", [], { projection: false, unfold: "across" });
    expect(size.w).toBeLessThanOrEqual(MAX_UNFOLD_W);
    expect(viewTags(wide, { projection: false, unfold: "down" }, (a) => `ax${a}`).hidden).toBe("unfold ax0·ax1 ↓");
  });

  it.each(["across", "down", "grid"] as const)("commits the full previewed Plan span when unfolded %s", (dir) => {
    const cfg = { ...unfolded(dir), tile: [1, 2, 2, 3] };
    const geom = gridGeometry(SHAPE, cfg, 0, 10);
    const drag = { r0: 1, c0: 1, r1: 5, c1: 4, p0: [0, 1, 0, 0], p1: [1, 3, 0, 0] };
    const preview = selectionBoxFromDrag(SHAPE, cfg, geom, drag, true);
    expect(preview).toEqual(box([0, 2], [0, 4], [0, 6], [0, 6]));
    expect(planGesture(SHAPE, cfg, geom, drag, false)).toEqual({ kind: "divide", tile: [2, 4, 6, 6], element: [0, 0, 0, 0] });
    // An existing plan continues inspecting the pressed task, without retiling.
    expect(planGesture(SHAPE, cfg, geom, drag, true)).toEqual({ kind: "inspect", element: [0, 1, 1, 1] });
  });

  it.each(["across", "down", "grid"] as const)("includes nested gaps and rounding in the %s canvas budget", (dir) => {
    const shape = [4, 8, 64, 64];
    const cfg = unfolded(dir);
    const size = canvasSize(shape, cfg, 100);
    expect(size.w).toBeLessThanOrEqual(MAX_UNFOLD_W);
    expect(size.h).toBeLessThanOrEqual(MAX_UNFOLD_H);
    const geom = gridGeometry(shape, cfg, 0, 100);
    expect([geom.canvasW, geom.canvasH]).toEqual([size.w, size.h]);
  });

  it("offers a direction that fits singleton panels and refuses ones that cannot", () => {
    const shape = [100, 8, 1];
    const cfg: ViewCfg = { projection: false, sliders: [0, 0, 0], unfold: "across" };
    expect(unfoldRefusal(shape, cfg)).toContain("cannot fit");
    expect(viewCfgFits(shape, cfg)).toBe(false);
    const down = { ...cfg, unfold: "down" as const };
    expect(viewCfgFits(shape, down)).toBe(true);
    const size = canvasSize(shape, down, 10);
    expect(size.w).toBe(14);
    expect(size.h).toBeLessThanOrEqual(MAX_UNFOLD_H);
    const hidden = cardViewMenu(shape, cfg, name)[0] as MenuChoice<string>;
    expect(hidden.options.find((o) => o.value === 2)?.disabled).toBe(true);
    expect(hidden.options.find((o) => o.value === 3)?.disabled).toBe(false);
    expect(unfoldRefusal([256, 8, 1], cfg)).toContain("cannot fit");
  });

  it("fits singleton sides without shrinking the scale when they already fit", () => {
    const cfg: ViewCfg = { projection: false, sliders: [0, 0, 0], unfold: "across" };
    const size = canvasSize([64, 8, 1], cfg, 10);
    expect(size).toEqual({ w: 64 * 14 + 63 * UNFOLD_GAP_PX, h: 80 });
  });
});

describe("the slice deck", () => {
  it("counts the slice in row-major order over the hidden axes", () => {
    expect(sliceOrdinal(SHAPE, { sliders: [1, 2, 0, 0] })).toEqual({ index: 6, count: 8 });
    expect(sliceOrdinal(SHAPE, { sliders: [0, 0, 5, 5], axes: [0, 1] })).toEqual({ index: 35, count: 48 });
  });

  it("stacks what remains behind the slice, thinning to nothing on the last", () => {
    expect([0, 1, 2, 3].map((i) => deckSheets(i, 4))).toEqual([3, 2, 1, 0]);
    const many = Array.from({ length: 64 }, (_, i) => deckSheets(i, 64));
    expect(many[0]).toBe(DECK_SHEETS);
    expect(many[63]).toBe(0);
    expect(many.slice(0, 63).every((n) => n >= 1)).toBe(true);
    expect(many.every((n, i) => i === 0 || n <= many[i - 1])).toBe(true);
  });

  it("keeps the deck's room whichever slice shows", () => {
    const projected = cardSize(SHAPE, 10, "C");
    const sliced = cardSize(SHAPE, 10, "C", [], { projection: false });
    expect(sliced.h - projected.h).toBe(2 * 20 + DECK_ROOM_PX);
    expect(cardShapeView(SHAPE, { projection: false })).toEqual({ projection: false });
  });
});

describe("projected tile controls and placement", () => {
  it("reserves a row for each narrow hidden tile axis, matching the worker footprint", () => {
    const cfg = { projection: true, tile: [1, 2, 4, 3] };
    const view = cardShapeView(SHAPE, cfg)!;
    expect(view).toEqual({ projection: true, tilePositions: [0, 1] });
    expect(cardSize(SHAPE, 10, "X", [], cfg).h - cardSize(SHAPE, 10, "X").h).toBe(40);
    expect(cardSize(SHAPE, 10, "X", [], view)).toEqual(cardSize(SHAPE, 10, "X", [], cfg));
    expect(cardShapeView(SHAPE, { ...cfg, tile: [2, 2, 4, 3] })).toEqual({ projection: true, tilePositions: [1] });
  });

  it("keeps the footprint stable when tile positions or extents change but the controls stay", () => {
    const first = { projection: true, tile: [1, 1, 4, 3], sliders: [0, 0, 0, 0] };
    const next = { ...first, tile: [1, 2, 8, 6], sliders: [1, 3, 0, 0] };
    expect(cardShapeView(SHAPE, first)).toEqual(cardShapeView(SHAPE, next));
    expect(cardShapeView(SHAPE, { projection: true, tile: SHAPE })).toBeNull();
    expect(cardShapeView(SHAPE, { ...next, axes: [0, 1] })).toEqual({ projection: true, axes: [0, 1] });
  });
});

describe("hidden readings in the workspace", () => {
  const S = () => useStore.getState();
  beforeEach(() => {
    S().applyDSL("B = 2\nH = 4\nM = 8\nN = 6\nX = Tensor(B, H, M, N, dtype=fp16)\nY = relu(X)\n");
  });

  it("stores an unfold as a slice-free reading and clears it on the way back", () => {
    S().setViewCfg("X", hiddenViewCfg(4));
    expect(S().viewCfgs.X).toMatchObject({ projection: false, unfold: "grid" });
    S().setViewCfg("X", hiddenViewCfg(0));
    expect(S().viewCfgs.X.projection).toBe(true);
    expect("unfold" in S().viewCfgs.X).toBe(false);
  });
});
