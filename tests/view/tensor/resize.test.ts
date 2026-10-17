import { describe, expect, it } from "vitest";
import { canvasSize, cardCanResize, cardScaleFor, cardShapeView, cardSize, scaleRatioLabel } from "../../../src/view/tensor/card-size";
import { gridGeometry } from "../../../src/view/tensor/grid";
import { resizeScale } from "../../../src/view/tensor/resize";
import { defaultViewCfg, MAX_CARD_SCALE, MIN_CARD_SCALE, viewCfgFits } from "../../../src/view/tensor/tensor-view";
import { cardDisplayMenu } from "../../../src/view/tensor/menus";

describe("manual card sizing", () => {
  it.each([[], [1], [1, 1], [8, 1, 1]].map((shape) => [shape]))("detects a fixed canvas for %j", (shape) => {
    const cfg = defaultViewCfg(shape);
    expect(cardCanResize(shape, cfg, 4)).toBe(false);
    const spec = cardDisplayMenu(shape, cfg, 4, true, false);
    expect(spec.find((e) => e.kind === "action" && e.id === "resize")).toMatchObject({ disabled: true, title: "this view has a fixed canvas size" });
    expect(spec.find((e) => e.kind === "action" && e.id === "setSize")).toMatchObject({ disabled: true });
  });
  it("does not state a stored scale that a fixed canvas cannot apply", () => {
    const shape = [8, 1, 1], cfg = { ...defaultViewCfg(shape), scale: 2 };
    expect(scaleRatioLabel(shape, cfg, 4)).toBeNull();
    expect(scaleRatioLabel(shape, { ...cfg, axes: [0, 2] }, 4)).toBe("×2");
  });
  it("permits sizing a line and tests the actual plane of remapped and unfolded views", () => {
    expect(cardCanResize([1, 16], undefined, 4)).toBe(true);
    const shape = [8, 1, 1], cfg = defaultViewCfg(shape);
    expect(cardCanResize(shape, { ...cfg, axes: [0, 2] }, 4)).toBe(true);
    expect(cardCanResize(shape, { ...cfg, projection: false, unfold: "across" }, 4)).toBe(false);
  });
  it("only offers making room when the card overlaps a node and placement is ready", () => {
    const cfg = { ...defaultViewCfg([8, 16]), scale: 4 };
    const action = (crowded: boolean, ready: boolean) => cardDisplayMenu([8, 16], cfg, 4, true, false, crowded, ready)
      .find((e) => e.kind === "action" && e.id === "makeRoom");
    expect(action(true, true)).toMatchObject({ disabled: false });
    expect(action(true, false)).toMatchObject({ disabled: true });
    expect(action(false, true)).toMatchObject({ disabled: true });
  });
  it("uses the fitted scale as its starting point, including unfolded panels", () => {
    const shape = [16, 128, 128];
    const cfg = { ...defaultViewCfg(shape), projection: false, unfold: "across" as const };
    expect(cardScaleFor(shape, { ...cfg, scale: 2 }, 10)).toBe(cardScaleFor(shape, cfg, 10) * 2);
    const bigger = canvasSize(shape, { ...cfg, scale: 2 }, 10);
    const normal = canvasSize(shape, cfg, 10);
    expect(bigger.w).toBeGreaterThan(normal.w);
    // Canvas dimensions round to whole pixels after scale is applied.
    expect(Math.abs(bigger.h - normal.h * 2)).toBeLessThanOrEqual(1);
    // Manual dimensions are projected onto the scene without re-running Dagre.
    expect(cardShapeView(shape, cfg)).toEqual(cardShapeView(shape, { ...cfg, scale: 2 }));
  });
  it("leaves gesture units unchanged while enlarging their drawn cells", () => {
    const shape = [128, 128], cfg = defaultViewCfg(shape);
    const normal = gridGeometry(shape, cfg, 0, 2);
    const bigger = gridGeometry(shape, { ...cfg, scale: 2 }, 0, 2);
    expect([bigger.rowTile, bigger.colTile]).toEqual([normal.rowTile, normal.colTile]);
    expect(bigger.cellW).toBe(normal.cellW * 2);
  });
  it("projects either corner along the canvas diagonal and bounds extreme drags", () => {
    expect(resizeScale(1, { w: 200, h: 100 }, { x: 100, y: 50 }, "bottom")).toBe(1.5);
    expect(resizeScale(1, { w: 200, h: 100 }, { x: 100, y: -50 }, "top")).toBe(1.5);
    expect(resizeScale(1, { w: 200, h: 100 }, { x: -1e6, y: -1e6 }, "bottom")).toBe(MIN_CARD_SCALE);
    expect(resizeScale(1, { w: 200, h: 100 }, { x: 1e6, y: 1e6 }, "bottom")).toBe(MAX_CARD_SCALE);
  });
  it("keeps requested scales to three decimals", () => {
    expect(resizeScale(1, { w: 300, h: 0 }, { x: 100, y: 0 }, "bottom")).toBe(1.333);
  });
  it.each([NaN, Infinity, 0, -1, 0.01, 20])("rejects an invalid saved scale %s", (scale) => {
    expect(viewCfgFits([8, 8], { ...defaultViewCfg([8, 8]), scale })).toBe(false);
  });
  it("states a card's ratio to the graph's scale only where it differs", () => {
    const shape = [8, 8], cfg = defaultViewCfg(shape);
    expect(scaleRatioLabel(shape, cfg, 2)).toBeNull();
    expect(scaleRatioLabel(shape, { ...cfg, scale: 1.5 }, 2)).toBe("×1.5");
    expect(scaleRatioLabel(shape, { ...cfg, scale: 0.25 }, 2)).toBe("÷4");
    expect(scaleRatioLabel(shape, { ...cfg, scale: 1.0004 }, 2)).toBeNull();
  });
  it.each([
    ["default", [8, 16], {}],
    ["sliced", [4, 8, 16], { projection: false }],
    ["unfolded", [4, 8, 16], { projection: false, unfold: "across" as const }],
    ["remapped", [64, 4, 512], { axes: [0, 2] as [number, number] }],
  ])("measures a %s card at scale 1 as placement does, and grows it monotonically", (_name, shape, view) => {
    const cfg = { ...defaultViewCfg(shape), ...view };
    const size = (scale?: number) => cardSize(shape, 4, "T", [], { ...cfg, scale });
    expect(size(1)).toEqual(cardSize(shape, 4, "T", [], cardShapeView(shape, cfg) ?? undefined));
    let last = size(1);
    for (let scale = 1.001; scale <= MAX_CARD_SCALE; scale += 0.25) {
      const next = size(scale);
      expect(next.w).toBeGreaterThanOrEqual(last.w);
      expect(next.h).toBeGreaterThanOrEqual(last.h);
      last = next;
    }
  });
  it("describes scale as menu information and explains unavailable resizing", () => {
    const cfg = { ...defaultViewCfg([8, 8]), scale: 2 };
    expect(cardDisplayMenu([8, 8], cfg, 2, true, false)[0]).toMatchObject({ kind: "info", label: "display scale ×2" });
    expect(cardDisplayMenu([8, 8], cfg, 2, false, false)[1]).toMatchObject({ id: "resize", disabled: true });
  });
});
