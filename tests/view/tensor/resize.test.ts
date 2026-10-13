import { describe, expect, it } from "vitest";
import { canvasSize, cardScaleFor, cardShapeView, cardSize, scaleRatioLabel } from "../../../src/view/tensor/card-size";
import { gridGeometry, tileOf } from "../../../src/view/tensor/grid";
import { constrainResize, displayedCardScales, resizeScale } from "../../../src/view/tensor/resize";
import { defaultViewCfg, MAX_CARD_SCALE, MIN_CARD_SCALE, viewCfgFits } from "../../../src/view/tensor/tensor-view";
import { cardDisplayMenu } from "../../../src/view/tensor/menus";

describe("manual card sizing", () => {
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
    expect(tileOf(shape, 0, 2, { ...cfg, scale: 2 } as typeof cfg)).toBe(tileOf(shape, 0, 2, cfg));
  });
  it("projects either corner along the canvas diagonal and bounds extreme drags", () => {
    expect(resizeScale(1, { w: 200, h: 100 }, { x: 100, y: 50 }, "bottom")).toBe(1.5);
    expect(resizeScale(1, { w: 200, h: 100 }, { x: 100, y: -50 }, "top")).toBe(1.5);
    expect(resizeScale(1, { w: 200, h: 100 }, { x: -1e6, y: -1e6 }, "bottom")).toBe(MIN_CARD_SCALE);
    expect(resizeScale(1, { w: 200, h: 100 }, { x: 1e6, y: 1e6 }, "bottom")).toBe(MAX_CARD_SCALE);
  });
  it("stops a single large request at a neighbour and permits shrinking away", () => {
    const measure = (s: number) => ({ x: 0, y: 0, w: 100 * s, h: 50 * s });
    const blockers = [{ x: 150, y: 0, w: 20, h: 100 }];
    expect(constrainResize(1, 8, measure, blockers)).toBeCloseTo(1.5, 6);
    expect(constrainResize(1, 0.5, measure, blockers)).toBe(0.5);
  });
  it("keeps requested and constrained scales to three decimals", () => {
    expect(resizeScale(1, { w: 300, h: 0 }, { x: 100, y: 0 }, "bottom")).toBe(1.333);
    const measure = (s: number) => ({ x: 0, y: 0, w: 100 * s, h: 50 * s });
    const blockers = [{ x: 133.37, y: 0, w: 20, h: 100 }];
    expect(constrainResize(1, 2, measure, blockers)).toBe(1.333);
    expect(constrainResize(1.0004, 2, (s) => ({ x: 0, y: 0, w: 100 * s, h: 50 * s }), [{ x: 100.05, y: 0, w: 20, h: 100 }])).toBe(1.0004);
  });
  it("allows reducing existing overlap but never increases it", () => {
    const measure = (s: number) => ({ x: 0, y: 100 - 100 * s, w: 100 * s, h: 100 * s });
    const blockers = [{ x: 75, y: 0, w: 100, h: 100 }];
    expect(constrainResize(1, 0.5, measure, blockers)).toBe(0.5);
    expect(constrainResize(1, 2, measure, blockers)).toBeCloseTo(1, 6);
  });
  it("draws a resized card at its stored scale while it has room, and smaller where it has not", () => {
    const nodes = [{ key: "t:A", x: 0, y: 0, w: 100, h: 50 }, { key: "op", x: 200, y: 0, w: 40, h: 50 }];
    const measure = (_key: string, s: number) => ({ x: 0, y: 0, w: 100 * s, h: 50 * s });
    expect(displayedCardScales(nodes, { "t:A": 1.5 }, measure)).toEqual({ "t:A": 1.5 });
    expect(displayedCardScales(nodes, { "t:A": 3 }, measure)).toEqual({ "t:A": 2 });
    expect(displayedCardScales(nodes, { "t:A": 0.5 }, measure)).toEqual({ "t:A": 0.5 });
  });
  it("gives contested room to resized cards in key order", () => {
    const nodes = [{ key: "t:A", x: 0, y: 0, w: 100, h: 50 }, { key: "t:B", x: 150, y: 0, w: 100, h: 50 }];
    const measure = (key: string, s: number) => ({ x: key === "t:A" ? 0 : 150, y: 0, w: 100 * s, h: 50 * s });
    // A grows to B's fitted card; B then grows from where it stands.
    expect(displayedCardScales(nodes, { "t:B": 2, "t:A": 2 }, measure)).toEqual({ "t:A": 1.5, "t:B": 2 });
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
  it("says when neighbours leave a card less room than it was resized to", () => {
    const shown = { ...defaultViewCfg([8, 8]), scale: 1.25 };
    const [info] = cardDisplayMenu([8, 8], shown, 2, true, false, { ...shown, scale: 2 });
    expect(info).toMatchObject({ kind: "info", label: "display scale ×1.25, limited by neighbours" });
    expect((info as { title: string }).title).toContain("Resized to ×2");
  });
  it("describes scale as menu information and explains unavailable resizing", () => {
    const cfg = { ...defaultViewCfg([8, 8]), scale: 2 };
    expect(cardDisplayMenu([8, 8], cfg, 2, true, false)[0]).toMatchObject({ kind: "info", label: "display scale ×2" });
    expect(cardDisplayMenu([8, 8], cfg, 2, false, false)[1]).toMatchObject({ id: "resize", disabled: true });
  });
});
