import { afterEach, describe, expect, it, vi } from "vitest";
import { box, fromBox } from "../core/region";
import { drawGrid, gridGeometry, paintScale, type Layer } from "../ui/grid";
import { CARD_SURFACE } from "../ui/palette";

// Record the canvas commands at the browser boundary, including state restored
// after a hatch. No DOM rasterizer is needed to check screen-space contracts.
function recordingCanvas() {
  const strokes: { color: string; width: number; dash: number[] }[] = [];
  const borders: number[] = [];
  let dash: number[] = [];
  const stack: { color: string; width: number; alpha: number; dash: number[] }[] = [];
  const ctx = {
    strokeStyle: "", fillStyle: "", lineWidth: 1, globalAlpha: 1,
    setTransform() {}, clearRect() {}, fillRect() {}, beginPath() {},
    rect() {}, clip() {}, moveTo() {}, lineTo() {},
    setLineDash(value: number[]) { dash = value; },
    save() { stack.push({ color: this.strokeStyle, width: this.lineWidth, alpha: this.globalAlpha, dash }); },
    restore() {
      const saved = stack.pop()!;
      this.strokeStyle = saved.color; this.lineWidth = saved.width;
      this.globalAlpha = saved.alpha; dash = saved.dash;
    },
    stroke() { strokes.push({ color: this.strokeStyle, width: this.lineWidth, dash }); },
    strokeRect() { borders.push(this.lineWidth); },
  };
  const canvas = { width: 0, height: 0, getContext: () => ctx };
  return { canvas: canvas as unknown as HTMLCanvasElement, strokes, borders };
}
const cfg = { sliders: [], projection: true };
const shape = [16, 16];
const geom = gridGeometry(shape, cfg, 0, 4);
afterEach(() => { vi.unstubAllGlobals(); });

describe("canvas screen-space rendering", () => {
  it("reuses backing dimensions when paint scale changes", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas } = recordingCanvas();
    drawGrid(canvas, shape, cfg, geom, [], true);
    const width = vi.fn(), height = vi.fn();
    Object.defineProperty(canvas, "width", { get: () => geom.canvasW, set: width });
    Object.defineProperty(canvas, "height", { get: () => geom.canvasH, set: height });
    drawGrid(canvas, shape, cfg, geom, [], true, 1, 0.3);
    expect(width).not.toHaveBeenCalled();
    expect(height).not.toHaveBeenCalled();
  });

  it("gates the lattice by screen cell size and keeps strokes one pixel", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    for (const scale of [0.3, 4]) {
      const { canvas, strokes, borders } = recordingCanvas();
      drawGrid(canvas, shape, cfg, geom, [], true, 1, scale);
      expect(strokes.length > 0).toBe(Math.min(geom.cellW, geom.cellH) * scale >= 5);
      for (const stroke of strokes) expect(stroke.width * scale).toBeCloseTo(1);
      expect(borders[0] * scale).toBeCloseTo(1);
    }
  });

  it.each([true, false])("contrasts approximate solid fills with the surface (dark=%s)", (dark) => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, strokes } = recordingCanvas();
    const layer: Layer = { region: fromBox(box([0, 16], [0, 16])), color: [10, 100, 200], alpha: 0.72, hatch: true };
    drawGrid(canvas, shape, cfg, geom, [layer], dark);
    const hatch = strokes.find((stroke) => stroke.dash.length)!;
    expect(hatch.color).toBe(dark ? CARD_SURFACE.dark : CARD_SURFACE.light);
  });

  it("shares paint buckets for tiny wheel changes with less than 2.2% scale error", () => {
    expect(paintScale(1.001)).toBe(paintScale(1.002));
    for (const scale of [0.03, 0.3, 0.99, 1, 1.1, 2.7, 4]) {
      expect(paintScale(scale)).toBeLessThanOrEqual(scale);
      expect(scale / paintScale(scale)).toBeLessThan(1.022);
    }
  });
});
