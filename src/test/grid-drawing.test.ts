import { afterEach, describe, expect, it, vi } from "vitest";
import { box, count, fromBox, union } from "../core/region";
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

describe("seed annotations", () => {
  it("draws seed corners after cone paint and distinguishes them from dependency layers", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, strokes } = recordingCanvas();
    const layer: Layer = { region: fromBox(box([4, 5], [4, 5])), color: [10, 100, 200], alpha: 0.9, hatch: false, seed: true };
    drawGrid(canvas, shape, cfg, geom, [layer], true, 1, 0.3);
    const marks = strokes.slice(-2);
    expect(marks.map((stroke) => stroke.color)).toEqual([CARD_SURFACE.dark, "rgb(10,100,200)"]);
    expect(marks[1].width * 0.3).toBeCloseTo(1);
    const ordinary = recordingCanvas();
    drawGrid(ordinary.canvas, shape, cfg, geom, [{ ...layer, seed: false }], true, 1, 0.3);
    expect(ordinary.strokes).toHaveLength(strokes.length - 2);
  });
});


describe("overlapping boxes are painted once", () => {
  /** Record every filled rect, not just strokes. */
  function fillRecorder() {
    const fills: { x: number; y: number; w: number; h: number; style: string }[] = [];
    const rects: { x: number; y: number; w: number; h: number; width: number }[] = [];
    const ctx = {
      strokeStyle: "", fillStyle: "", lineWidth: 1, globalAlpha: 1,
      setTransform() {}, clearRect() {}, beginPath() {}, rect() {}, clip() {},
      moveTo() {}, lineTo() {}, setLineDash() {}, save() {}, restore() {},
      stroke() {},
      strokeRect(x: number, y: number, w: number, h: number) {
        rects.push({ x, y, w, h, width: this.lineWidth });
      },
      fillRect(x: number, y: number, w: number, h: number) {
        fills.push({ x, y, w, h, style: String(this.fillStyle) });
      },
    };
    const canvas = { width: 0, height: 0, getContext: () => ctx };
    return { canvas: canvas as unknown as HTMLCanvasElement, fills, rects };
  }

  // The `matmul(A, A)` shape at card scale: a row band and a column band that
  // share a square. The region stores both whole.
  const bands = union(fromBox(box([4, 8], [0, 16])), fromBox(box([0, 16], [2, 6])));

  it("keeps the region's own boxes overlapping", () => {
    expect(bands.boxes).toHaveLength(2);
    expect(count(bands)).toBeLessThan(
      bands.boxes.reduce((a, b) => a + b.reduce((v, i) => v * (i.hi - i.lo), 1), 0)
    );
  });

  it("emits fill rects that do not overlap each other", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, fills } = fillRecorder();
    const layer: Layer = { region: bands, color: [10, 20, 30], alpha: 0.72, hatch: false };
    drawGrid(canvas, shape, cfg, geom, [layer], true);
    // drop the opening surface wash, which covers the whole card by design
    const rects = fills.filter((f) => !(f.x === 0 && f.y === 0 && f.w === geom.canvasW && f.h === geom.canvasH));
    expect(rects.length).toBeGreaterThan(0);
    for (let i = 0; i < rects.length; i++)
      for (let j = i + 1; j < rects.length; j++) {
        const a = rects[i], b = rects[j];
        const overlaps =
          a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
        expect(overlaps).toBe(false);
      }
    // Every rect carries one identical fill, alpha included: no element is
    // painted twice, so alpha cannot come to encode how many boxes cover a
    // point - the one channel this renderer deliberately holds constant.
    expect(new Set(rects.map((r) => r.style)).size).toBe(1);
    expect(rects[0].style).toBe("rgba(10,20,30,0.72)");
  });

  it("still covers the whole union", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, fills } = fillRecorder();
    const layer: Layer = { region: bands, color: [10, 20, 30], alpha: 0.72, hatch: false };
    drawGrid(canvas, shape, cfg, geom, [layer], true);
    const rects = fills.filter((f) => !(f.x === 0 && f.y === 0 && f.w === geom.canvasW && f.h === geom.canvasH));
    const painted = rects.reduce((a, r) => a + r.w * r.h, 0);
    const perElement = (geom.canvasW / shape[1]) * (geom.canvasH / shape[0]);
    expect(painted / perElement).toBeCloseTo(count(bands), 5);
  });
});

describe("a border traces the region, not the split", () => {
  function recorder() {
    const rects: { x: number; y: number; w: number; h: number; width: number }[] = [];
    const fills: { x: number; y: number; w: number; h: number }[] = [];
    const ctx = {
      strokeStyle: "", fillStyle: "", lineWidth: 1, globalAlpha: 1,
      setTransform() {}, clearRect() {}, beginPath() {}, rect() {},
      clip() {}, moveTo() {}, lineTo() {}, setLineDash() {}, save() {}, restore() {},
      stroke() {},
      fillRect(x: number, y: number, w: number, h: number) {
        fills.push({ x, y, w, h });
      },
      strokeRect(x: number, y: number, w: number, h: number) {
        rects.push({ x, y, w, h, width: this.lineWidth });
      },
    };
    const canvas = { width: 0, height: 0, getContext: () => ctx };
    return { canvas: canvas as unknown as HTMLCanvasElement, rects, fills };
  }

  const bands = union(fromBox(box([4, 8], [0, 16])), fromBox(box([0, 16], [2, 6])));

  it("emphasises two crossing bands rather than three fragments", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, rects } = recorder();
    const layer: Layer = {
      region: bands, color: [10, 20, 30], alpha: 0.72, hatch: false,
      outline: true, lineWidth: 1.5,
    };
    drawGrid(canvas, shape, cfg, geom, [layer], true);
    // the card border is drawn at its own weight; the emphasis is at 1.5
    const outlines = rects.filter((r) => r.width === 1.5);
    expect(outlines).toHaveLength(2);
    // the row band spans the full width, the column band the full height
    expect(outlines.some((r) => r.w > r.h)).toBe(true);
    expect(outlines.some((r) => r.h > r.w)).toBe(true);
    // and they overlap, which is the reading: one crosses the other
    const [a, b] = outlines;
    expect(a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h).toBe(true);
  });

  it("fills the split form under a border that traces the whole", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, rects, fills } = recorder();
    const layer: Layer = {
      region: bands, color: [10, 20, 30], alpha: 0.72, hatch: false,
      outline: true, lineWidth: 1.5,
    };
    drawGrid(canvas, shape, cfg, geom, [layer], true);
    const painted = fills.filter(
      (f) => !(f.x === 0 && f.y === 0 && f.w === geom.canvasW && f.h === geom.canvasH)
    );
    // The two facts the design rests on, in one place: the paint is split so
    // nothing composites twice, and the border is not, so the reader sees the
    // two regions an operand actually reads.
    expect(painted).toHaveLength(3);
    expect(rects.filter((r) => r.width === 1.5)).toHaveLength(2);
  });
});
