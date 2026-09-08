import { afterEach, describe, expect, it, vi } from "vitest";
import { box, count, fromBox, union } from "../core/region";
import {
  drawGrid, gridGeometry, latticeStride, MIN_LATTICE_PX, paintScale, type Layer,
} from "../ui/grid";
import { CARD_SURFACE } from "../ui/palette";

// Record the canvas commands at the browser boundary, including state restored
// after a hatch. No DOM rasterizer is needed to check screen-space contracts.
function recordingCanvas() {
  const strokes: { color: string; width: number; dash: number[]; path: number[][] }[] = [];
  const borders: number[] = [];
  // Stipple dots, so the entanglement texture can be checked for where it drew
  // rather than only that it ran.
  const arcs: { x: number; y: number; r: number }[] = [];
  const fills: { color: string; count: number }[] = [];
  let dash: number[] = [];
  // Points of the path being built, so a stroke can be checked for *where* it
  // drew and not only how. beginPath resets it, exactly as the canvas does.
  let path: number[][] = [];
  const stack: { color: string; width: number; alpha: number; dash: number[] }[] = [];
  const ctx = {
    strokeStyle: "", fillStyle: "", lineWidth: 1, globalAlpha: 1,
    setTransform() {}, clearRect() {}, fillRect() {},
    beginPath() { path = []; },
    arc(x: number, y: number, r: number) { arcs.push({ x, y, r }); },
    fill() { fills.push({ color: this.fillStyle, count: arcs.length }); },
    rect() {}, clip() {},
    moveTo(x: number, y: number) { path.push([x, y]); },
    lineTo() {},
    setLineDash(value: number[]) { dash = value; },
    save() { stack.push({ color: this.strokeStyle, width: this.lineWidth, alpha: this.globalAlpha, dash }); },
    restore() {
      const saved = stack.pop()!;
      this.strokeStyle = saved.color; this.lineWidth = saved.width;
      this.globalAlpha = saved.alpha; dash = saved.dash;
    },
    stroke() { strokes.push({ color: this.strokeStyle, width: this.lineWidth, dash, path: [...path] }); },
    strokeRect() { borders.push(this.lineWidth); },
  };
  const canvas = { width: 0, height: 0, getContext: () => ctx };
  return { canvas: canvas as unknown as HTMLCanvasElement, strokes, borders, arcs, fills };
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

  it("keeps lattice and border strokes one screen pixel at any scale", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    for (const scale of [0.3, 4]) {
      const { canvas, strokes, borders } = recordingCanvas();
      drawGrid(canvas, shape, cfg, geom, [], true, 1, scale);
      for (const stroke of strokes) expect(stroke.width * scale).toBeCloseTo(1);
      expect(borders[0] * scale).toBeCloseTo(1);
    }
  });

  /* The lattice is what a snapped edge lands on, so it may not disappear at a
     zoom where snapping is still active. It is drawn at a stride instead: the
     boundaries thin out, and the ones that remain are real ones. */
  it.each([
    ["an 8-cell card at 40%", [16, 16], 4, 0.4],
    ["an 8-cell card at 30%", [16, 16], 4, 0.3],
    ["a 16-cell card at 2%", [256, 256], 8, 0.02],
  ])("draws a strided lattice rather than nothing: %s", (_case, sh, tile, scale) => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const cardShape = sh as number[];
    const g = gridGeometry(cardShape, cfg, 0, tile as number);
    const { canvas, strokes } = recordingCanvas();
    drawGrid(canvas, cardShape, cfg, g, [], true, 1, scale as number);

    const lattice = strokes.find((stroke) => stroke.color === "rgba(255,255,255,0.10)");
    expect(lattice).toBeDefined();
    expect(lattice!.path.length).toBeGreaterThan(0);

    // Every drawn line sits on a tile boundary, so a snapped edge that lands on
    // a drawn line lands where the line says it does.
    for (const [x, y] of lattice!.path) {
      const onGrid = y === 0 ? x / g.cellW : y / g.cellH;
      expect(Math.abs(onGrid - Math.round(onGrid))).toBeLessThan(1e-9);
    }

    // And they are far enough apart on screen to be told apart.
    const xs = lattice!.path.filter(([, y]) => y === 0).map(([x]) => x).sort((a, b) => a - b);
    for (let i = 1; i < xs.length; i++)
      expect((xs[i] - xs[i - 1]) * (scale as number)).toBeGreaterThanOrEqual(MIN_LATTICE_PX);
  });

  it("draws no lattice only when the card itself is too small to divide", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    // 8 cells of 8 canvas px at 5%: the whole card is 3 screen px wide, so no
    // stride produces a boundary anyone could see.
    const { canvas, strokes } = recordingCanvas();
    drawGrid(canvas, shape, cfg, geom, [], true, 1, 0.05);
    expect(strokes.find((stroke) => stroke.color === "rgba(255,255,255,0.10)")).toBeUndefined();
  });

  it("strides by powers of two, so a drawn line is never off the tile grid", () => {
    for (const scale of [0.05, 0.15, 0.4, 1, 4]) {
      const stride = latticeStride(geom.cellW, 64, scale);
      expect(Number.isInteger(Math.log2(stride))).toBe(true);
      if (stride > 1) {
        // Smallest stride that clears the threshold: half of it would not.
        expect(geom.cellW * stride * scale).toBeGreaterThanOrEqual(MIN_LATTICE_PX);
        expect(geom.cellW * (stride / 2) * scale).toBeLessThan(MIN_LATTICE_PX);
      }
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


/**
 * The entanglement texture at the canvas boundary.
 *
 * `buildLayers` decides that a stipple is requested; only this reaches the
 * renderer, where the mark either exists and is made of dots or does not. The
 * distinction from a ruling has to hold here, not in the layer description.
 */
describe("stipple rendering", () => {
  const stippleLayer = (region = fromBox(box([0, 8], [0, 8]))): Layer => ({
    region,
    color: [20, 120, 220],
    alpha: 1,
    hatch: false,
    pattern: { kind: "stipple", density: 0.5 },
  });

  /** The lattice is stroked whatever the layers are; only layer marks matter. */
  const layerStrokes = (strokes: { color: string; path: number[][] }[]) =>
    strokes.filter((s) => s.path.length > 0 && s.color.includes("20,120,220"));

  it("draws dots, and no lines that could read as a ruling", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, arcs, strokes } = recordingCanvas();
    drawGrid(canvas, shape, cfg, geom, [stippleLayer()], false, 1, 1);
    expect(arcs.length).toBeGreaterThan(4);
    // A ruling strokes; a stipple fills. The layer must contribute no line, or
    // the two encodings become confusable at a glance.
    expect(layerStrokes(strokes)).toHaveLength(0);
  });

  it("keeps every dot inside the region it describes", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, arcs } = recordingCanvas();
    drawGrid(canvas, shape, cfg, geom, [stippleLayer(fromBox(box([4, 12], [4, 12])))], false, 1, 1);
    expect(arcs.length).toBeGreaterThan(0);
    // Pixels per *element*, not per drawn cell: a cell may span several
    // elements, and the region is stated in elements.
    const pxX = geom.canvasW / geom.cols;
    const pxY = geom.canvasH / geom.rows;
    for (const a of arcs) {
      expect(a.x).toBeGreaterThanOrEqual(4 * pxX);
      expect(a.x).toBeLessThanOrEqual(12 * pxX);
      expect(a.y).toBeGreaterThanOrEqual(4 * pxY);
      expect(a.y).toBeLessThanOrEqual(12 * pxY);
    }
  });

  it("degrades to a flat fill where no pattern fits, as a ruling does", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const { canvas, arcs } = recordingCanvas();
    // Zoomed far out, the region's screen extent falls under the shared
    // pattern floor, so direction degrades to alpha rather than drawing a mark
    // too small to read as anything.
    drawGrid(canvas, shape, cfg, geom, [stippleLayer(fromBox(box([0, 1], [0, 1])))], false, 1, 0.1);
    expect(arcs).toHaveLength(0);
  });
});
