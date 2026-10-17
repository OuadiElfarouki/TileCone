import { seedCornerSegments } from "../../../src/components/card/draw-grid";
import { describe, expect, it } from "vitest";
import {
  AUTO_TILE_FRACTION,
  axisTile,
  cardPx,
  displayTile,
  graphScale,
  effectiveTileScaleIndex,
  effectiveTileScaleStops,
  MAX_ELEM_PX,
  MAX_GRAPH_W,
  MAX_GRAPH_H,
  planeExtents,
  MIN_SIDE_PX,
  pow2Floor,
  pow2Round,
  tileScaleLabel,
  TILE_SCALE_MAX,
  TILE_SCALE_MIN,
} from "../../../src/view/tensor/tiling";
import { gridGeometry, MIN_MARK_PX, regionRects } from "../../../src/view/tensor/grid";
import { outlineFitsRect, patternFitsRect, rulingSegments, stripePitchPx } from "../../../src/components/card/draw-grid";
import { box, fromBox } from "../../../src/core/region";
import { viewAxes, type ViewCfg } from "../../../src/view/tensor/tensor-view";

const cfg: ViewCfg = { sliders: [0, 0, 0, 0], projection: true };
const isPow2 = (n: number) => n >= 1 && (n & (n - 1)) === 0;

/** The scale a graph containing only this tensor would render at. Card geometry
 * is a property of the graph now, so every shape-level assertion needs one. */
const pxFor = (rows: number, cols: number) => graphScale([{ rows, cols }]);
const pxOf = (shape: number[]) => {
  const { rowAxis, colAxis } = viewAxes(shape);
  return graphScale([planeExtents(shape, rowAxis, colAxis)]);
};

describe("power-of-two helpers", () => {
  it("pow2Round snaps in log space", () => {
    expect(pow2Round(5)).toBe(4); // log2(5)=2.32 -> 2
    expect(pow2Round(10)).toBe(8); // log2(10)=3.32 -> 3
    expect(pow2Round(6)).toBe(8); // log2(6)=2.58 -> 3
    expect(pow2Round(0.2)).toBe(1); // clamped up
    expect(pow2Round(0)).toBe(1);
  });
  it("pow2Floor never exceeds its argument", () => {
    expect(pow2Floor(50)).toBe(32);
    expect(pow2Floor(64)).toBe(64);
    expect(pow2Floor(0.5)).toBe(1);
  });
});

describe("the display tile on an axis", () => {
  it("is about 5% of the axis, rounded to a power of two", () => {
    expect(axisTile(100, 0)).toBe(4); // 5 -> 4
    expect(axisTile(200, 0)).toBe(8); // 10 -> 8
    expect(axisTile(704, 0)).toBe(32); // 35.2 -> 32
    for (const extent of [100, 200, 214, 512, 1000, 4096]) {
      const t = axisTile(extent, 0);
      const ideal = AUTO_TILE_FRACTION * extent;
      expect(t).toBeGreaterThanOrEqual(ideal / 2);
      expect(t).toBeLessThanOrEqual(ideal * 2);
    }
  });

  it("is always a power of two that leaves at least two tiles", () => {
    for (const extent of [2, 3, 4, 7, 13, 100, 214, 4096]) {
      for (let k = TILE_SCALE_MIN; k <= TILE_SCALE_MAX; k++) {
        const t = axisTile(extent, k);
        expect(isPow2(t), `tile ${t} for ${extent} @${k}`).toBe(true);
        expect(Math.floor(extent / t), `${extent} @${k}`).toBeGreaterThanOrEqual(2);
      }
    }
  });

  it("is one element on an axis of one, and under none", () => {
    expect(axisTile(1, 3)).toBe(1);
    expect(axisTile(4096, TILE_SCALE_MIN)).toBe(1);
  });

  /* The two properties the rule exists for. */
  it("is the same for an axis of one length on every tensor, whatever the other axis", () => {
    // seq = 128 beside D = 32, and beside another 128: the square tile from
    // each tensor's smaller axis gave seq 2 on the first and 8 on the second.
    expect(displayTile([128, 32], 0)[0]).toBe(displayTile([128, 128], 0)[0]);
    expect(displayTile([4, 128, 32], 0)).toEqual([axisTile(4, 0), 8, 2]);
  });

  it("does not depend on the graph's scale, so an edit elsewhere never re-grids a tensor", () => {
    // W [256, 64] drawn at the scale of a graph whose largest tensor grows.
    const at = (px: number) => gridGeometry([256, 64], cfg, 0, px);
    expect([at(1.172).rowTile, at(1.172).colTile]).toEqual([at(0.219).rowTile, at(0.219).colTile]);
  });
});

describe("the global detail scale", () => {
  it("shifts every axis's tile by powers of two and stays monotone", () => {
    for (const extent of [64, 256, 704]) {
      let prev = 0;
      for (let k = TILE_SCALE_MIN; k <= TILE_SCALE_MAX; k++) {
        const t = axisTile(extent, k);
        expect(t).toBeGreaterThanOrEqual(prev);
        prev = t;
      }
      expect(axisTile(extent, TILE_SCALE_MAX)).toBe(pow2Floor(extent / 2));
    }
  });

  it("none is one element per tile even when boundaries are sub-pixel", () => {
    const geom = gridGeometry([256, 256], cfg, TILE_SCALE_MIN, pxFor(256, 256));
    expect([geom.rowTile, geom.colTile]).toEqual([1, 1]);
    expect(geom.tileRows).toBe(256);
    expect(geom.tileCols).toBe(256);
    expect(Math.min(geom.cellW, geom.cellH)).toBeLessThan(5);
  });

  it("is rectangular where the drawn axes differ in length", () => {
    const geom = gridGeometry([128, 704], cfg, 0, pxFor(128, 704));
    expect([geom.rowTile, geom.colTile]).toEqual([8, 32]);
  });

  it("reads as about how many tiles cross each axis, the same in every graph", () => {
    expect(tileScaleLabel(0)).toBe("≈20 per axis");
    expect(tileScaleLabel(1)).toBe("≈10 per axis");
    expect(tileScaleLabel(-1)).toBe("≈40 per axis");
    expect(tileScaleLabel(3)).toBe("≈2 per axis");
    expect(tileScaleLabel(TILE_SCALE_MAX)).toBe("≈2 per axis");
    expect(tileScaleLabel(TILE_SCALE_MIN)).toBe("elements");
  });

  it("offers only settings that change the tiling of some axis the graph draws", () => {
    const planes = [
      { rows: 256, cols: 512 },
      { rows: 512, cols: 256 },
      { rows: 256, cols: 256 },
    ];
    const stops = effectiveTileScaleStops(planes);
    // Past 3 both lengths sit at half their axis, so 4 and 5 add nothing.
    expect(stops).toEqual([TILE_SCALE_MIN, -4, -3, -2, -1, 0, 1, 2, 3]);
    const state = (scale: number) => [256, 512].map((extent) => axisTile(extent, scale));
    for (let i = 1; i < stops.length; i++) expect(state(stops[i])).not.toEqual(state(stops[i - 1]));
  });

  it("maps legacy plateau values onto their equivalent visible stop", () => {
    const planes = [{ rows: 256, cols: 512 }];
    const stops = effectiveTileScaleStops(planes);
    expect(stops[effectiveTileScaleIndex(planes, stops, TILE_SCALE_MIN)]).toBe(TILE_SCALE_MIN);
    expect(stops[effectiveTileScaleIndex(planes, stops, TILE_SCALE_MAX)]).toBe(3);
  });

  it("does not confuse the default with none when both happen to be one element", () => {
    const planes = [{ rows: 8, cols: 8 }];
    const stops = effectiveTileScaleStops(planes);
    expect(stops).toContain(TILE_SCALE_MIN);
    expect(stops).toContain(0);
    expect(stops[effectiveTileScaleIndex(planes, stops, TILE_SCALE_MIN)]).toBe(TILE_SCALE_MIN);
    expect(stops[effectiveTileScaleIndex(planes, stops, 0)]).toBe(0);
  });
});

describe("card sizing is independent of the detail setting", () => {
  const SHAPES = [
    [256, 256], [256, 512], [100, 200], [4, 4], [64, 64], [4096, 4096], [1, 512], [200, 214],
  ];

  it("the on-screen grid never changes size when detail changes", () => {
    for (const [r, c] of SHAPES) {
      const base = gridGeometry([r, c], cfg, 0, pxFor(r, c));
      for (let k = TILE_SCALE_MIN; k <= TILE_SCALE_MAX; k++) {
        const g = gridGeometry([r, c], cfg, k, pxFor(r, c));
        expect(g.canvasW, `${r}x${c} @${k} width`).toBe(base.canvasW);
        expect(g.canvasH, `${r}x${c} @${k} height`).toBe(base.canvasH);
      }
    }
  });

  it("only the lattice density changes", () => {
    const fine = gridGeometry([256, 256], cfg, -2, pxFor(256, 256));
    const coarse = gridGeometry([256, 256], cfg, 2, pxFor(256, 256));
    expect(fine.canvasW).toBe(coarse.canvasW);
    expect(fine.tileCols).toBeGreaterThan(coarse.tileCols);
    expect(fine.cellW).toBeLessThan(coarse.cellW);
  });

  it("card size follows the tensor's shape and aspect ratio", () => {
    const square = cardPx(256, 256, pxFor(256, 256));
    expect(square.w).toBe(square.h);
    const wide = cardPx(256, 512, pxFor(256, 512));
    expect(wide.w / wide.h).toBeCloseTo(2, 1);
    // a small tensor gets a small card, not a full-size one
    expect(cardPx(4, 4, pxFor(4, 4)).w).toBeLessThan(square.w);
  });

  it("every card is drawable", () => {
    for (const [r, c] of SHAPES) {
      const { w, h } = cardPx(r, c, pxFor(r, c));
      expect(w, `${r}x${c} width`).toBeGreaterThan(0);
      expect(h, `${r}x${c} height`).toBeGreaterThan(0);
    }
  });

  it("the largest tensor fits the graph budget unless a floor overrides it", () => {
    // There is no per-card cap any more: MAX_GRAPH_* is the budget for the whole
    // graph, and the visibility floors are allowed to beat it. A tensor nobody
    // can see is worse than one that needs a pan.
    const planes = SHAPES.map(([rows, cols]) => ({ rows, cols }));
    const px = graphScale(planes);
    const maxR = Math.max(...planes.map((p) => p.rows));
    const maxC = Math.max(...planes.map((p) => p.cols));
    const budgeted = px <= Math.min(MAX_GRAPH_W / maxC, MAX_GRAPH_H / maxR);
    const flooredByVisibility = px >= MIN_SIDE_PX / Math.min(...planes.flatMap((p) =>
      [p.rows, p.cols].filter((d) => d > 1)));
    expect(budgeted || flooredByVisibility).toBe(true);
    expect(px).toBeLessThanOrEqual(MAX_ELEM_PX);
  });
});

describe("one scale for the whole graph", () => {
  it("a dimension two tensors share is drawn at the same length in both", () => {
    // The reason the scale is global at all: for C[M,N] = A[M,K] @ B[K,N] the
    // contraction K must read as one axis, not as two different lengths.
    const planes = [
      { rows: 256, cols: 512 }, // A [M, K]
      { rows: 512, cols: 256 }, // B [K, N]
      { rows: 256, cols: 256 }, // C [M, N]
    ];
    const px = graphScale(planes);
    const A = cardPx(256, 512, px);
    const B = cardPx(512, 256, px);
    const C = cardPx(256, 256, px);
    expect(A.w, "K along A's columns == K down B's rows").toBe(B.h);
    expect(A.h, "M down A's rows == M down C's rows").toBe(C.h);
    expect(B.w, "N along B's columns == N along C's columns").toBe(C.w);
  });

  it("an axis's tile is drawn at one length on every card that draws the axis", () => {
    // With one scale and a tile per axis length, the 256 axis is ruled alike
    // as A's rows and as B's columns: the cells line up across cards.
    const planes = [{ rows: 256, cols: 512 }, { rows: 512, cols: 256 }];
    const px = graphScale(planes);
    const a = gridGeometry([256, 512], cfg, 0, px);
    const b = gridGeometry([512, 256], cfg, 0, px);
    expect(a.cellH).toBeCloseTo(b.cellW, 6);
    expect(a.cellW).toBeCloseTo(b.cellH, 6);
  });

  it("a degenerate axis does not drag the whole graph to the cap", () => {
    // A rank-1 bias vector has rows === 1. Counting that as the smallest side
    // would demand 14px for it and peg every card at MAX_ELEM_PX - one bias
    // would blow a 64x64 tensor up from 300px to 896px.
    const withoutBias = graphScale([{ rows: 64, cols: 64 }]);
    const withBias = graphScale([{ rows: 64, cols: 64 }, { rows: 1, cols: 64 }]);
    expect(withBias).toBe(withoutBias);
    expect(withBias).toBeLessThan(MAX_ELEM_PX);
  });

  it("an all-degenerate graph still resolves to a usable scale", () => {
    const px = graphScale([{ rows: 1, cols: 1 }]);
    expect(px).toBeGreaterThan(0);
    expect(Number.isFinite(px)).toBe(true);
  });
});

describe("regions are drawn at element precision, not tile precision", () => {
  const rectsOf = (region: Parameters<typeof regionRects>[0], shape: number[], c = cfg) =>
    regionRects(region, shape, c, gridGeometry(shape, c, 0, pxOf(shape)));

  it("a box maps to its exact fraction of the canvas", () => {
    const shape = [16, 16];
    const geom = gridGeometry(shape, cfg, 0, pxOf(shape));
    const [rect] = regionRects(fromBox(box([4, 8], [0, 16])), shape, cfg, geom);
    expect(rect.x).toBeCloseTo(0, 5);
    expect(rect.w).toBeCloseTo(geom.canvasW, 5);
    expect(rect.y).toBeCloseTo(geom.canvasH * (4 / 16), 5);
    expect(rect.h).toBeCloseTo(geom.canvasH * (4 / 16), 5);
  });

  it("does not round a range out to the tile lattice", () => {
    // The whole point: an unsnapped 3-element range must not paint a whole
    // 8-element cell, nor half-light one.
    const shape = [64, 64];
    const geom = gridGeometry(shape, cfg, 2, pxOf(shape));
    expect(geom.rowTile).toBeGreaterThan(1); // cells aggregate several elements
    const [rect] = regionRects(fromBox(box([3, 6], [0, 64])), shape, cfg, geom);
    expect(rect.y).toBeCloseTo(geom.canvasH * (3 / 64), 5);
    expect(rect.h).toBeCloseTo(geom.canvasH * (3 / 64), 5);
  });

  it("paints a region at full strength regardless of tile alignment", () => {
    // Coverage shading used to make a partly-covered cell translucent, which
    // read as "partly selected" when the truth was "these exact elements".
    const shape = [64, 64];
    for (const b of [box([0, 8], [0, 8]), box([1, 3], [5, 6])])
      for (const rect of rectsOf(fromBox(b), shape)) expect(rect.alpha).toBe(1);
  });

  it("keeps a sub-pixel region visible rather than losing it", () => {
    const shape = [4096, 4096];
    const rects = rectsOf(fromBox(box([0, 1], [0, 1])), shape);
    expect(rects[0].w).toBeGreaterThanOrEqual(MIN_MARK_PX);
    expect(rects[0].h).toBeGreaterThanOrEqual(MIN_MARK_PX);
  });

  it("never paints outside the canvas", () => {
    const shape = [16, 16];
    const geom = gridGeometry(shape, cfg, 0, pxOf(shape));
    for (const rect of regionRects(fromBox(box([15, 16], [15, 16])), shape, cfg, geom)) {
      expect(rect.x + rect.w).toBeLessThanOrEqual(geom.canvasW + 1e-6);
      expect(rect.y + rect.h).toBeLessThanOrEqual(geom.canvasH + 1e-6);
    }
  });

  it("hidden axes stay fractional, because they are not on screen", () => {
    const shape = [4, 8, 8]; // axis 0 hidden, projected
    const full = rectsOf(fromBox(box([0, 4], [0, 8], [0, 8])), shape);
    const quarter = rectsOf(fromBox(box([0, 1], [0, 8], [0, 8])), shape);
    expect(full[0].alpha).toBeCloseTo(1, 5);
    expect(quarter[0].alpha).toBeCloseTo(0.25, 5);
    // the drawn plane is identical; only the unseen axis differs
    expect(quarter[0].w).toBeCloseTo(full[0].w, 5);
  });

  it("slice mode shows only the current index", () => {
    const shape = [4, 8, 8];
    const sliceCfg = { sliders: [2, 0, 0], projection: false };
    expect(rectsOf(fromBox(box([2, 3], [0, 8], [0, 8])), shape, sliceCfg)).toHaveLength(1);
    expect(rectsOf(fromBox(box([0, 1], [0, 8], [0, 8])), shape, sliceCfg)).toHaveLength(0);
  });

  it("an empty region paints nothing", () => {
    expect(rectsOf({ boxes: [], exact: true, reasons: [] }, [16, 16])).toEqual([]);
  });
});

describe("thin-region paint fallbacks", () => {
  it("drops an outline that would be wider than the mark", () => {
    expect(outlineFitsRect({ w: 1, h: 20 }, 1.5, 1)).toBe(false);
    expect(outlineFitsRect({ w: 8, h: 20 }, 1.5, 1)).toBe(true);
  });

  it("uses solid paint only when the mark is too thin on screen", () => {
    expect(patternFitsRect({ w: 1, h: 20 }, 1)).toBe(false);
    expect(patternFitsRect({ w: 20, h: 20 }, 0.5)).toBe(true);
    expect(patternFitsRect({ w: 100, h: 100 }, 0.1)).toBe(true);
    expect(patternFitsRect({ w: 20, h: 20 }, 0.1)).toBe(false);
  });

  it("delimits a ruled region wherever it rules it", () => {
    // The hairline and the ruling share a 3px floor, so a rect never draws one
    // without the other - a ruled fill is never left without its edge.
    for (const rect of [{ w: 3, h: 20 }, { w: 20, h: 20 }, { w: 2, h: 20 }])
      expect(outlineFitsRect(rect, 0.75, 1)).toBe(patternFitsRect(rect, 1));
  });
});

describe("downstream density is spacing", () => {
  it("closes the gap as the share grows", () => {
    expect(stripePitchPx(1)).toBeLessThan(stripePitchPx(0.5));
    expect(stripePitchPx(0.5)).toBeLessThan(stripePitchPx(0));
  });

  it("stays a ruling at both ends of the scale", () => {
    // Wide enough at the floor to read as separate lines, tight enough at the
    // ceiling to read as dense - but never so tight it fills in as solid.
    expect(stripePitchPx(0)).toBeLessThanOrEqual(12);
    expect(stripePitchPx(1)).toBeGreaterThanOrEqual(2.5);
  });

  it("clamps out-of-range densities instead of inverting the ruling", () => {
    expect(stripePitchPx(-1)).toBe(stripePitchPx(0));
    expect(stripePitchPx(4)).toBe(stripePitchPx(1));
  });
});

describe("ruled fills", () => {
  const rect = { x: 0, y: 0, w: 40, h: 40 };
  const perpendicular = (a: { x1: number; y1: number; x2: number; y2: number }, pitch: number) => {
    // Distance between neighbouring lines, measured along their shared normal.
    const dx = a.x2 - a.x1;
    const dy = a.y2 - a.y1;
    const len = Math.hypot(dx, dy);
    return { nx: -dy / len, ny: dx / len, pitch };
  };

  it("spaces lines at the requested perpendicular pitch", () => {
    for (const angle of [15, 75, 135, 165]) {
      const segs = rulingSegments(rect, angle, 5);
      expect(segs.length, `angle ${angle}`).toBeGreaterThan(1);
      const { nx, ny } = perpendicular(segs[0], 5);
      const offsets = segs.map((s) => s.x1 * nx + s.y1 * ny);
      for (let i = 1; i < offsets.length; i++)
        expect(Math.abs(offsets[i] - offsets[i - 1]), `angle ${angle}`).toBeCloseTo(5, 6);
    }
  });

  it("anchors phase to the canvas, so neighbouring regions stay on one ruling", () => {
    // Two rects side by side on the same card must not each restart the pattern:
    // a seam at the boundary would read as a change in the encoded quantity.
    const left = rulingSegments({ x: 0, y: 0, w: 20, h: 40 }, 135, 5);
    const right = rulingSegments({ x: 20, y: 0, w: 20, h: 40 }, 135, 5);
    const { nx, ny } = perpendicular(left[0], 5);
    const offsetOf = (s: { x1: number; y1: number }) => s.x1 * nx + s.y1 * ny;
    for (const s of right) {
      const gap = (offsetOf(s) - offsetOf(left[0])) / 5;
      expect(Math.abs(gap - Math.round(gap))).toBeLessThan(1e-6);
    }
  });

  it("covers the rect at every angle", () => {
    // Every corner has to fall inside the span the lines cross, or one end of
    // the region would be left unpainted.
    for (const angle of [15, 75, 135, 165]) {
      const segs = rulingSegments(rect, angle, 5);
      const { nx, ny } = perpendicular(segs[0], 5);
      const offsets = segs.map((s) => s.x1 * nx + s.y1 * ny);
      const corners = [
        [rect.x, rect.y],
        [rect.x + rect.w, rect.y],
        [rect.x, rect.y + rect.h],
        [rect.x + rect.w, rect.y + rect.h],
      ].map(([cx, cy]) => cx * nx + cy * ny);
      expect(Math.min(...offsets) - Math.min(...corners), `angle ${angle}`).toBeLessThanOrEqual(5);
      expect(Math.max(...corners) - Math.max(...offsets), `angle ${angle}`).toBeLessThanOrEqual(5);
    }
  });

  it("refuses a degenerate or unpayable ruling instead of stalling the frame", () => {
    expect(rulingSegments(rect, 135, 0)).toEqual([]);
    expect(rulingSegments(rect, 135, -1)).toEqual([]);
    expect(rulingSegments({ x: 0, y: 0, w: 8192, h: 8192 }, 135, 0.001)).toEqual([]);
  });
});


describe("screen-space minimum marks", () => {
  it("preserves a full pixel at fitted zoom, including the far corner", () => {
    const shape = [4096, 4096];
    const cfg = { sliders: [], projection: true };
    const geom = gridGeometry(shape, cfg, 0, 0.1);
    for (const index of [0, 4095]) {
      const [rect] = regionRects(fromBox(box([index, index + 1], [index, index + 1])), shape, cfg, geom, 0.3);
      expect(rect.w * 0.3).toBeGreaterThanOrEqual(1);
      expect(rect.h * 0.3).toBeGreaterThanOrEqual(1);
      expect(rect.x).toBeGreaterThanOrEqual(0);
      expect(rect.y).toBeGreaterThanOrEqual(0);
      expect(rect.x + rect.w).toBeLessThanOrEqual(geom.canvasW);
      expect(rect.y + rect.h).toBeLessThanOrEqual(geom.canvasH);
    }
  });
});

describe("external seed corners", () => {
  it("keeps tick strokes outside the represented region at any zoom", () => {
    const rect = { x: 20, y: 30, w: 1, h: 50 };
    for (const scale of [0.1, 0.3, 1, 4]) {
      const segments = seedCornerSegments(rect, scale);
      expect(segments).toHaveLength(8);
      for (const s of segments) {
        expect(Math.hypot(s.x2 - s.x1, s.y2 - s.y1) * scale).toBeCloseTo(3);
        const horizontalOutside = s.y1 === s.y2 && (s.y1 < rect.y || s.y1 > rect.y + rect.h);
        const verticalOutside = s.x1 === s.x2 && (s.x1 < rect.x || s.x1 > rect.x + rect.w);
        expect(horizontalOutside || verticalOutside).toBe(true);
      }
    }
  });
});
