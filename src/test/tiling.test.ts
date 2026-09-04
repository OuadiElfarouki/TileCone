import { describe, expect, it } from "vitest";
import {
  AUTO_TILE_FRACTION,
  autoTile,
  cardPx,
  governingAxis,
  graphScale,
  effectiveTileScaleIndex,
  effectiveTileScaleStops,
  MAX_ELEM_PX,
  MAX_GRAPH_W,
  MAX_GRAPH_H,
  maxTile,
  planeExtents,
  settledTiles,
  MIN_CELL_PX,
  MIN_SIDE_PX,
  pow2Floor,
  pow2Round,
  tileFor,
  TILE_SCALE_MAX,
  TILE_SCALE_MIN,
} from "../ui/tiling";
import {
  gridGeometry,
  MIN_MARK_PX,
  outlineFitsRect,
  patternFitsRect,
  regionRects,
  rulingSegments,
  stripePitchPx,
} from "../ui/grid";
import { box, fromBox } from "../core/region";
import { viewAxes } from "../ui/store";
import type { ViewCfg } from "../ui/store";

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

describe("auto tile sizing", () => {
  it("matches the stated rule on the worked examples", () => {
    expect(autoTile(100, 200)).toBe(4); // 5% of 100 = 5 -> 4
    expect(autoTile(200, 214)).toBe(8); // 5% of 200 = 10 -> 8
  });

  it("uses the smallest non-degenerate axis", () => {
    expect(governingAxis(100, 200)).toBe(100);
    expect(governingAxis(1, 4096)).toBe(4096); // a 1-D tensor uses its only axis
    expect(governingAxis(1, 1)).toBe(1);
  });

  it("is always a power of two within [1, minAxis/2]", () => {
    for (const [r, c] of [[100, 200], [200, 214], [4, 4], [1, 4096], [4096, 4096], [7, 13], [2, 2]]) {
      const t = autoTile(r, c);
      expect(isPow2(t), `tile ${t} for ${r}x${c}`).toBe(true);
      expect(t).toBeGreaterThanOrEqual(1);
      expect(t).toBeLessThanOrEqual(maxTile(r, c));
    }
  });

  it("leaves at least a 2x2 tile grid on the smallest axis", () => {
    for (const [r, c] of [[4, 4], [8, 64], [100, 200]]) {
      const t = maxTile(r, c);
      expect(Math.floor(governingAxis(r, c) / t)).toBeGreaterThanOrEqual(2);
    }
  });

  it("small tensors reach one element per cell", () => {
    expect(autoTile(4, 4)).toBe(1);
    expect(tileFor(8, 8, TILE_SCALE_MIN, pxFor(8, 8))).toBe(1);
  });
});

describe("the global detail scale", () => {
  it("shifts the tile by powers of two and stays monotone", () => {
    let prev = 0;
    for (let k = TILE_SCALE_MIN; k <= TILE_SCALE_MAX; k++) {
      const t = tileFor(256, 256, k, pxFor(256, 256));
      expect(isPow2(t)).toBe(true);
      expect(t).toBeGreaterThanOrEqual(prev);
      prev = t;
    }
  });

  it("reaches half the smallest axis at the coarse end", () => {
    expect(tileFor(256, 256, TILE_SCALE_MAX, pxFor(256, 256))).toBe(maxTile(256, 256));
    expect(tileFor(100, 200, TILE_SCALE_MAX, pxFor(100, 200))).toBe(maxTile(100, 200));
  });

  it("reaches one element per cell when the tensor is small enough to draw", () => {
    expect(tileFor(64, 64, TILE_SCALE_MIN, pxFor(64, 64))).toBe(1);
    expect(tileFor(32, 90, TILE_SCALE_MIN, pxFor(32, 90))).toBe(1);
  });

  it("none is one element per tile even when boundaries are sub-pixel", () => {
    const finest = tileFor(256, 256, TILE_SCALE_MIN, pxFor(256, 256));
    expect(finest).toBe(1);
    const geom = gridGeometry([256, 256], cfg, TILE_SCALE_MIN, pxFor(256, 256));
    expect(geom.tile).toBe(1);
    expect(geom.tileRows).toBe(256);
    expect(geom.tileCols).toBe(256);
    expect(Math.min(geom.cellW, geom.cellH)).toBeLessThan(MIN_CELL_PX);
  });

  it("normal scale values never render cells below the minimum size", () => {
    for (const [r, c] of [[4096, 4096], [1, 100000], [2048, 512]]) {
      for (let k = TILE_SCALE_MIN + 1; k <= TILE_SCALE_MAX; k++) {
        const geom = gridGeometry([r, c], cfg, k, pxFor(r, c));
        expect(Math.min(geom.cellW, geom.cellH), `${r}x${c} @${k}`).toBeGreaterThanOrEqual(MIN_CELL_PX);
      }
    }
  });

  it("the auto default is ~5% of the smallest axis", () => {
    for (const [r, c] of [[100, 200], [200, 214], [512, 512], [1000, 1000]]) {
      const t = autoTile(r, c);
      const ideal = AUTO_TILE_FRACTION * governingAxis(r, c);
      expect(t).toBeGreaterThanOrEqual(ideal / 2);
      expect(t).toBeLessThanOrEqual(ideal * 2);
    }
  });

  it("offers only graph-wide lattice changes as slider stops", () => {
    const planes = [
      { rows: 256, cols: 512 },
      { rows: 512, cols: 256 },
      { rows: 256, cols: 256 },
    ];
    const px = graphScale(planes);
    const stops = effectiveTileScaleStops(planes, px);
    // TILE_SCALE_MIN leads as the explicit "no tiling" request; the rest are
    // the scales that genuinely change the graph-wide lattice.
    expect(stops).toEqual([TILE_SCALE_MIN, 0, 1, 2, 3]);
    const state = (scale: number) =>
      planes.map(({ rows, cols }) => tileFor(rows, cols, scale, px));
    for (let i = 2; i < stops.length; i++) expect(state(stops[i])).not.toEqual(state(stops[i - 1]));
  });

  it("keeps no tiling semantically distinct from fitted scales", () => {
    const planes = [{ rows: 256, cols: 512 }, { rows: 512, cols: 256 }];
    const px = graphScale(planes);
    expect(effectiveTileScaleStops(planes, px)[0]).toBe(TILE_SCALE_MIN);
    expect(settledTiles(planes, TILE_SCALE_MIN, px)).toEqual({ min: 1, max: 1 });
    expect(planes.map(({ rows, cols }) => tileFor(rows, cols, TILE_SCALE_MIN + 1, px)))
      .not.toEqual([1, 1]);
  });

  it("settles at one cell per element where that does fit", () => {
    const planes = [{ rows: 8, cols: 8 }];
    const px = graphScale(planes);
    expect(settledTiles(planes, TILE_SCALE_MIN, px)).toEqual({ min: 1, max: 1 });
  });

  it("names the spread when the graph does not settle on one tile", () => {
    const planes = [{ rows: 8, cols: 8 }, { rows: 512, cols: 512 }];
    const px = graphScale(planes);
    const { min, max } = settledTiles(planes, 0, px);
    expect(max).toBeGreaterThan(min); // a single number would be a lie here
  });

  it("maps legacy plateau values onto their equivalent visible stop", () => {
    const planes = [{ rows: 256, cols: 512 }];
    const px = graphScale(planes);
    const stops = effectiveTileScaleStops(planes, px);
    expect(stops[effectiveTileScaleIndex(planes, px, stops, TILE_SCALE_MIN)]).toBe(TILE_SCALE_MIN);
    expect(stops[effectiveTileScaleIndex(planes, px, stops, TILE_SCALE_MAX)]).toBe(3);
  });

  it("does not confuse auto with none when both happen to be 1x1", () => {
    const planes = [{ rows: 8, cols: 8 }];
    const px = graphScale(planes);
    const stops = effectiveTileScaleStops(planes, px);
    expect(stops).toContain(TILE_SCALE_MIN);
    expect(stops).toContain(0);
    expect(stops[effectiveTileScaleIndex(planes, px, stops, TILE_SCALE_MIN)])
      .toBe(TILE_SCALE_MIN);
    expect(stops[effectiveTileScaleIndex(planes, px, stops, 0)]).toBe(0);
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

  it("cells stay square, so a tile reads as a tile", () => {
    const planes = [{ rows: 256, cols: 512 }, { rows: 512, cols: 256 }];
    const px = graphScale(planes);
    for (const { rows, cols } of planes) {
      const geom = gridGeometry([rows, cols], cfg, 0, px);
      expect(Math.abs(geom.cellW - geom.cellH), `${rows}x${cols}`).toBeLessThan(1);
    }
  });

  it("a degenerate axis does not drag the whole graph to the cap", () => {
    // A rank-1 bias vector has rows === 1. Counting that as the smallest side
    // would demand 14px for it and peg every card at MAX_ELEM_PX — one bias
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
    expect(geom.tile).toBeGreaterThan(1); // cells aggregate several elements
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
    // without the other — a ruled fill is never left without its edge.
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
    // ceiling to read as dense — but never so tight it fills in as solid.
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
