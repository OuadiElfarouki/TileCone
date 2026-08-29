import { describe, expect, it } from "vitest";
import {
  AUTO_TILE_FRACTION,
  autoTile,
  cardPx,
  governingAxis,
  isTileClamped,
  MAX_CARD_W,
  MAX_CARD_H,
  maxTile,
  MIN_CELL_PX,
  pow2Floor,
  pow2Round,
  tileFor,
  TILE_SCALE_MAX,
  TILE_SCALE_MIN,
} from "../ui/tiling";
import { gridGeometry, tileCoverage } from "../ui/grid";
import { box, fromBox, union, count } from "../core/region";
import type { ViewCfg } from "../ui/store";

const cfg: ViewCfg = { sliders: [0, 0, 0, 0], projection: true };
const isPow2 = (n: number) => n >= 1 && (n & (n - 1)) === 0;

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
    expect(tileFor(8, 8, TILE_SCALE_MIN)).toBe(1);
  });
});

describe("the global detail scale", () => {
  it("shifts the tile by powers of two and stays monotone", () => {
    let prev = 0;
    for (let k = TILE_SCALE_MIN; k <= TILE_SCALE_MAX; k++) {
      const t = tileFor(256, 256, k);
      expect(isPow2(t)).toBe(true);
      expect(t).toBeGreaterThanOrEqual(prev);
      prev = t;
    }
  });

  it("reaches half the smallest axis at the coarse end", () => {
    expect(tileFor(256, 256, TILE_SCALE_MAX)).toBe(maxTile(256, 256));
    expect(tileFor(100, 200, TILE_SCALE_MAX)).toBe(maxTile(100, 200));
  });

  it("reaches one element per cell when the tensor is small enough to draw", () => {
    expect(tileFor(64, 64, TILE_SCALE_MIN)).toBe(1);
    expect(tileFor(32, 90, TILE_SCALE_MIN)).toBe(1);
  });

  it("larger tensors bottom out at a minimum elementary tile instead", () => {
    // 256 columns at one element per cell would need a card ~768px wide, so the
    // fit rule stops at the finest tile that still draws at MIN_CELL_PX.
    const finest = tileFor(256, 256, TILE_SCALE_MIN);
    expect(finest).toBeGreaterThan(1);
    expect(isTileClamped(256, 256, TILE_SCALE_MIN)).toBe(true);
    const geom = gridGeometry([256, 256], cfg, TILE_SCALE_MIN);
    expect(Math.min(geom.cellW, geom.cellH)).toBeGreaterThanOrEqual(MIN_CELL_PX);
  });

  it("never renders cells below the minimum size, however fine the request", () => {
    for (const [r, c] of [[4096, 4096], [1, 100000], [2048, 512]]) {
      for (let k = TILE_SCALE_MIN; k <= TILE_SCALE_MAX; k++) {
        const geom = gridGeometry([r, c], cfg, k);
        expect(Math.min(geom.cellW, geom.cellH), `${r}x${c} @${k}`).toBeGreaterThanOrEqual(MIN_CELL_PX);
        expect(geom.canvasW).toBeLessThanOrEqual(MAX_CARD_W);
        expect(geom.canvasH).toBeLessThanOrEqual(MAX_CARD_H);
      }
    }
  });

  it("reports when the fit rule overrode the request", () => {
    expect(isTileClamped(4096, 4096, TILE_SCALE_MIN)).toBe(true); // too fine to draw
    expect(isTileClamped(256, 256, 0)).toBe(false);
  });

  it("the auto default is ~5% of the smallest axis", () => {
    for (const [r, c] of [[100, 200], [200, 214], [512, 512], [1000, 1000]]) {
      const t = autoTile(r, c);
      const ideal = AUTO_TILE_FRACTION * governingAxis(r, c);
      expect(t).toBeGreaterThanOrEqual(ideal / 2);
      expect(t).toBeLessThanOrEqual(ideal * 2);
    }
  });
});

describe("card sizing is independent of the detail setting", () => {
  const SHAPES = [
    [256, 256], [256, 512], [100, 200], [4, 4], [64, 64], [4096, 4096], [1, 512], [200, 214],
  ];

  it("the on-screen grid never changes size when detail changes", () => {
    for (const [r, c] of SHAPES) {
      const base = gridGeometry([r, c], cfg, 0);
      for (let k = TILE_SCALE_MIN; k <= TILE_SCALE_MAX; k++) {
        const g = gridGeometry([r, c], cfg, k);
        expect(g.canvasW, `${r}x${c} @${k} width`).toBe(base.canvasW);
        expect(g.canvasH, `${r}x${c} @${k} height`).toBe(base.canvasH);
      }
    }
  });

  it("only the lattice density changes", () => {
    const fine = gridGeometry([256, 256], cfg, -2);
    const coarse = gridGeometry([256, 256], cfg, 2);
    expect(fine.canvasW).toBe(coarse.canvasW);
    expect(fine.tileCols).toBeGreaterThan(coarse.tileCols);
    expect(fine.cellW).toBeLessThan(coarse.cellW);
  });

  it("card size follows the tensor's shape and aspect ratio", () => {
    const square = cardPx(256, 256);
    expect(square.w).toBe(square.h);
    const wide = cardPx(256, 512);
    expect(wide.w / wide.h).toBeCloseTo(2, 1);
    // a small tensor gets a small card, not a full-size one
    expect(cardPx(4, 4).w).toBeLessThan(square.w);
  });

  it("never exceeds the card bounds", () => {
    for (const [r, c] of SHAPES) {
      const { w, h } = cardPx(r, c);
      expect(w).toBeLessThanOrEqual(MAX_CARD_W);
      expect(h).toBeLessThanOrEqual(MAX_CARD_H);
      expect(w).toBeGreaterThan(0);
      expect(h).toBeGreaterThan(0);
    }
  });
});

describe("tile coverage shading", () => {
  it("a fully covered tile reads 1 and an empty one reads 0", () => {
    const shape = [16, 16];
    const geom = gridGeometry(shape, cfg, 2); // coarse: tile 4 on a 16x16
    expect(geom.tile).toBeGreaterThan(1);
    const t = geom.tile;
    const cov = tileCoverage(fromBox(box([0, t], [0, t])), shape, cfg, geom);
    expect(cov[0]).toBeCloseTo(1, 6);
    expect(cov[1]).toBeCloseTo(0, 6);
  });

  it("a partially covered tile is shaded by its exact fraction", () => {
    const shape = [16, 16];
    const geom = gridGeometry(shape, cfg, 2);
    const t = geom.tile;
    // cover a quarter of the first tile: half the rows, half the cols
    const cov = tileCoverage(fromBox(box([0, t / 2], [0, t / 2])), shape, cfg, geom);
    expect(cov[0]).toBeCloseTo(0.25, 6);
  });

  it("total coverage equals the region's element count over tile volume", () => {
    const shape = [32, 32];
    const geom = gridGeometry(shape, cfg, 1);
    const region = union(fromBox(box([3, 19], [5, 27])), fromBox(box([22, 30], [1, 9])));
    const cov = tileCoverage(region, shape, cfg, geom);
    let total = 0;
    for (const v of cov) total += v;
    expect(total * geom.tile * geom.tile).toBeCloseTo(count(region), 4);
  });

  it("hidden axes contribute their fraction in projection mode", () => {
    const shape = [4, 8, 8]; // axis 0 hidden
    const geom = gridGeometry(shape, cfg, 0);
    const full = tileCoverage(fromBox(box([0, 4], [0, 8], [0, 8])), shape, cfg, geom);
    const oneSlice = tileCoverage(fromBox(box([0, 1], [0, 8], [0, 8])), shape, cfg, geom);
    expect(full[0]).toBeCloseTo(1, 6);
    expect(oneSlice[0]).toBeCloseTo(0.25, 6); // 1 of 4 hidden indices
  });

  it("slice mode shows only the current slider index", () => {
    const shape = [4, 8, 8];
    const sliceCfg: ViewCfg = { sliders: [2, 0, 0], projection: false };
    const geom = gridGeometry(shape, sliceCfg, 0);
    const onSlice = tileCoverage(fromBox(box([2, 3], [0, 8], [0, 8])), shape, sliceCfg, geom);
    const offSlice = tileCoverage(fromBox(box([0, 1], [0, 8], [0, 8])), shape, sliceCfg, geom);
    expect(onSlice[0]).toBeCloseTo(1, 6);
    expect(offSlice[0]).toBeCloseTo(0, 6);
  });

  it("a fully selected edge tile reads 1, not a fraction", () => {
    // 100 columns with tile 8 leaves a 4-wide partial tile at the right edge;
    // selecting all of it must render solid, not half-lit.
    const shape = [100, 100];
    const geom = gridGeometry(shape, cfg, 1);
    const t = geom.tile;
    const lastCol = geom.tileCols - 1;
    expect(shape[1] % t).not.toBe(0); // precondition: the edge tile is partial
    const cov = tileCoverage(fromBox(box([0, t], [lastCol * t, shape[1]])), shape, cfg, geom);
    expect(cov[lastCol]).toBeCloseTo(1, 6);
  });

  it("coverage never exceeds 1 and is never negative", () => {
    const shape = [24, 24];
    const geom = gridGeometry(shape, cfg, 1);
    const region = union(fromBox(box([0, 24], [0, 24])), fromBox(box([4, 12], [4, 12])));
    for (const v of tileCoverage(region, shape, cfg, geom)) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
