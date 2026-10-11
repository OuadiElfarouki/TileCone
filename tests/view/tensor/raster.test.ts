import { describe, expect, it } from "vitest";
import {
  graphRasterAllocation, rasterSize,
  MAX_CARD_RASTER_PIXELS, MAX_GRAPH_RASTER_PIXELS, MAX_RASTER_SIDE,
} from "../../../src/view/tensor/raster";

describe("tensor raster allocation", () => {
  it("bounds both area and side length across large, thin, and high-DPR cards", () => {
    for (const w of [0.01, 1, 32, 64000, 1e12]) {
      for (const h of [0.01, 1, 32, 64000, 1e12]) {
        for (const scale of [0.125, 1, 8]) {
          for (const budget of [1, 4096, MAX_CARD_RASTER_PIXELS]) {
            const size = rasterSize(w, h, scale, budget);
            expect(size.width).toBeGreaterThanOrEqual(1);
            expect(size.height).toBeGreaterThanOrEqual(1);
            expect(size.width).toBeLessThanOrEqual(MAX_RASTER_SIDE);
            expect(size.height).toBeLessThanOrEqual(MAX_RASTER_SIDE);
            expect(size.width * size.height).toBeLessThanOrEqual(budget);
          }
        }
      }
    }
  });

  it("preserves requested resolution for cards within budget", () => {
    expect(rasterSize(320, 180, 2)).toEqual({ width: 640, height: 360 });
    expect(rasterSize(320, 180, 0.5)).toEqual({ width: 160, height: 90 });
  });

  it("uses screen coordinates for visibility and includes a small prepaint margin", () => {
    const nodes = [
      { id: "in", kind: "tensor" as const, x: 400, y: 100, w: 100, h: 100 },
      { id: "near", kind: "tensor" as const, x: 610, y: 100, w: 100, h: 100 },
      { id: "far", kind: "tensor" as const, x: 800, y: 100, w: 100, h: 100 },
      { id: "op", kind: "op" as const, x: 400, y: 100, w: 100, h: 100 },
    ];
    const allocation = graphRasterAllocation(nodes, { x: -700, y: -100, k: 2 }, { width: 500, height: 400 });
    expect([...allocation.visible]).toEqual(["in", "near"]);
    expect(graphRasterAllocation(nodes, { x: 0, y: 0, k: 1 }, { width: 0, height: 0 }).visible.size).toBe(0);
  });

  it("shares the graph budget even when an overview shows hundreds of huge cards", () => {
    const nodes = Array.from({ length: 257 }, (_, i) => ({
      id: String(i), kind: "tensor" as const, x: i * 64001, y: 0, w: 64000, h: 64000,
    }));
    const allocation = graphRasterAllocation(nodes, { x: 0, y: 0, k: 0.00001 }, { width: 1000, height: 600 });
    expect(allocation.visible.size).toBe(nodes.length);
    const total = nodes.reduce((sum, node) => {
      const size = rasterSize(node.w, node.h, 4, allocation.pixelsPerCard);
      return sum + size.width * size.height;
    }, 0);
    expect(total).toBeLessThanOrEqual(MAX_GRAPH_RASTER_PIXELS);
  });
});
