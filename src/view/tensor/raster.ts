import type { Rect } from "../graph/graph-geometry";

/** Raster policy, independent of tensor geometry and selection precision.
 * RGBA storage is at most 16 MiB per card and 64 MiB across active cards.
 * Browser/GPU overhead is additional; the side limit is a compatibility guard. */
export const MAX_CARD_RASTER_PIXELS = 4 * 1024 * 1024;
export const MAX_GRAPH_RASTER_PIXELS = 16 * 1024 * 1024;
export const MAX_RASTER_SIDE = 8192;
const RASTER_MARGIN_PX = 64;

export function rasterSize(w: number, h: number, requestedScale: number, maxPixels = MAX_CARD_RASTER_PIXELS) {
  const budget = Math.max(1, Math.min(MAX_CARD_RASTER_PIXELS, Math.floor(maxPixels)));
  const scale = Math.min(
    requestedScale,
    MAX_RASTER_SIDE / Math.max(1, w),
    MAX_RASTER_SIDE / Math.max(1, h),
    Math.sqrt(budget / Math.max(1, w * h)),
  );
  // A very thin card still needs one pixel on its short axis. Bound the long
  // axis too, because rounding the short axis up could otherwise exceed area.
  const width = Math.max(1, Math.min(budget, Math.floor(w * scale)));
  const height = Math.max(1, Math.min(Math.floor(budget / width), Math.floor(h * scale)));
  return { width, height };
}

/** Keep DOM nodes available to keyboard navigation while releasing offscreen
 * rasters. Share one allocation budget among cards near the viewport. */
export function graphRasterAllocation(
  nodes: readonly (Rect & { id: string; kind: "tensor" | "op" })[],
  transform: { x: number; y: number; k: number },
  viewport: { width: number; height: number },
) {
  const visible = new Set<string>();
  if (viewport.width > 0 && viewport.height > 0) {
    for (const node of nodes) {
      if (node.kind !== "tensor") continue;
      const x = transform.x + node.x * transform.k;
      const y = transform.y + node.y * transform.k;
      if (x + node.w * transform.k >= -RASTER_MARGIN_PX && x <= viewport.width + RASTER_MARGIN_PX &&
          y + node.h * transform.k >= -RASTER_MARGIN_PX && y <= viewport.height + RASTER_MARGIN_PX)
        visible.add(node.id);
    }
  }
  return {
    visible,
    pixelsPerCard: Math.min(MAX_CARD_RASTER_PIXELS, Math.floor(MAX_GRAPH_RASTER_PIXELS / Math.max(1, visible.size))),
  };
}
