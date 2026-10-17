import type { ResolvedGraph } from "../../core/graph";
import { nodeKey } from "../graph/graph-scene";
import { tensorCardSize } from "./card-size";
import { MAX_CARD_SCALE, MIN_CARD_SCALE, type ViewCfg } from "./tensor-view";

export type ResizeCorner = "top" | "bottom";

/** Scales are kept to three decimals, which keeps files and links short. */
const SCALE_STEPS = 1000;

/** Project the pointer onto the canvas diagonal, preserving its proportions. */
export function resizeScale(scale: number, size: { w: number; h: number }, delta: { x: number; y: number }, corner: ResizeCorner): number {
  const along = (size.w * delta.x + size.h * delta.y * (corner === "top" ? -1 : 1)) / (size.w ** 2 + size.h ** 2);
  return Math.max(MIN_CARD_SCALE, Math.min(MAX_CARD_SCALE, Math.round(scale * (1 + along) * SCALE_STEPS) / SCALE_STEPS));
}

/** The scale and size of every resized card, by node key. A card is drawn at
 * the scale it was resized to wherever it sits; a resize moves its neighbours
 * aside rather than the card shrinking to fit between them. */
export function displayedCards(
  viewCfgs: Readonly<Record<string, ViewCfg>>,
  resolved: ResolvedGraph,
  graphPx: number
): { scales: Record<string, number>; sizes: Record<string, { w: number; h: number }> } {
  const scales: Record<string, number> = {};
  const sizes: Record<string, { w: number; h: number }> = {};
  for (const [id, cfg] of Object.entries(viewCfgs)) {
    if (cfg.scale === undefined || !resolved.tensors[id]) continue;
    const key = nodeKey("tensor", id);
    scales[key] = cfg.scale;
    sizes[key] = tensorCardSize(resolved.tensors[id], graphPx, cfg);
  }
  return { scales, sizes };
}
