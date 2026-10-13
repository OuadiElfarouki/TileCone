import type { ResolvedGraph } from "../../core/graph";
import type { Rect } from "../graph/graph-geometry";
import { nodeKey, type BaseGraphLayout } from "../graph/graph-scene";
import type { NodeOffsets } from "../graph/node-layout";
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

const overlap = (a: Rect, b: Rect) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
  Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));

/** Stop at the first neighbour, without worsening any pre-existing overlap.
 * An anchored uniform resize sweeps nested rectangles, so bisection cannot
 * jump through a blocker even when pointer events are coalesced. */
export function constrainResize(from: number, to: number, measure: (scale: number) => Rect, blockers: readonly Rect[]): number {
  const before = measure(from);
  const fits = (scale: number) => {
    const rect = measure(scale);
    return blockers.every((b) => overlap(rect, b) <= overlap(before, b) + 1e-6);
  };
  if (fits(to)) return to;
  let accepted = from, refused = to;
  for (let i = 0; i < 36; i++) {
    const middle = (accepted + refused) / 2;
    if (fits(middle)) accepted = middle; else refused = middle;
  }
  // The stop lands between two steps. Every scale between `from` and one that
  // fits also fits, so stepping back toward `from` finds a fitting step within two.
  const dir = Math.sign(to - from);
  for (let step = (dir > 0 ? Math.floor : Math.ceil)(refused * SCALE_STEPS); (step / SCALE_STEPS - from) * dir > 0; step -= dir)
    if (fits(step / SCALE_STEPS)) return step / SCALE_STEPS;
  return from;
}

/** The scale each resized card is drawn at. Layout does not reserve room for
 * a stored scale, so a later layout can leave a card less room than its resize
 * had; it is then drawn smaller, stopping at its neighbours, and keeps its
 * stored scale for when room returns. Cards claim room in key order, which
 * makes the result a function of the saved workspace alone.
 *
 * `nodes` are placed at their fitted size; `measure` gives a card's rect at a
 * scale, from the same top-left corner. */
export function displayedCardScales(
  nodes: readonly (Rect & { key: string })[],
  stored: Readonly<Record<string, number>>,
  measure: (key: string, scale: number) => Rect
): Record<string, number> {
  const rects = new Map<string, Rect>(nodes.map((node) => [node.key, node]));
  const shown: Record<string, number> = {};
  for (const key of Object.keys(stored).sort()) {
    if (!rects.has(key)) continue;
    const blockers = [...rects].filter(([other]) => other !== key).map(([, rect]) => rect);
    shown[key] = constrainResize(1, stored[key], (scale) => measure(key, scale), blockers);
    rects.set(key, measure(key, shown[key]));
  }
  return shown;
}

/** The drawn scale and size of every resized card in a placed graph, by node key. */
export function displayedCards(
  base: BaseGraphLayout,
  nodeOffsets: Readonly<NodeOffsets>,
  viewCfgs: Readonly<Record<string, ViewCfg>>,
  resolved: ResolvedGraph,
  graphPx: number
): { scales: Record<string, number>; sizes: Record<string, { w: number; h: number }> } {
  const ids = new Map<string, string>();
  const stored: Record<string, number> = {};
  for (const [id, cfg] of Object.entries(viewCfgs)) {
    if (cfg.scale === undefined || !resolved.tensors[id]) continue;
    ids.set(nodeKey("tensor", id), id);
    stored[nodeKey("tensor", id)] = cfg.scale;
  }
  if (!ids.size) return { scales: {}, sizes: {} };
  const placed = base.nodes.map((node) => {
    const key = nodeKey(node.kind, node.id), offset = nodeOffsets[key];
    return { key, x: node.x + (offset?.dx ?? 0), y: node.y + (offset?.dy ?? 0), w: node.w, h: node.h };
  });
  const corners = new Map(placed.map((node) => [node.key, node]));
  const sizeAt = (key: string, scale: number) => {
    const id = ids.get(key)!;
    return tensorCardSize(resolved.tensors[id], graphPx, { ...viewCfgs[id], scale });
  };
  const scales = displayedCardScales(placed, stored, (key, scale) =>
    ({ ...sizeAt(key, scale), x: corners.get(key)!.x, y: corners.get(key)!.y }));
  const sizes = Object.fromEntries(Object.entries(scales).map(([key, scale]) => [key, sizeAt(key, scale)]));
  return { scales, sizes };
}
