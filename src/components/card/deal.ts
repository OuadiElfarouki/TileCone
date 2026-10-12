/**
 * The deal: what a slice card does when its slice changes. Stepping forward
 * deals the front card off the deck and brings the next one up from behind;
 * stepping back deals the earlier slice onto the top while the one shown slips
 * back under. The animation only moves pixels already drawn - `out` holds a
 * copy of the canvas from before the change - so it never shows a picture the
 * data does not.
 */

import { DECK_STEP_PX } from "../../view/tensor/deck";

/** Long enough to read the direction, short enough that scrubbing stays live. */
const DEAL_MS = 200;
const DEAL_EASE = "cubic-bezier(0.2, 0.7, 0.3, 1)";

const activeDeals = new WeakMap<HTMLCanvasElement, () => void>();

/** Release the temporary raster, including a deal interrupted by culling. */
export function cancelDeal(front: HTMLCanvasElement, out: HTMLCanvasElement | null): void {
  if (out) {
    activeDeals.get(out)?.();
    if (out.width !== 0) out.width = 0;
    if (out.height !== 0) out.height = 0;
  }
  front.getAnimations?.().forEach((animation) => animation.cancel());
}

/** Copy what `canvas` shows into `out`, at the same resolution. */
export function holdPicture(canvas: HTMLCanvasElement, out: HTMLCanvasElement): boolean {
  cancelDeal(canvas, out);
  if (!canvas.width || !canvas.height) return false;
  if (out.width !== canvas.width) out.width = canvas.width;
  if (out.height !== canvas.height) out.height = canvas.height;
  const ctx = out.getContext("2d");
  if (!ctx) {
    cancelDeal(canvas, out);
    return false;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, out.width, out.height);
  ctx.drawImage(canvas, 0, 0);
  return true;
}

export const prefersStill = (): boolean =>
  typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/**
 * Animate a deal: `front` now shows the new slice and `out` the old one. `way`
 * is the sign of the step in the deck's row-major order. A new deal cancels a
 * running one, so a scrub reads as cards flicked through rather than a queue.
 */
export function deal(front: HTMLCanvasElement, out: HTMLCanvasElement, way: 1 | -1): () => void {
  if (typeof front.animate !== "function" || typeof out.animate !== "function") {
    cancelDeal(front, out);
    return () => {};
  }
  for (const el of [front, out]) el.getAnimations?.().forEach((a) => a.cancel());
  let live = true;
  const animations: Animation[] = [];
  const dispose = () => {
    // A cancelled deal's promise may settle after the next snapshot is copied.
    if (!live) return;
    live = false;
    activeDeals.delete(out);
    animations.forEach((animation) => animation.cancel());
    out.width = 0;
    out.height = 0;
  };
  const behind = `translate(${DECK_STEP_PX}px, ${-DECK_STEP_PX}px)`;
  const away = `translate(${-4 * DECK_STEP_PX}px, ${5 * DECK_STEP_PX}px) rotate(-3deg)`;
  const timing = { duration: DEAL_MS, easing: DEAL_EASE };
  // Forward, the old card leaves over the new one; back, it goes under it.
  out.style.zIndex = way > 0 ? "2" : "0";
  if (way > 0) {
    animations.push(out.animate([{ transform: "none", opacity: 1 }, { transform: away, opacity: 0 }], timing));
    animations.push(front.animate([{ transform: behind }, { transform: "none" }], timing));
  } else {
    animations.push(out.animate([{ transform: "none", opacity: 1 }, { transform: behind, opacity: 0 }], timing));
    animations.push(front.animate([{ transform: away, opacity: 0 }, { transform: "none", opacity: 1 }], timing));
  }
  activeDeals.set(out, dispose);
  void animations[0].finished.then(dispose, dispose);
  return dispose;
}
