/**
 * Shared symbolic-extent mappings for `OpSpec.inferSymShapes`.
 *
 * Every helper returns a full-length shape, writing an axis's own literal
 * extent wherever no symbol carries across. See the note on `inferSymShapes`
 * for why this is a separate question from axis names.
 */

import { Sym } from "../shapes";
import { OpCtx } from "./types";

/** The output has the input's extents unchanged. */
export function sameSymShape(inSyms: Sym[][]): Sym[][] {
  return [inSyms[0].slice()];
}

/**
 * Trailing-aligned, like the broadcast it describes. A symbol carries only
 * where the input axis already has the output's extent: an axis stretched from
 * 1 says nothing about how wide the result is.
 */
export function broadcastSymShape(
  inSyms: Sym[][],
  ctx: OpCtx,
  outSlot = 0
): Sym[] {
  const outShape = ctx.outShapes[outSlot];
  return outShape.map((extent, axis) => {
    for (let slot = 0; slot < inSyms.length; slot++) {
      const aligned = axis - (outShape.length - ctx.inShapes[slot].length);
      if (aligned < 0 || ctx.inShapes[slot][aligned] !== extent) continue;
      const sym = inSyms[slot][aligned];
      if (typeof sym === "string") return sym;
    }
    return extent;
  });
}
