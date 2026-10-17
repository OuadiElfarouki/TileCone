/**
 * Shared symbolic-extent mappings for `OpSpec.inferSymShapes`.
 *
 * Every helper returns a full-length shape, writing an axis's own literal
 * extent wherever no symbol carries across. See the note on `inferSymShapes`
 * for why this is a separate question from axis names.
 */

import { Sym } from "../shapes";
import { OpCtx, SymEquality } from "./types";

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

/**
 * Broadcasting requires each pair of aligned axes to be equal unless one of
 * them is 1. "Is 1" is asked of the expression, not the extent: an axis
 * written `N` broadcasts against `M` only because `N` is bound to 1, which is
 * as much a coincidence as `N` being bound to `M`'s value.
 *
 * Every pair is compared, not each against the first: an unknown first axis
 * would otherwise leave two provably different ones behind it unchecked.
 */
export function broadcastSymEqualities(inSyms: Sym[][], ctx: OpCtx, outSlot = 0): SymEquality[] {
  const rank = ctx.outShapes[outSlot].length;
  const equalities: SymEquality[] = [];
  for (let axis = 0; axis < rank; axis++) {
    const aligned: { slot: number; at: number; sym: Sym }[] = [];
    for (let slot = 0; slot < inSyms.length; slot++) {
      const at = axis - (rank - inSyms[slot].length);
      if (at < 0) continue;
      aligned.push({ slot, at, sym: inSyms[slot][at] });
    }
    for (let i = 0; i < aligned.length; i++)
      for (let j = i + 1; j < aligned.length; j++)
        equalities.push({
          left: aligned[i].sym,
          right: aligned[j].sym,
          what: `broadcast axes input ${aligned[i].slot} axis ${aligned[i].at} and input ${aligned[j].slot} axis ${aligned[j].at}`,
          input: aligned[j].slot,
          orOne: "either",
        });
  }
  return equalities;
}
