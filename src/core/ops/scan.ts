import { z } from "zod";
import { Box, fromBox, iv } from "../region";
import { normAxis } from "./reduce";
import { OpCtx, OpSpec } from "./types";

type ScanAttrs = { axis: number; reverse: boolean };

function cfg(ctx: OpCtx) {
  const a = ctx.attrs as ScanAttrs;
  return { axis: normAxis(a.axis, ctx.inShapes[0].length), reverse: a.reverse };
}

/** Triangular dependency cone: output o depends on inputs [0, o] (or [o, n) reversed). */
export const cumsumOp: OpSpec = {
  name: "cumsum",
  attrSchema: z.object({ axis: z.number().int(), reverse: z.boolean().default(false) }),
  arity: { inputs: 1, outputs: 1 },
  inferShapes: (inShapes) => [inShapes[0].slice()],
  backward: (_s, outBox, ctx) => {
    const { axis, reverse } = cfg(ctx);
    const n = ctx.inShapes[0][axis];
    const b: Box = outBox.map((I, ax) =>
      ax === axis ? (reverse ? iv(I.lo, n) : iv(0, I.hi)) : { ...I }
    );
    return [fromBox(b)];
  },
  forward: (_s, inBox, ctx) => {
    const { axis, reverse } = cfg(ctx);
    const n = ctx.outShapes[0][axis];
    const b: Box = inBox.map((I, ax) =>
      ax === axis ? (reverse ? iv(0, I.hi) : iv(I.lo, n)) : { ...I }
    );
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const { axis, reverse } = cfg(ctx);
    const n = ctx.inShapes[0][axis];
    const deps: number[][] = [];
    const [lo, hi] = reverse ? [outIndex[axis], n] : [0, outIndex[axis] + 1];
    for (let v = lo; v < hi; v++) {
      const idx = outIndex.slice();
      idx[axis] = v;
      deps.push(idx);
    }
    return [deps];
  },
  flopsFor: (_s, outBox, ctx) => {
    let vol = 1;
    for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
    void ctx;
    return vol;
  },
};
