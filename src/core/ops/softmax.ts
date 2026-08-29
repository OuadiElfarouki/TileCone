import { z } from "zod";
import { Box, fromBox, iv } from "../region";
import { normAxis } from "./reduce";
import { OpCtx, OpSpec } from "./types";

const axisOf = (ctx: OpCtx) => normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);

/** Backward and forward: full extent along `axis`, identity elsewhere. */
export const softmaxOp: OpSpec = {
  name: "softmax",
  attrSchema: z.object({ axis: z.number().int() }),
  arity: { inputs: 1, outputs: 1 },
  inferShapes: (inShapes) => [inShapes[0].slice()],
  backward: (_s, outBox, ctx) => {
    const ax = axisOf(ctx);
    const b: Box = outBox.map((I, i) => (i === ax ? iv(0, ctx.inShapes[0][ax]) : { ...I }));
    return [fromBox(b)];
  },
  forward: (_s, inBox, ctx) => {
    const ax = axisOf(ctx);
    const b: Box = inBox.map((I, i) => (i === ax ? iv(0, ctx.outShapes[0][ax]) : { ...I }));
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const ax = axisOf(ctx);
    const deps: number[][] = [];
    for (let v = 0; v < ctx.inShapes[0][ax]; v++) {
      const idx = outIndex.slice();
      idx[ax] = v;
      deps.push(idx);
    }
    return [deps];
  },
  flopsFor: (_s, outBox, ctx) => {
    let vol = 1;
    for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
    void ctx;
    return 5 * vol; // max + sub + exp + sum + div, amortized
  },
};
