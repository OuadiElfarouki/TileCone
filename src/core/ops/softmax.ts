import { z } from "zod";
import { Box, Region, canonicalize, count, fromBox, iv } from "../region";
import { normAxis } from "./reduce";
import { OpCtx, OpSpec } from "./types";

const axisOf = (ctx: OpCtx) => normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);

function softmaxFlops(outRegion: Region, ctx: OpCtx): number {
  const axis = axisOf(ctx);
  const fullAxis = canonicalize({
    boxes: outRegion.boxes.map((box) =>
      box.map((interval, i) => (i === axis ? iv(0, ctx.inShapes[0][axis]) : { ...interval }))
    ),
    exact: outRegion.exact,
    reasons: outRegion.reasons,
  });
  // Matches the primitive lowering: max + subtraction + exp (4 FLOPs) + sum
  // over the full softmax rows, then division only for requested outputs.
  return 7 * count(fullAxis) + count(outRegion);
}

/** Backward and forward: full extent along `axis`, identity elsewhere. */
export const softmaxOp: OpSpec = {
  name: "softmax",
  attrSchema: z.object({ axis: z.number().int() }),
  arity: { inputs: 1, outputs: 1 },
  inferShapes: (inShapes, attrs) => {
    normAxis(attrs.axis as number, inShapes[0].length);
    return [inShapes[0].slice()];
  },
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
  flopsFor: (_s, outBox, ctx) => softmaxFlops(fromBox(outBox), ctx),
  flopsForRegion: (_s, outRegion, ctx) => softmaxFlops(outRegion, ctx),
};
