import { z } from "zod";
import { Box, Region, canonicalize, count, coversAxisFully, fromBox, iv } from "../region";
import { normAxis } from "./reduce";
import { DependencyNoteDraft, OpCtx, OpSpec, uniformDTypeOutputs, NoteCtx } from "./types";

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

/**
 * A softmax normalises over its whole axis, so no tile of the output can be
 * produced from a slice of that axis. That is the single fact a reader needs
 * before trying to split the tensor along it.
 */
function softmaxDependencyNote(ctx: NoteCtx): DependencyNoteDraft | null {
  const region = ctx.inRegions[0];
  if (!region) return null;
  const axis = axisOf(ctx);
  const extent = ctx.inShapes[0][axis];
  const pullsWholeAxis = coversAxisFully(region, axis, extent);
  if (!pullsWholeAxis) return null;
  return {
    key: `softmax:${axis}:${extent}`,
    subject: ctx.outNames[0],
    severity: 2,
    flags: [
      { tensorId: ctx.inIds[0], text: `full axis ${axis} — softmax normalises across it` },
    ],
    text:
      `softmax normalises ${ctx.inNames[0]} along axis ${axis} (${extent} wide), so a tile ` +
      `of ${ctx.outNames[0]} needs that axis of ${ctx.inNames[0]} complete. Splitting ` +
      `${ctx.inNames[0]} along axis ${axis} breaks fusion unless the reduction is done online.`,
  };
}

/** Backward and forward: full extent along `axis`, identity elsewhere. */
export const softmaxOp: OpSpec = {
  name: "softmax",
  dependencyNote: softmaxDependencyNote,
  attrSchema: z.object({ axis: z.number().int() }),
  arity: { inputs: 1, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("softmax"),
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
