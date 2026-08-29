import { z } from "zod";
import { Box, Region, fromBox, iv } from "../region";
import { normAxes } from "./reduce";
import { OpCtx, OpSpec } from "./types";

type NAttrs = { kind: "layernorm" | "rmsnorm"; axes: number[]; hasWeight: boolean; hasBias: boolean };

function cfg(ctx: OpCtx) {
  const a = ctx.attrs as NAttrs;
  return { ...a, axes: normAxes(a.axes, ctx.inShapes[0].length) };
}

/**
 * Inputs: data, [weight], [bias] — weight/bias shaped as the normalized axes' extents.
 * Backward on data: full extent along `axes`, identity elsewhere.
 * Backward on weight/bias: the OUTPUT BOX's intervals restricted to `axes`.
 */
export const normalizeOp: OpSpec = {
  name: "normalize",
  attrSchema: z.object({
    kind: z.enum(["layernorm", "rmsnorm"]),
    axes: z.array(z.number().int()),
    hasWeight: z.boolean().default(false),
    hasBias: z.boolean().default(false),
  }),
  arity: { inputs: "variadic", outputs: 1 },
  inferShapes: (inShapes, a) => {
    const attrs = a as NAttrs;
    const sh = inShapes[0];
    const axes = normAxes(attrs.axes, sh.length);
    const paramShape = axes.map((ax) => sh[ax]);
    const expected = 1 + (attrs.hasWeight ? 1 : 0) + (attrs.hasBias ? 1 : 0);
    if (inShapes.length !== expected)
      throw new Error(`normalize: expected ${expected} inputs, got ${inShapes.length}`);
    let slot = 1;
    for (const flag of [attrs.hasWeight, attrs.hasBias]) {
      if (!flag) continue;
      const got = inShapes[slot++];
      if (got.length !== paramShape.length || got.some((e, i) => e !== paramShape[i]))
        throw new Error(`normalize: param shape [${got}] != normalized axes [${paramShape}]`);
    }
    return [sh.slice()];
  },
  backward: (_s, outBox, ctx) => {
    const { axes, hasWeight, hasBias } = cfg(ctx);
    const dataBox: Box = outBox.map((I, ax) =>
      axes.includes(ax) ? iv(0, ctx.inShapes[0][ax]) : { ...I }
    );
    const paramBox: Box = axes.map((ax) => ({ ...outBox[ax] }));
    const out: Region[] = [fromBox(dataBox)];
    if (hasWeight) out.push(fromBox(paramBox.map((I) => ({ ...I }))));
    if (hasBias) out.push(fromBox(paramBox.map((I) => ({ ...I }))));
    return out;
  },
  forward: (inSlot, inBox, ctx) => {
    const { axes } = cfg(ctx);
    const outShape = ctx.outShapes[0];
    if (inSlot === 0) {
      const b: Box = inBox.map((I, ax) => (axes.includes(ax) ? iv(0, outShape[ax]) : { ...I }));
      return [fromBox(b)];
    }
    // weight or bias: axes' intervals from the param box, full extent elsewhere
    const b: Box = outShape.map((e, ax) => {
      const k = axes.indexOf(ax);
      return k >= 0 ? { ...inBox[k] } : iv(0, e);
    });
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const { axes, hasWeight, hasBias } = cfg(ctx);
    const inShape = ctx.inShapes[0];
    const dataDeps: number[][] = [];
    const rec = (i: number, idx: number[]) => {
      if (i === axes.length) {
        dataDeps.push(idx.slice());
        return;
      }
      const ax = axes[i];
      for (let v = 0; v < inShape[ax]; v++) {
        idx[ax] = v;
        rec(i + 1, idx);
      }
      idx[ax] = outIndex[ax];
    };
    rec(0, outIndex.slice());
    const paramIdx = [axes.map((ax) => outIndex[ax])];
    const deps: number[][][] = [dataDeps];
    if (hasWeight) deps.push(paramIdx.map((p) => p.slice()));
    if (hasBias) deps.push(paramIdx.map((p) => p.slice()));
    return deps;
  },
  flopsFor: (_s, outBox, ctx) => {
    let vol = 1;
    for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
    return ((ctx.attrs as NAttrs).kind === "layernorm" ? 8 : 5) * vol;
  },
};
