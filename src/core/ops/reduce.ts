import { z } from "zod";
import { Box, fromBox, iv } from "../region";
import { DependencyNoteDraft, OpCtx, OpSpec, uniformDTypeOutputs, NoteCtx } from "./types";

/** Normalize one Python-style axis and reject anything outside the tensor rank. */
export const normAxis = (a: number, rank: number): number => {
  const axis = a < 0 ? a + rank : a;
  if (axis < 0 || axis >= rank)
    throw new Error(`axis ${a} is out of range for rank ${rank}`);
  return axis;
};
export const normAxes = (axes: number[], rank: number) =>
  [...new Set(axes.map((a) => normAxis(a, rank)))].sort((x, y) => x - y);

type RAttrs = { fn: string; axes: number[]; keepdim: boolean };

function attrs(ctx: OpCtx): { axes: number[]; keepdim: boolean; fn: string } {
  const a = ctx.attrs as RAttrs;
  return { axes: normAxes(a.axes, ctx.inShapes[0].length), keepdim: a.keepdim, fn: a.fn };
}

/** A reduction collapses its axes, so an output tile spans them entirely. */
function reduceDependencyNote(ctx: NoteCtx): DependencyNoteDraft | null {
  const region = ctx.inRegions[0];
  if (!region) return null;
  const shape = ctx.inShapes[0];
  const attrs = ctx.attrs as RAttrs;
  const axes = normAxes(attrs.axes, shape.length);
  const full = axes.filter((axis) =>
    region.boxes.some((box) => box[axis].hi - box[axis].lo === shape[axis])
  );
  if (!full.length) return null;
  const list = full.map((axis) => `${axis} (${shape[axis]} wide)`).join(", ");
  const one = full.length === 1;
  return {
    key: `reduce:${attrs.fn}:${full.join(",")}`,
    subject: ctx.outNames[0],
    text:
      `${attrs.fn} reduces ${ctx.inNames[0]} over ${one ? "axis" : "axes"} ${list}, so one ` +
      `element of ${ctx.outNames[0]} depends on every element along ${one ? "it" : "them"}. ` +
      `Tiling ${ctx.inNames[0]} across ${one ? "that axis" : "those axes"} requires a ` +
      `partial-result accumulator.`,
  };
}

export const reduceOp: OpSpec = {
  name: "reduce",
  dependencyNote: reduceDependencyNote,
  attrSchema: z.object({
    fn: z.enum(["sum", "max", "min", "mean", "prod", "logsumexp"]),
    axes: z.array(z.number().int()),
    keepdim: z.boolean().default(false),
  }),
  arity: { inputs: 1, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("reduce"),
  inferShapes: (inShapes, a) => {
    const sh = inShapes[0];
    const axes = normAxes((a as RAttrs).axes, sh.length);
    const keep = (a as RAttrs).keepdim;
    const out: number[] = [];
    sh.forEach((e, i) => {
      if (axes.includes(i)) {
        if (keep) out.push(1);
      } else out.push(e);
    });
    return [out];
  },
  backward: (_s, outBox, ctx) => {
    const { axes, keepdim } = attrs(ctx);
    const inShape = ctx.inShapes[0];
    const b: Box = [];
    let o = 0;
    for (let ax = 0; ax < inShape.length; ax++) {
      if (axes.includes(ax)) {
        b.push(iv(0, inShape[ax]));
        if (keepdim) o++;
      } else {
        b.push({ ...outBox[o] });
        o++;
      }
    }
    return [fromBox(b)];
  },
  forward: (_s, inBox, ctx) => {
    const { axes, keepdim } = attrs(ctx);
    const b: Box = [];
    for (let ax = 0; ax < inBox.length; ax++) {
      if (axes.includes(ax)) {
        if (keepdim) b.push(iv(0, 1));
      } else b.push({ ...inBox[ax] });
    }
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const { axes, keepdim } = attrs(ctx);
    const inShape = ctx.inShapes[0];
    // Map output coords back to non-reduced input coords.
    const fixed: (number | null)[] = [];
    let o = 0;
    for (let ax = 0; ax < inShape.length; ax++) {
      if (axes.includes(ax)) {
        fixed.push(null);
        if (keepdim) o++;
      } else {
        fixed.push(outIndex[o]);
        o++;
      }
    }
    const deps: number[][] = [];
    const rec = (ax: number, idx: number[]) => {
      if (ax === inShape.length) {
        deps.push(idx.slice());
        return;
      }
      if (fixed[ax] !== null) {
        idx.push(fixed[ax]!);
        rec(ax + 1, idx);
        idx.pop();
      } else {
        for (let v = 0; v < inShape[ax]; v++) {
          idx.push(v);
          rec(ax + 1, idx);
          idx.pop();
        }
      }
    };
    rec(0, []);
    return [deps];
  },
  flopsFor: (_s, outBox, ctx) => {
    const { axes } = attrs(ctx);
    let vol = 1;
    for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
    let red = 1;
    for (const ax of axes) red *= ctx.inShapes[0][ax];
    return vol * red;
  },
};
