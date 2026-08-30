import { z } from "zod";
import { Box, Interval, canonicalize, empty, fromBox, full, iv, markInexact } from "../region";
import { normAxis } from "./reduce";
import { OpCtx, OpSpec } from "./types";
import { productBoxes } from "./shape-ops";

type GatherAttrs = { axis: number; indexValues?: number[] };

function cfg(ctx: OpCtx) {
  const a = ctx.attrs as GatherAttrs;
  return { axis: normAxis(a.axis, ctx.inShapes[0].length), indexValues: a.indexValues };
}

function toIntervals(vals: number[]): Interval[] {
  const sorted = [...new Set(vals)].sort((a, b) => a - b);
  const list: Interval[] = [];
  for (const v of sorted) {
    const last = list[list.length - 1];
    if (last && last.hi === v) last.hi = v + 1;
    else list.push(iv(v, v + 1));
  }
  return list;
}

/**
 * index_select / embedding lookup. Inputs: data, indices (1-D i32).
 * The data dependency is data-dependent: without concrete `indexValues` in attrs
 * it is `full()` with `exact: false`; with them the exact preimage is computed.
 */
export const gatherOp: OpSpec = {
  name: "gather",
  attrSchema: z.object({
    axis: z.number().int(),
    indexValues: z.array(z.number().int().min(0)).optional(),
  }),
  arity: { inputs: 2, outputs: 1 },
  inferDTypes: (inDTypes, _attrs, outShapes) => {
    const [data, indices] = inDTypes;
    if (indices !== "i32")
      throw new Error(`gather: indices must be i32, got ${indices}`);
    return outShapes.map(() => data);
  },
  inferShapes: (inShapes, attrs) => {
    const [dataSh, idxSh] = inShapes;
    if (idxSh.length !== 1) throw new Error("gather: indices must be rank-1");
    const ax = normAxis((attrs as GatherAttrs).axis, dataSh.length);
    const vals = (attrs as GatherAttrs).indexValues;
    if (vals && vals.length !== idxSh[0])
      throw new Error(`gather: indexValues length ${vals.length} != indices extent ${idxSh[0]}`);
    if (vals && vals.some((v) => v >= dataSh[ax])) throw new Error("gather: index out of range");
    return [dataSh.map((e, i) => (i === ax ? idxSh[0] : e))];
  },
  backward: (_s, outBox, ctx) => {
    const { axis, indexValues } = cfg(ctx);
    const dataSh = ctx.inShapes[0];
    const idxRegion = fromBox([{ ...outBox[axis] }]);
    if (!indexValues) {
      // Copy non-axis intervals, full extent + inexact on the gathered axis.
      const b: Box = outBox.map((I, ax) => (ax === axis ? iv(0, dataSh[axis]) : { ...I }));
      return [markInexact(fromBox(b), "data-dependent index"), idxRegion];
    }
    const selected = indexValues.slice(outBox[axis].lo, outBox[axis].hi);
    if (selected.length === 0) return [empty(dataSh.length), idxRegion];
    const perAxis: Interval[][] = outBox.map((I, ax) =>
      ax === axis ? toIntervals(selected) : [{ ...I }]
    );
    return [
      canonicalize({ boxes: productBoxes(perAxis), exact: true, reasons: [] }),
      idxRegion,
    ];
  },
  forward: (inSlot, inBox, ctx) => {
    const { axis, indexValues } = cfg(ctx);
    const outSh = ctx.outShapes[0];
    if (inSlot === 1) {
      // indices positions [p0,p1) -> output axis interval, full elsewhere
      const b: Box = outSh.map((e, ax) => (ax === axis ? { ...inBox[0] } : iv(0, e)));
      return [fromBox(b)];
    }
    if (!indexValues) return [markInexact(full(outSh), "data-dependent index")];
    const positions: number[] = [];
    indexValues.forEach((v, p) => {
      if (v >= inBox[axis].lo && v < inBox[axis].hi) positions.push(p);
    });
    if (positions.length === 0) return [empty(outSh.length)];
    const perAxis: Interval[][] = outSh.map((_, ax) =>
      ax === axis ? toIntervals(positions) : [{ ...inBox[ax] }]
    );
    return [canonicalize({ boxes: productBoxes(perAxis), exact: true, reasons: [] })];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const { axis, indexValues } = cfg(ctx);
    if (!indexValues) throw new Error("gather oracle requires indexValues");
    const p = outIndex[axis];
    const dataIdx = outIndex.slice();
    dataIdx[axis] = indexValues[p];
    return [[dataIdx], [[p]]];
  },
  flopsFor: () => 0,
};
