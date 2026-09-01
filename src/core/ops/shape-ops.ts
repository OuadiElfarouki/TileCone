import { z } from "zod";
import { DTYPES, DType } from "../dtypes";
import { Box, Interval, Region, canonicalize, empty, fromBox, iv } from "../region";
import { resolveShape } from "../shapes";
import {
  broadcastBackwardBox,
  broadcastForwardBox,
  broadcastOracleIndex,
} from "./elementwise";
import { normAxis } from "./reduce";
import { OpSpec, STRIDED_ENUM_CAP, uniformDTypeOutputs } from "./types";

const zero = () => 0;

/** Cartesian product of per-axis interval lists into boxes. */
export function productBoxes(perAxis: Interval[][]): Box[] {
  let boxes: Box[] = [[]];
  for (const list of perAxis) {
    const next: Box[] = [];
    for (const b of boxes) for (const I of list) next.push([...b, { ...I }]);
    boxes = next;
  }
  return boxes;
}

function mergeIntervalList(list: Interval[]): Interval[] {
  const s = list.filter((I) => I.hi > I.lo).sort((a, b) => a.lo - b.lo);
  const out: Interval[] = [];
  for (const I of s) {
    const last = out[out.length - 1];
    if (last && I.lo <= last.hi) last.hi = Math.max(last.hi, I.hi);
    else out.push({ ...I });
  }
  return out;
}

// ---------------------------------------------------------------- transpose

export const transposeOp: OpSpec = {
  name: "transpose",
  attrSchema: z.object({ perm: z.array(z.number().int()) }),
  arity: { inputs: 1, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("transpose"),
  inferShapes: (inShapes, attrs) => {
    const perm = attrs.perm as number[];
    const sh = inShapes[0];
    if (perm.length !== sh.length || [...perm].sort((x, y) => x - y).some((v, i) => v !== i))
      throw new Error(`transpose: bad perm [${perm}] for rank ${sh.length}`);
    return [perm.map((p) => sh[p])];
  },
  backward: (_s, outBox, ctx) => {
    const perm = ctx.attrs.perm as number[];
    const inBox: Box = new Array(perm.length);
    perm.forEach((p, i) => (inBox[p] = { ...outBox[i] }));
    return [fromBox(inBox)];
  },
  forward: (_s, inBox, ctx) => {
    const perm = ctx.attrs.perm as number[];
    return [fromBox(perm.map((p) => ({ ...inBox[p] })))];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const perm = ctx.attrs.perm as number[];
    const inIdx: number[] = new Array(perm.length);
    perm.forEach((p, i) => (inIdx[p] = outIndex[i]));
    return [[inIdx]];
  },
  flopsFor: zero,
};

// -------------------------------------------------------------------- slice

type SliceAttrs = { starts: number[]; stops: number[]; steps: number[] };

export const sliceOp: OpSpec = {
  name: "slice",
  attrSchema: z.object({
    starts: z.array(z.number().int().min(0)),
    stops: z.array(z.number().int().min(0)),
    steps: z.array(z.number().int().min(1)),
  }),
  arity: { inputs: 1, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("slice"),
  inferShapes: (inShapes, attrs) => {
    const { starts, stops, steps } = attrs as SliceAttrs;
    const sh = inShapes[0];
    for (const [name, values] of Object.entries({ starts, stops, steps }))
      if (values.length !== sh.length)
        throw new Error(`slice: ${name} has length ${values.length}, expected rank ${sh.length}`);
    return [
      sh.map((e, ax) => {
        const stop = Math.min(stops[ax], e);
        return Math.max(0, Math.ceil((stop - starts[ax]) / steps[ax]));
      }),
    ];
  },
  backward: (_s, outBox, ctx) => {
    const { starts, steps } = ctx.attrs as SliceAttrs;
    const perAxis: Interval[][] = [];
    let stridedCombos = 1;
    outBox.forEach((I, ax) => {
      if (steps[ax] === 1) {
        perAxis.push([iv(starts[ax] + I.lo, starts[ax] + I.hi)]);
      } else {
        perAxis.push([]); // filled below
        stridedCombos *= Math.max(0, I.hi - I.lo);
      }
    });
    let inexact = false;
    outBox.forEach((I, ax) => {
      if (steps[ax] === 1) return;
      if (stridedCombos <= STRIDED_ENUM_CAP) {
        const list: Interval[] = [];
        for (let o = I.lo; o < I.hi; o++) {
          const i = starts[ax] + o * steps[ax];
          list.push(iv(i, i + 1));
        }
        perAxis[ax] = list;
      } else {
        inexact = true;
        perAxis[ax] = [iv(starts[ax] + I.lo * steps[ax], starts[ax] + (I.hi - 1) * steps[ax] + 1)];
      }
    });
    if (perAxis.some((l) => l.length === 0)) return [empty(outBox.length)];
    const r = canonicalize({ boxes: productBoxes(perAxis), exact: !inexact, reasons: inexact ? ["strided slice"] : [] });
    return [r];
  },
  forward: (_s, inBox, ctx) => {
    const { starts, stops, steps } = ctx.attrs as SliceAttrs;
    const outShape = ctx.outShapes[0];
    const b: Box = [];
    for (let ax = 0; ax < inBox.length; ax++) {
      const start = starts[ax], step = steps[ax];
      const stop = Math.min(stops[ax], ctx.inShapes[0][ax]);
      const lo = Math.max(inBox[ax].lo, start);
      const hi = Math.min(inBox[ax].hi, stop);
      if (hi <= lo) return [empty(outShape.length)];
      const oLo = Math.ceil((lo - start) / step);
      const oHi = Math.floor((hi - 1 - start) / step) + 1;
      if (oHi <= oLo) return [empty(outShape.length)];
      b.push(iv(oLo, Math.min(oHi, outShape[ax])));
    }
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const { starts, steps } = ctx.attrs as SliceAttrs;
    return [[outIndex.map((o, ax) => starts[ax] + o * steps[ax])]];
  },
  flopsFor: zero,
};

// ---------------------------------------------------------------------- pad

type PadAttrs = { pads: [number, number][]; mode: "constant" | "reflect" | "replicate" };

/** Per-axis input indices an output index depends on; null => padding constant (no dep). */
function padAxisPreimage(o: number, lo: number, n: number, mode: PadAttrs["mode"]): number | null {
  const j = o - lo;
  if (j >= 0 && j < n) return j;
  if (mode === "constant") return null;
  if (mode === "replicate") return j < 0 ? 0 : n - 1;
  return j < 0 ? -j : 2 * n - 2 - j; // reflect
}

export const padOp: OpSpec = {
  name: "pad",
  attrSchema: z.object({
    pads: z.array(z.tuple([z.number().int().min(0), z.number().int().min(0)])),
    mode: z.enum(["constant", "reflect", "replicate"]).default("constant"),
  }),
  arity: { inputs: 1, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("pad"),
  inferShapes: (inShapes, attrs) => {
    const { pads, mode } = attrs as PadAttrs;
    const sh = inShapes[0];
    if (pads.length !== sh.length) throw new Error("pad: pads rank mismatch");
    if (mode === "reflect")
      sh.forEach((e, ax) => {
        if (pads[ax][0] > e - 1 || pads[ax][1] > e - 1)
          throw new Error(`pad reflect: pad ${pads[ax]} too large for extent ${e}`);
      });
    return [sh.map((e, ax) => e + pads[ax][0] + pads[ax][1])];
  },
  backward: (_s, outBox, ctx) => {
    const { pads, mode } = ctx.attrs as PadAttrs;
    const inShape = ctx.inShapes[0];
    const perAxis: Interval[][] = [];
    for (let ax = 0; ax < inShape.length; ax++) {
      const [lo] = pads[ax];
      const n = inShape[ax];
      const { lo: o0, hi: o1 } = outBox[ax];
      const list: Interval[] = [];
      // interior
      const a = Math.max(o0, lo) - lo;
      const b = Math.min(o1, lo + n) - lo;
      if (b > a) list.push(iv(a, b));
      // left pad segment [o0, min(o1, lo))
      const l1 = Math.min(o1, lo);
      if (l1 > o0) {
        if (mode === "replicate") list.push(iv(0, 1));
        else if (mode === "reflect") list.push(iv(lo - (l1 - 1), lo - o0 + 1));
      }
      // right pad segment [max(o0, lo+n), o1)
      const r0 = Math.max(o0, lo + n);
      if (o1 > r0) {
        if (mode === "replicate") list.push(iv(n - 1, n));
        else if (mode === "reflect")
          list.push(iv(2 * n - 2 - (o1 - 1 - lo), 2 * n - 2 - (r0 - lo) + 1));
      }
      const merged = mergeIntervalList(list);
      if (merged.length === 0) return [empty(inShape.length)];
      perAxis.push(merged);
    }
    return [canonicalize({ boxes: productBoxes(perAxis), exact: true, reasons: [] })];
  },
  forward: (_s, inBox, ctx) => {
    const { pads, mode } = ctx.attrs as PadAttrs;
    const inShape = ctx.inShapes[0];
    const perAxis: Interval[][] = [];
    for (let ax = 0; ax < inShape.length; ax++) {
      const [lo, hi] = pads[ax];
      const n = inShape[ax];
      const { lo: i0, hi: i1 } = inBox[ax];
      const list: Interval[] = [iv(i0 + lo, i1 + lo)];
      if (mode === "replicate") {
        if (i0 === 0 && lo > 0) list.push(iv(0, lo));
        if (i1 === n && hi > 0) list.push(iv(lo + n, lo + n + hi));
      } else if (mode === "reflect") {
        // left-pad outputs o in [0, lo): source i = lo - o, i.e. i in [1, lo]
        const la = Math.max(i0, 1);
        const lb = Math.min(i1, lo + 1);
        if (lb > la) list.push(iv(lo - (lb - 1), lo - la + 1));
        // right-pad outputs o in [lo+n, lo+n+hi): source i = 2n-2-(o-lo), i in [n-1-hi, n-1)
        const ra = Math.max(i0, n - 1 - hi);
        const rb = Math.min(i1, n - 1);
        if (rb > ra) list.push(iv(2 * n - 2 - (rb - 1) + lo, 2 * n - 2 - ra + lo + 1));
      }
      perAxis.push(mergeIntervalList(list));
    }
    return [canonicalize({ boxes: productBoxes(perAxis), exact: true, reasons: [] })];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const { pads, mode } = ctx.attrs as PadAttrs;
    const inShape = ctx.inShapes[0];
    const idx: number[] = [];
    for (let ax = 0; ax < inShape.length; ax++) {
      const v = padAxisPreimage(outIndex[ax], pads[ax][0], inShape[ax], mode);
      if (v === null) return [[]]; // constant padding: no dependency
      idx.push(v);
    }
    return [[idx]];
  },
  flopsFor: zero,
};

// ------------------------------------------------------------ concat / split

export const concatOp: OpSpec = {
  name: "concat",
  attrSchema: z.object({ axis: z.number().int() }),
  arity: { inputs: { min: 1 }, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("concat"),
  inferShapes: (inShapes, attrs) => {
    const ax = normAxis(attrs.axis as number, inShapes[0].length);
    const out = inShapes[0].slice();
    out[ax] = inShapes.reduce((a, s) => a + s[ax], 0);
    inShapes.forEach((s) => {
      s.forEach((e, i) => {
        if (i !== ax && e !== out[i]) throw new Error(`concat: shape mismatch on axis ${i}`);
      });
    });
    return [out];
  },
  backward: (_s, outBox, ctx) => {
    const ax = normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);
    const regions: Region[] = [];
    let ofs = 0;
    for (const sh of ctx.inShapes) {
      const lo = Math.max(outBox[ax].lo, ofs);
      const hi = Math.min(outBox[ax].hi, ofs + sh[ax]);
      if (hi <= lo) regions.push(empty(sh.length));
      else {
        const b = outBox.map((I, i) => (i === ax ? iv(lo - ofs, hi - ofs) : { ...I }));
        regions.push(fromBox(b));
      }
      ofs += sh[ax];
    }
    return regions;
  },
  forward: (inSlot, inBox, ctx) => {
    const ax = normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);
    let ofs = 0;
    for (let i = 0; i < inSlot; i++) ofs += ctx.inShapes[i][ax];
    const b = inBox.map((I, i) => (i === ax ? iv(I.lo + ofs, I.hi + ofs) : { ...I }));
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const ax = normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);
    const deps: number[][][] = ctx.inShapes.map(() => []);
    let ofs = 0;
    for (let k = 0; k < ctx.inShapes.length; k++) {
      const e = ctx.inShapes[k][ax];
      if (outIndex[ax] >= ofs && outIndex[ax] < ofs + e) {
        const idx = outIndex.slice();
        idx[ax] -= ofs;
        deps[k].push(idx);
        break;
      }
      ofs += e;
    }
    return deps;
  },
  flopsFor: zero,
};

export const splitOp: OpSpec = {
  name: "split",
  attrSchema: z.object({ axis: z.number().int(), sizes: z.array(z.number().int().min(1)).min(1) }),
  arity: { inputs: 1, outputs: { min: 1 } },
  validateArity: (_inputCount, outputCount, attrs) => {
    const sizeCount = (attrs.sizes as number[]).length;
    if (outputCount !== sizeCount)
      throw new Error(
        `sizes declares ${sizeCount} outputs, got ${outputCount}`
      );
  },
  inferDTypes: uniformDTypeOutputs("split"),
  inferShapes: (inShapes, attrs) => {
    const sh = inShapes[0];
    const ax = normAxis(attrs.axis as number, sh.length);
    const sizes = attrs.sizes as number[];
    if (sizes.reduce((a, b) => a + b, 0) !== sh[ax])
      throw new Error(`split: sizes [${sizes}] don't sum to extent ${sh[ax]}`);
    return sizes.map((s) => sh.map((e, i) => (i === ax ? s : e)));
  },
  backward: (outSlot, outBox, ctx) => {
    const ax = normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);
    const sizes = ctx.attrs.sizes as number[];
    const ofs = sizes.slice(0, outSlot).reduce((a, b) => a + b, 0);
    const b = outBox.map((I, i) => (i === ax ? iv(I.lo + ofs, I.hi + ofs) : { ...I }));
    return [fromBox(b)];
  },
  forward: (_s, inBox, ctx) => {
    const ax = normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);
    const sizes = ctx.attrs.sizes as number[];
    const out: Region[] = [];
    let ofs = 0;
    for (const s of sizes) {
      const lo = Math.max(inBox[ax].lo, ofs);
      const hi = Math.min(inBox[ax].hi, ofs + s);
      if (hi <= lo) out.push(empty(inBox.length));
      else out.push(fromBox(inBox.map((I, i) => (i === ax ? iv(lo - ofs, hi - ofs) : { ...I }))));
      ofs += s;
    }
    return out;
  },
  oracleDeps: (outSlot, outIndex, ctx) => {
    const ax = normAxis(ctx.attrs.axis as number, ctx.inShapes[0].length);
    const sizes = ctx.attrs.sizes as number[];
    const ofs = sizes.slice(0, outSlot).reduce((a, b) => a + b, 0);
    const idx = outIndex.slice();
    idx[ax] += ofs;
    return [[idx]];
  },
  flopsFor: zero,
};

// ------------------------------------------------------------------- expand

export const expandOp: OpSpec = {
  name: "expand",
  attrSchema: z.object({ shape: z.array(z.union([z.string(), z.number().int().min(1)])) }),
  arity: { inputs: 1, outputs: 1 },
  inferDTypes: uniformDTypeOutputs("expand"),
  inferShapes: (inShapes, attrs, params) => {
    const target = resolveShape(attrs.shape as (string | number)[], params ?? {});
    const inSh = inShapes[0];
    const off = target.length - inSh.length;
    if (off < 0) throw new Error("expand: target rank below input rank");
    inSh.forEach((e, i) => {
      if (e !== 1 && e !== target[i + off])
        throw new Error(`expand: cannot expand extent ${e} to ${target[i + off]}`);
    });
    return [target];
  },
  backward: (_s, outBox, ctx) => [fromBox(broadcastBackwardBox(outBox, ctx.inShapes[0]))],
  forward: (_s, inBox, ctx) => [
    fromBox(broadcastForwardBox(inBox, ctx.inShapes[0], ctx.outShapes[0])),
  ],
  oracleDeps: (_s, outIndex, ctx) => [[broadcastOracleIndex(outIndex, ctx.inShapes[0])]],
  flopsFor: zero,
};

// ---------------------------------------------------------- identity family

export function identityLike(name: string): OpSpec {
  return {
    name,
    attrSchema: z.object({}),
    arity: { inputs: 1, outputs: 1 },
    inferDTypes: uniformDTypeOutputs(name),
    inferShapes: (inShapes) => [inShapes[0].slice()],
    backward: (_s, outBox) => [fromBox(outBox.map((I) => ({ ...I })))],
    forward: (_s, inBox) => [fromBox(inBox.map((I) => ({ ...I })))],
    oracleDeps: (_s, outIndex) => [[outIndex.slice()]],
    flopsFor: zero,
  };
}

export const castOp: OpSpec = {
  ...identityLike("cast"),
  attrSchema: z.object({ dtype: z.enum(DTYPES) }),
  inferDTypes: (_inDTypes, attrs, outShapes) =>
    outShapes.map(() => attrs.dtype as DType),
};
