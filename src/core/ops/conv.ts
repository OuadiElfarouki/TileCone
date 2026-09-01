import { z } from "zod";
import { Box, Interval, Region, canonicalize, empty, fromBox, iv } from "../region";
import { productBoxes } from "./shape-ops";
import { DependencyNoteDraft, NoteCtx, OpCtx, OpSpec, STRIDED_ENUM_CAP, uniformDTypeOutputs } from "./types";
import { sameAxisNames } from "./axis-names";

/**
 * Layout: NC* (batch, channels, spatial...). Weight: [Cout, Cin/groups, *kernel].
 * Dependency semantics for the weight input: an output element depends on every
 * weight element of its (cout, group) slice, including kernel positions that land
 * in padding — matching the MAC loop, which always reads the full filter. The
 * activation dependency, by contrast, only includes in-range input positions.
 */

type ConvAttrs = {
  stride: number[];
  pads: [number, number][];
  dilation: number[];
  groups: number;
};

function outSpatial(n: number, k: number, s: number, d: number, p: [number, number]): number {
  return Math.floor((n + p[0] + p[1] - (k - 1) * d - 1) / s) + 1;
}

function validateSpatialActivationRank(op: "conv" | "pool", rank: number): void {
  if (rank < 3 || rank > 5)
    throw new Error(
      `${op}: activation rank ${rank} is unsupported; expected 3, 4, or 5 (NCW, NCHW, or NCDHW)`
    );
}

/**
 * Preimage of output interval [o0,o1) on one spatial axis.
 * Exact single interval when dilation == 1 && stride <= kernel (windows union
 * contiguously); otherwise enumerate, falling back to an inexact bound (IDEA §3.1).
 */
function spatialBackward(
  o0: number,
  o1: number,
  s: number,
  d: number,
  K: number,
  pLo: number,
  n: number
): { list: Interval[]; exact: boolean } {
  if (o1 <= o0) return { list: [], exact: true };
  if (d === 1 && s <= K) {
    const lo = Math.max(0, o0 * s - pLo);
    const hi = Math.min(n, (o1 - 1) * s + K - pLo);
    return { list: hi > lo ? [iv(lo, hi)] : [], exact: true };
  }
  if ((o1 - o0) * K <= STRIDED_ENUM_CAP) {
    const seen = new Set<number>();
    for (let o = o0; o < o1; o++)
      for (let k = 0; k < K; k++) {
        const i = o * s - pLo + k * d;
        if (i >= 0 && i < n) seen.add(i);
      }
    const sorted = [...seen].sort((a, b) => a - b);
    const list: Interval[] = [];
    for (const i of sorted) {
      const last = list[list.length - 1];
      if (last && last.hi === i) last.hi = i + 1;
      else list.push(iv(i, i + 1));
    }
    return { list, exact: true };
  }
  const lo = Math.max(0, o0 * s - pLo);
  const hi = Math.min(n, (o1 - 1) * s + (K - 1) * d + 1 - pLo);
  return { list: hi > lo ? [iv(lo, hi)] : [], exact: false };
}

/** Image of input interval [i0,i1) on one spatial axis. */
function spatialForward(
  i0: number,
  i1: number,
  s: number,
  d: number,
  K: number,
  pLo: number,
  oN: number
): { list: Interval[]; exact: boolean } {
  if (i1 <= i0) return { list: [], exact: true };
  const oLo = Math.max(0, Math.ceil((i0 + pLo - (K - 1) * d) / s));
  const oHi = Math.min(oN, Math.floor((i1 - 1 + pLo) / s) + 1);
  if (oHi <= oLo) return { list: [], exact: true };
  if (d === 1) return { list: [iv(oLo, oHi)], exact: true };
  if ((oHi - oLo) * K <= STRIDED_ENUM_CAP) {
    const list: Interval[] = [];
    for (let o = oLo; o < oHi; o++) {
      let touches = false;
      for (let k = 0; k < K; k++) {
        const i = o * s - pLo + k * d;
        if (i >= i0 && i < i1) {
          touches = true;
          break;
        }
      }
      if (!touches) continue;
      const last = list[list.length - 1];
      if (last && last.hi === o) last.hi = o + 1;
      else list.push(iv(o, o + 1));
    }
    return { list, exact: true };
  }
  return { list: [iv(oLo, oHi)], exact: false };
}

function convCfg(ctx: OpCtx) {
  const a = ctx.attrs as ConvAttrs;
  const xSh = ctx.inShapes[0];
  const wSh = ctx.inShapes[1];
  const nSp = xSh.length - 2;
  return {
    ...a,
    N: xSh[0],
    Cin: xSh[1],
    Cout: wSh[0],
    kernel: wSh.slice(2),
    spatialIn: xSh.slice(2),
    spatialOut: ctx.outShapes[0].slice(2),
    nSp,
    cpgIn: xSh[1] / a.groups,
    cpgOut: wSh[0] / a.groups,
  };
}

function regionFromPerAxis(perAxis: Interval[][], exact: boolean, reason: string): Region {
  if (perAxis.some((l) => l.length === 0)) return empty(perAxis.length);
  return canonicalize({
    boxes: productBoxes(perAxis),
    exact,
    reasons: exact ? [] : [reason],
  });
}

/**
 * The halo is the whole story for convolution: neighbouring output tiles read
 * overlapping input windows, so tiling is not free the way it is for pointwise
 * work, and stacking layers compounds the overlap.
 */
function convDependencyNote(ctx: NoteCtx): DependencyNoteDraft | null {
  if (!ctx.inRegions[0]) return null;
  const a = ctx.attrs as ConvAttrs;
  const kernel = ctx.inShapes[1].slice(2);
  if (!kernel.length) return null;
  const stride = a.stride ?? [];
  const dilation = a.dilation ?? [];
  // effective window per spatial axis, accounting for dilation
  const reach = kernel.map((k, i) => (k - 1) * (dilation[i] ?? 1) + 1);
  const slack = reach.map((r, i) => r - (stride[i] ?? 1));
  // stride >= reach tiles cleanly; there is no halo to warn about
  if (!slack.some((v) => v > 0)) return null;
  const halo = slack.map((v) => Math.max(0, v)).join("×");
  return {
    key: `conv:${kernel.join("x")}:${stride.join("x")}:${dilation.join("x")}`,
    subject: ctx.outNames[0],
    severity: 1,
    flags: [
      { tensorId: ctx.inIds[0], text: `halo — neighbouring tiles overlap by ${halo}` },
    ],
    text:
      `conv reads a ${reach.join("×")} window of ${ctx.inNames[0]} per output element at ` +
      `stride ${stride.join("×")}, so adjacent tiles of ${ctx.outNames[0]} overlap by ` +
      `${slack.map((v) => Math.max(0, v)).join("×")} on ${ctx.inNames[0]}. That halo is ` +
      `re-read by every neighbouring tile, and stacking layers widens it further.`,
  };
}

export const convOp: OpSpec = {
  name: "conv",
  attrSchema: z.object({
    stride: z.array(z.number().int().min(1)),
    pads: z.array(z.tuple([z.number().int().min(0), z.number().int().min(0)])),
    dilation: z.array(z.number().int().min(1)),
    groups: z.number().int().min(1).default(1),
  }),
  arity: { inputs: 2, outputs: 1 },
  inferAxisNames: (inNames) => [
    [
      // Batch and spatial coordinates remain the activation's axes. The
      // channel coordinate does not: output channels are weight axis 0, not
      // the activation's input-channel axis.
      inNames[0][0],
      inNames[1][0],
      ...inNames[0].slice(2),
    ],
  ],
  dependencyNote: convDependencyNote,
  inferDTypes: uniformDTypeOutputs("conv"),
  inferShapes: (inShapes, attrs) => {
    const a = attrs as ConvAttrs;
    const [xSh, wSh] = inShapes;
    validateSpatialActivationRank("conv", xSh.length);
    if (wSh.length !== xSh.length)
      throw new Error(
        `conv: weight rank ${wSh.length} must match activation rank ${xSh.length}`
      );
    if (xSh[1] % a.groups || wSh[0] % a.groups) throw new Error("conv: channels not divisible by groups");
    if (wSh[1] !== xSh[1] / a.groups)
      throw new Error(`conv: weight Cin ${wSh[1]} != ${xSh[1]}/${a.groups}`);
    const sp = xSh
      .slice(2)
      .map((n, i) => outSpatial(n, wSh[2 + i], a.stride[i], a.dilation[i], a.pads[i]));
    if (sp.some((e) => e <= 0)) throw new Error("conv: non-positive output spatial extent");
    return [[xSh[0], wSh[0], ...sp]];
  },
  backward: (_s, outBox, ctx) => {
    const c = convCfg(ctx);
    const [cLo, cHi] = [outBox[1].lo, outBox[1].hi];
    const g0 = Math.floor(cLo / c.cpgOut);
    const g1 = Math.floor((cHi - 1) / c.cpgOut) + 1;
    // activation
    let exact = true;
    const perAxis: Interval[][] = [
      [{ ...outBox[0] }],
      [iv(g0 * c.cpgIn, g1 * c.cpgIn)],
    ];
    for (let i = 0; i < c.nSp; i++) {
      const r = spatialBackward(
        outBox[2 + i].lo,
        outBox[2 + i].hi,
        c.stride[i],
        c.dilation[i],
        c.kernel[i],
        c.pads[i][0],
        c.spatialIn[i]
      );
      exact = exact && r.exact;
      perAxis.push(r.list);
    }
    const xRegion = regionFromPerAxis(perAxis, exact, "strided conv");
    // weight: cout interval, everything else full
    const wBox: Box = [
      { ...outBox[1] },
      iv(0, c.cpgIn),
      ...c.kernel.map((k) => iv(0, k)),
    ];
    return [xRegion, fromBox(wBox)];
  },
  forward: (inSlot, inBox, ctx) => {
    const c = convCfg(ctx);
    if (inSlot === 0) {
      const g0 = Math.floor(inBox[1].lo / c.cpgIn);
      const g1 = Math.floor((inBox[1].hi - 1) / c.cpgIn) + 1;
      let exact = true;
      const perAxis: Interval[][] = [[{ ...inBox[0] }], [iv(g0 * c.cpgOut, g1 * c.cpgOut)]];
      for (let i = 0; i < c.nSp; i++) {
        const r = spatialForward(
          inBox[2 + i].lo,
          inBox[2 + i].hi,
          c.stride[i],
          c.dilation[i],
          c.kernel[i],
          c.pads[i][0],
          c.spatialOut[i]
        );
        exact = exact && r.exact;
        perAxis.push(r.list);
      }
      return [regionFromPerAxis(perAxis, exact, "strided conv")];
    }
    // weight box -> outputs: full batch & spatial, cout interval
    const b: Box = [iv(0, c.N), { ...inBox[0] }, ...c.spatialOut.map((e) => iv(0, e))];
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const c = convCfg(ctx);
    const [n, co, ...osp] = outIndex;
    const g = Math.floor(co / c.cpgOut);
    const xDeps: number[][] = [];
    const wDeps: number[][] = [];
    const kIdx = new Array(c.nSp).fill(0);
    const walkKernel = (cb: (k: number[]) => void) => {
      const rec = (i: number): void => {
        if (i === c.nSp) return cb(kIdx.slice());
        for (let k = 0; k < c.kernel[i]; k++) {
          kIdx[i] = k;
          rec(i + 1);
        }
      };
      rec(0);
    };
    for (let ciL = 0; ciL < c.cpgIn; ciL++) {
      const ci = g * c.cpgIn + ciL;
      walkKernel((k) => {
        const pos = osp.map((o, i) => o * c.stride[i] - c.pads[i][0] + k[i] * c.dilation[i]);
        if (pos.every((p, i) => p >= 0 && p < c.spatialIn[i])) xDeps.push([n, ci, ...pos]);
      });
      walkKernel((k) => wDeps.push([co, ciL, ...k]));
    }
    return [xDeps, wDeps];
  },
  flopsFor: (_s, outBox, ctx) => {
    const c = convCfg(ctx);
    let vol = 1;
    for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
    return 2 * vol * c.cpgIn * c.kernel.reduce((a, b) => a * b, 1);
  },
};

type PoolAttrs = {
  kind: "max" | "avg";
  kernelShape: number[];
  stride: number[];
  pads: [number, number][];
  dilation: number[];
};

export const poolOp: OpSpec = {
  name: "pool",
  attrSchema: z.object({
    kind: z.enum(["max", "avg"]).default("max"),
    kernelShape: z.array(z.number().int().min(1)),
    stride: z.array(z.number().int().min(1)),
    pads: z.array(z.tuple([z.number().int().min(0), z.number().int().min(0)])),
    dilation: z.array(z.number().int().min(1)).optional(),
  }),
  arity: { inputs: 1, outputs: 1 },
  inferAxisNames: sameAxisNames,
  inferDTypes: uniformDTypeOutputs("pool"),
  inferShapes: (inShapes, attrs) => {
    const a = attrs as PoolAttrs;
    const sh = inShapes[0];
    validateSpatialActivationRank("pool", sh.length);
    const d = a.dilation ?? a.kernelShape.map(() => 1);
    const sp = sh.slice(2).map((n, i) => outSpatial(n, a.kernelShape[i], a.stride[i], d[i], a.pads[i]));
    if (sp.some((e) => e <= 0)) throw new Error("pool: non-positive output spatial extent");
    return [[sh[0], sh[1], ...sp]];
  },
  backward: (_s, outBox, ctx) => {
    const a = ctx.attrs as PoolAttrs;
    const d = a.dilation ?? a.kernelShape.map(() => 1);
    const inSh = ctx.inShapes[0];
    let exact = true;
    const perAxis: Interval[][] = [[{ ...outBox[0] }], [{ ...outBox[1] }]];
    for (let i = 0; i < inSh.length - 2; i++) {
      const r = spatialBackward(
        outBox[2 + i].lo,
        outBox[2 + i].hi,
        a.stride[i],
        d[i],
        a.kernelShape[i],
        a.pads[i][0],
        inSh[2 + i]
      );
      exact = exact && r.exact;
      perAxis.push(r.list);
    }
    return [regionFromPerAxis(perAxis, exact, "strided pool")];
  },
  forward: (_s, inBox, ctx) => {
    const a = ctx.attrs as PoolAttrs;
    const d = a.dilation ?? a.kernelShape.map(() => 1);
    const outSh = ctx.outShapes[0];
    let exact = true;
    const perAxis: Interval[][] = [[{ ...inBox[0] }], [{ ...inBox[1] }]];
    for (let i = 0; i < outSh.length - 2; i++) {
      const r = spatialForward(
        inBox[2 + i].lo,
        inBox[2 + i].hi,
        a.stride[i],
        d[i],
        a.kernelShape[i],
        a.pads[i][0],
        outSh[2 + i]
      );
      exact = exact && r.exact;
      perAxis.push(r.list);
    }
    return [regionFromPerAxis(perAxis, exact, "strided pool")];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const a = ctx.attrs as PoolAttrs;
    const d = a.dilation ?? a.kernelShape.map(() => 1);
    const inSh = ctx.inShapes[0];
    const [n, c, ...osp] = outIndex;
    const deps: number[][] = [];
    const nSp = osp.length;
    const kIdx = new Array(nSp).fill(0);
    const rec = (i: number): void => {
      if (i === nSp) {
        const pos = osp.map((o, j) => o * a.stride[j] - a.pads[j][0] + kIdx[j] * d[j]);
        if (pos.every((p, j) => p >= 0 && p < inSh[2 + j])) deps.push([n, c, ...pos]);
        return;
      }
      for (let k = 0; k < a.kernelShape[i]; k++) {
        kIdx[i] = k;
        rec(i + 1);
      }
    };
    rec(0);
    return [deps];
  },
  flopsFor: (_s, outBox, ctx) => {
    const a = ctx.attrs as PoolAttrs;
    let vol = 1;
    for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
    return vol * a.kernelShape.reduce((x, y) => x * y, 1);
  },
};
