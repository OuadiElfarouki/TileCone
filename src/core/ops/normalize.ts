import { z } from "zod";
import { Box, Region, canonicalize, count, coversAxisFully, fromBox, iv } from "../region";
import { normAxes } from "./reduce";
import { DependencyNoteDraft, OpCtx, OpSpec, uniformDTypeOutputs, NoteCtx } from "./types";

type NAttrs = { kind: "layernorm" | "rmsnorm"; axes: number[]; hasWeight: boolean; hasBias: boolean };

function cfg(ctx: OpCtx) {
  const a = ctx.attrs as NAttrs;
  return { ...a, axes: normAxes(a.axes, ctx.inShapes[0].length) };
}

function normalizeFlops(outRegion: Region, ctx: OpCtx): number {
  const { kind, axes, hasWeight, hasBias } = cfg(ctx);
  const axisSet = new Set(axes);
  const expanded = canonicalize({
    boxes: outRegion.boxes.map((box) =>
      box.map((interval, axis) =>
        axisSet.has(axis) ? iv(0, ctx.inShapes[0][axis]) : { ...interval }
      )
    ),
    exact: outRegion.exact,
    reasons: outRegion.reasons,
  });
  const statistics = canonicalize({
    boxes: outRegion.boxes.map((box) =>
      box.map((interval, axis) => (axisSet.has(axis) ? iv(0, 1) : { ...interval }))
    ),
    exact: outRegion.exact,
    reasons: outRegion.reasons,
  });
  const selected = count(outRegion);
  const affineOps = Number(hasWeight) + Number(hasBias);
  if (kind === "layernorm")
    return 4 * count(expanded) + 4 * count(statistics) + (1 + affineOps) * selected;
  return 2 * count(expanded) + 4 * count(statistics) + (1 + affineOps) * selected;
}

/**
 * Inputs: data, [weight], [bias] — weight/bias shaped as the normalized axes' extents.
 * Backward on data: full extent along `axes`, identity elsewhere.
 * Backward on weight/bias: the OUTPUT BOX's intervals restricted to `axes`.
 */
/**
 * Normalization is the interesting middle case: it needs a whole axis like a
 * reduction, but the rows over that axis stay independent of each other, so it
 * is safe to fuse along every *other* axis. Saying only the first half would
 * make it look less fusable than it is.
 */
function normalizeDependencyNote(ctx: NoteCtx): DependencyNoteDraft | null {
  const region = ctx.inRegions[0];
  if (!region) return null;
  const shape = ctx.inShapes[0];
  const attrs = ctx.attrs as NAttrs;
  const axes = normAxes(attrs.axes, shape.length);
  const full = axes.filter((axis) => coversAxisFully(region, axis, shape[axis]));
  if (!full.length) return null;
  const free = shape.map((_, axis) => axis).filter((axis) => !axes.includes(axis));
  const one = full.length === 1;
  const tail = free.length
    ? `but ${free.length === 1 ? `axis ${free[0]} stays` : `axes ${free.join(", ")} stay`} ` +
      `independent — safe to fuse there.`
    : "and no free axis is left to fuse along.";
  return {
    key: `normalize:${attrs.kind}:${full.join(",")}`,
    subject: ctx.outNames[0],
    severity: 2,
    flags: [
      {
        tensorId: ctx.inIds[0],
        text: `full ${one ? `axis ${full[0]}` : `axes ${full.join(", ")}`} — ${attrs.kind} statistics span ${one ? "it" : "them"}`,
      },
    ],
    text:
      `${attrs.kind} takes statistics over ${one ? "axis" : "axes"} ${full.join(", ")} of ` +
      `${ctx.inNames[0]}, so a tile of ${ctx.outNames[0]} needs ` +
      `${one ? "that axis" : "those axes"} complete, ${tail}`,
  };
}

export const normalizeOp: OpSpec = {
  name: "normalize",
  dependencyNote: normalizeDependencyNote,
  attrSchema: z.object({
    kind: z.enum(["layernorm", "rmsnorm"]),
    axes: z.array(z.number().int()),
    hasWeight: z.boolean().default(false),
    hasBias: z.boolean().default(false),
  }),
  arity: { inputs: { min: 1, max: 3 }, outputs: 1 },
  validateArity: (inputCount, _outputCount, attrs) => {
    const typed = attrs as NAttrs;
    const expected = 1 + Number(typed.hasWeight) + Number(typed.hasBias);
    if (inputCount !== expected)
      throw new Error(
        `flags require ${expected} input${expected === 1 ? "" : "s"}, got ${inputCount}`
      );
  },
  inferDTypes: uniformDTypeOutputs("normalize"),
  inferShapes: (inShapes, a) => {
    const attrs = a as NAttrs;
    const sh = inShapes[0];
    const axes = normAxes(attrs.axes, sh.length);
    const paramShape = axes.map((ax) => sh[ax]);
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
  flopsFor: (_s, outBox, ctx) => normalizeFlops(fromBox(outBox), ctx),
  flopsForRegion: (_s, outRegion, ctx) => normalizeFlops(outRegion, ctx),
};
