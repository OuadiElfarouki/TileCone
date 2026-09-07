import { z } from "zod";
import { Box, fromBox, iv } from "../region";
import { OpCtx, OpSpec, promotingDTypeOutputs } from "./types";
import { broadcastAxisNames } from "./axis-names";
import { broadcastSymShape } from "./sym-shape";

/**
 * The elementwise functions this graph understands, with what each one costs
 * and how many operands it takes.
 *
 * This table is the definition, not a convenience: `attrSchema` enumerates its
 * keys, so an unrecognized `fn` is a hard error with a did-you-mean rather than
 * a silently accepted string that quietly costs one FLOP. The DSL's call-name
 * sugar reads the same table, so the two cannot drift.
 *
 * `unitCost` is the cost of applying the function once per output element,
 * beyond the `inputs - 1` combining operations. Transcendentals are charged 4,
 * a conventional stand-in for a polynomial approximation rather than a measured
 * figure; what matters is that they are not charged as one add.
 *
 * `maxInputs: null` marks an associative function that may be written n-ary.
 * Non-associative ones are pinned to exactly two operands, so `div(a, b, c)` is
 * rejected instead of silently meaning something.
 */
export type ElementwiseFnSpec = {
  minInputs: number;
  maxInputs: number | null;
  unitCost: number;
};

export const ELEMENTWISE_FNS: Record<string, ElementwiseFnSpec> = {
  // associative binary-or-more
  add: { minInputs: 2, maxInputs: null, unitCost: 0 },
  mul: { minInputs: 2, maxInputs: null, unitCost: 0 },
  maximum: { minInputs: 2, maxInputs: null, unitCost: 0 },
  minimum: { minInputs: 2, maxInputs: null, unitCost: 0 },
  // strictly binary
  sub: { minInputs: 2, maxInputs: 2, unitCost: 0 },
  div: { minInputs: 2, maxInputs: 2, unitCost: 0 },
  pow: { minInputs: 2, maxInputs: 2, unitCost: 4 },
  // unary
  relu: { minInputs: 1, maxInputs: 1, unitCost: 0 },
  neg: { minInputs: 1, maxInputs: 1, unitCost: 0 },
  abs: { minInputs: 1, maxInputs: 1, unitCost: 0 },
  exp: { minInputs: 1, maxInputs: 1, unitCost: 4 },
  log: { minInputs: 1, maxInputs: 1, unitCost: 4 },
  sqrt: { minInputs: 1, maxInputs: 1, unitCost: 4 },
  rsqrt: { minInputs: 1, maxInputs: 1, unitCost: 4 },
  sigmoid: { minInputs: 1, maxInputs: 1, unitCost: 4 },
  tanh: { minInputs: 1, maxInputs: 1, unitCost: 4 },
  gelu: { minInputs: 1, maxInputs: 1, unitCost: 4 },
  silu: { minInputs: 1, maxInputs: 1, unitCost: 4 },
};

export const ELEMENTWISE_FN_NAMES = Object.keys(ELEMENTWISE_FNS) as [string, ...string[]];

function broadcastShapes(shapes: number[][]): number[] {
  const rank = Math.max(...shapes.map((s) => s.length));
  const out: number[] = [];
  for (let i = 0; i < rank; i++) {
    let e = 1;
    for (const s of shapes) {
      const d = s[s.length - rank + i];
      if (d === undefined || d === 1) continue;
      if (e !== 1 && e !== d) throw new Error(`broadcast mismatch: ${e} vs ${d}`);
      e = d;
    }
    out.push(e);
  }
  return out;
}

/** Output box -> input box under NumPy broadcasting (trailing alignment). */
export function broadcastBackwardBox(outBox: Box, inShape: number[]): Box {
  const off = outBox.length - inShape.length;
  return inShape.map((e, ax) => (e === 1 ? iv(0, 1) : { ...outBox[ax + off] }));
}

/** Input box -> output box under NumPy broadcasting. */
export function broadcastForwardBox(inBox: Box, inShape: number[], outShape: number[]): Box {
  const off = outShape.length - inShape.length;
  return outShape.map((e, ax) => {
    if (ax < off) return iv(0, e);
    const inAx = ax - off;
    return inShape[inAx] === 1 && e > 1 ? iv(0, e) : { ...inBox[inAx] };
  });
}

export function broadcastOracleIndex(outIndex: number[], inShape: number[]): number[] {
  const off = outIndex.length - inShape.length;
  return inShape.map((e, ax) => (e === 1 ? 0 : outIndex[ax + off]));
}

export const elementwiseOp: OpSpec = {
  name: "elementwise",
  attrSchema: z.object({
    fn: z.enum(ELEMENTWISE_FN_NAMES),
    nary: z.number().int().min(1),
  }),
  arity: { inputs: { min: 1 }, outputs: 1 },
  inferAxisNames: (inNames, ctx) => [
    broadcastAxisNames(inNames, ctx.inShapes, ctx.outShapes[0]),
  ],
  inferSymShapes: (inSyms, ctx) => [broadcastSymShape(inSyms, ctx)],
  validateArity: (inputCount, _outputCount, attrs) => {
    const nary = attrs.nary as number;
    if (nary !== inputCount)
      throw new Error(`nary=${nary} does not match ${inputCount} input${inputCount === 1 ? "" : "s"}`);
    // The function's own arity, which `nary` alone never checked: `relu(a, b)`
    // and `div(a, b, c)` both used to resolve and quietly compute something the
    // author did not write.
    const fn = attrs.fn as string;
    const shape = ELEMENTWISE_FNS[fn];
    if (!shape) return; // unknown fn is already an attribute error
    if (inputCount < shape.minInputs || (shape.maxInputs !== null && inputCount > shape.maxInputs))
      throw new Error(
        `${fn} takes ${
          shape.maxInputs === null
            ? `at least ${shape.minInputs} inputs`
            : shape.minInputs === shape.maxInputs
              ? `exactly ${shape.minInputs} input${shape.minInputs === 1 ? "" : "s"}`
              : `${shape.minInputs} to ${shape.maxInputs} inputs`
        }, got ${inputCount}`
      );
  },
  inferDTypes: promotingDTypeOutputs("elementwise"),
  inferShapes: (inShapes) => [broadcastShapes(inShapes)],
  backward: (_slot, outBox, ctx) =>
    ctx.inShapes.map((sh) => fromBox(broadcastBackwardBox(outBox, sh))),
  forward: (inSlot, inBox, ctx) => [
    fromBox(broadcastForwardBox(inBox, ctx.inShapes[inSlot], ctx.outShapes[0])),
  ],
  oracleDeps: (_slot, outIndex, ctx) =>
    ctx.inShapes.map((sh) => [broadcastOracleIndex(outIndex, sh)]),
  flopsFor: (_slot, outBox, ctx) => {
    let vol = 1;
    for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
    return vol * elementwiseFlopsPerElement(ctx);
  },
  flopsPerElement: (_slot, ctx) => elementwiseFlopsPerElement(ctx),
};

function elementwiseFlopsPerElement(ctx: OpCtx): number {
  const spec = ELEMENTWISE_FNS[ctx.attrs.fn as string];
  // `inputs - 1` combining operations plus the function's own cost, and never
  // zero: a unary `relu` still touches every element it produces.
  return Math.max(1, ctx.inShapes.length - 1 + (spec?.unitCost ?? 0));
}
