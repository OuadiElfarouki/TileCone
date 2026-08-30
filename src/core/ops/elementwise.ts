import { z } from "zod";
import { Box, fromBox, iv } from "../region";
import { OpSpec, uniformDTypeOutputs } from "./types";

export function broadcastShapes(shapes: number[][]): number[] {
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
  attrSchema: z.object({ fn: z.string(), nary: z.number().int().min(1) }),
  arity: { inputs: "variadic", outputs: 1 },
  inferDTypes: uniformDTypeOutputs("elementwise"),
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
    return vol * Math.max(1, ctx.inShapes.length - 1 + (["exp", "gelu", "silu", "tanh", "sigmoid", "sqrt", "rsqrt"].includes(ctx.attrs.fn as string) ? 4 : 0));
  },
};
