import { z } from "zod";
import { Box, Region, empty, fromBox, iv, canonicalize } from "../region";
import { Attrs, DIAG_ENUM_CAP, OpCtx, OpSpec, uniformDTypeOutputs } from "./types";

export type ParsedEquation = { operands: string[][]; output: string[] };

export function parseEquation(eq: string, nInputs?: number): ParsedEquation {
  const clean = eq.replace(/\s+/g, "");
  const m = clean.split("->");
  if (m.length !== 2) throw new Error(`einsum equation "${eq}" must contain "->"`);
  const operands = m[0].split(",").map((s) => s.split(""));
  const output = m[1] === "" ? [] : m[1].split("");
  for (const labs of [...operands, output])
    for (const L of labs)
      if (!/^[a-zA-Z]$/.test(L)) throw new Error(`bad einsum label "${L}" in "${eq}"`);
  if (new Set(output).size !== output.length)
    throw new Error(`einsum output labels must be unique in "${eq}"`);
  const known = new Set(operands.flat());
  for (const L of output)
    if (!known.has(L)) throw new Error(`einsum output label "${L}" not in any operand`);
  if (nInputs !== undefined && operands.length !== nInputs)
    throw new Error(`einsum "${eq}" has ${operands.length} operands but node has ${nInputs} inputs`);
  return { operands, output };
}

function labelExtents(pe: ParsedEquation, inShapes: number[][]): Map<string, number> {
  const ext = new Map<string, number>();
  pe.operands.forEach((labs, i) => {
    if (labs.length !== inShapes[i].length)
      throw new Error(`einsum operand ${i}: ${labs.length} labels vs rank ${inShapes[i].length}`);
    labs.forEach((L, ax) => {
      const e = inShapes[i][ax];
      const prev = ext.get(L);
      if (prev !== undefined && prev !== e)
        throw new Error(`einsum label "${L}": inconsistent extents ${prev} vs ${e}`);
      ext.set(L, e);
    });
  });
  return ext;
}

export function einsumInferShapes(eq: string, inShapes: number[][]): number[][] {
  const pe = parseEquation(eq, inShapes.length);
  const ext = labelExtents(pe, inShapes);
  return [pe.output.map((L) => ext.get(L)!)];
}

export function einsumBackward(eq: string, outBox: Box, ctx: OpCtx): Region[] {
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  const labelIv = new Map<string, { lo: number; hi: number }>();
  pe.output.forEach((L, ax) => labelIv.set(L, outBox[ax]));
  for (const [L, e] of ext) if (!labelIv.has(L)) labelIv.set(L, iv(0, e));

  return pe.operands.map((labs) => {
    const base: Box = labs.map((L) => ({ ...labelIv.get(L)! }));
    const counts = new Map<string, number>();
    labs.forEach((L) => counts.set(L, (counts.get(L) ?? 0) + 1));
    const repeated = [...counts.entries()].filter(([, c]) => c > 1).map(([L]) => L);
    if (repeated.length === 0) return fromBox(base);

    // Diagonal: the true preimage constrains repeated-label axes to equal indices.
    // Enumerate diagonal positions when small; otherwise the rectangular box is a
    // strict superset and must be marked inexact (see IDEA.md §3.1).
    let combos = 1;
    for (const L of repeated) {
      const I = labelIv.get(L)!;
      combos *= Math.max(0, I.hi - I.lo);
    }
    if (combos === 0) return empty(labs.length);
    if (combos > DIAG_ENUM_CAP)
      return {
        boxes: [base],
        exact: false,
        reasons: ["diagonal einsum"],
      };
    const boxes: Box[] = [];
    const rec = (li: number, assign: Map<string, number>) => {
      if (li === repeated.length) {
        const b: Box = labs.map((L, ax) =>
          assign.has(L) ? iv(assign.get(L)!, assign.get(L)! + 1) : { ...base[ax] }
        );
        boxes.push(b);
        return;
      }
      const L = repeated[li];
      const I = labelIv.get(L)!;
      for (let v = I.lo; v < I.hi; v++) {
        assign.set(L, v);
        rec(li + 1, assign);
      }
      assign.delete(L);
    };
    rec(0, new Map());
    return canonicalize({ boxes, exact: true, reasons: [] });
  });
}

export function einsumForward(eq: string, inSlot: number, inBox: Box, ctx: OpCtx): Region[] {
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  const labs = pe.operands[inSlot];
  // Intersection of the box's intervals over each label's occurrences: elements off
  // the diagonal of a repeated label are never read, so they influence nothing.
  const lblIv = new Map<string, { lo: number; hi: number }>();
  for (let ax = 0; ax < labs.length; ax++) {
    const L = labs[ax];
    const prev = lblIv.get(L);
    const cur = inBox[ax];
    lblIv.set(L, prev ? iv(Math.max(prev.lo, cur.lo), Math.min(prev.hi, cur.hi)) : { ...cur });
  }
  for (const I of lblIv.values()) if (I.hi <= I.lo) return [empty(pe.output.length)];
  const outBox: Box = pe.output.map((L) => (lblIv.has(L) ? { ...lblIv.get(L)! } : iv(0, ext.get(L)!)));
  return [fromBox(outBox)];
}

export function einsumOracleDeps(eq: string, outIndex: number[], ctx: OpCtx): number[][][] {
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  const assign = new Map<string, number>();
  pe.output.forEach((L, ax) => assign.set(L, outIndex[ax]));
  const contracted = [...ext.keys()].filter((L) => !assign.has(L));
  const deps: number[][][] = pe.operands.map(() => []);
  const rec = (ci: number) => {
    if (ci === contracted.length) {
      pe.operands.forEach((labs, i) => deps[i].push(labs.map((L) => assign.get(L)!)));
      return;
    }
    const L = contracted[ci];
    for (let v = 0; v < ext.get(L)!; v++) {
      assign.set(L, v);
      rec(ci + 1);
    }
    assign.delete(L);
  };
  rec(0);
  // Deduplicate (repeated contracted labels can produce duplicate tuples).
  return deps.map((list) => {
    const seen = new Set<string>();
    return list.filter((t) => {
      const k = t.join(",");
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  });
}

export function einsumFlops(eq: string, outBox: Box, ctx: OpCtx): number {
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  let vol = 1;
  for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
  let contr = 1;
  const outSet = new Set(pe.output);
  for (const [L, e] of ext) if (!outSet.has(L)) contr *= e;
  const n = pe.operands.length;
  if (n >= 2) return 2 * (n - 1) * vol * contr;
  return contr > 1 ? vol * contr : 0;
}

const eqOf = (attrs: Attrs) => attrs.equation as string;

export const einsumOp: OpSpec = {
  name: "einsum",
  attrSchema: z.object({ equation: z.string() }),
  arity: { inputs: "variadic", outputs: 1 },
  inferDTypes: uniformDTypeOutputs("einsum"),
  inferShapes: (inShapes, attrs) => einsumInferShapes(eqOf(attrs), inShapes),
  backward: (_slot, outBox, ctx) => einsumBackward(eqOf(ctx.attrs), outBox, ctx),
  forward: (inSlot, inBox, ctx) => einsumForward(eqOf(ctx.attrs), inSlot, inBox, ctx),
  oracleDeps: (_slot, outIndex, ctx) => einsumOracleDeps(eqOf(ctx.attrs), outIndex, ctx),
  flopsFor: (_slot, outBox, ctx) => einsumFlops(eqOf(ctx.attrs), outBox, ctx),
};

/** Sugar: build an OpSpec that lowers to a fixed-arity einsum with a shape-derived equation. */
export function einsumSugar(
  name: string,
  nInputs: number,
  makeEq: (inShapes: number[][]) => string
): OpSpec {
  const eqFor = (ctx: OpCtx) => makeEq(ctx.inShapes);
  return {
    name,
    attrSchema: z.object({}).passthrough(),
    arity: { inputs: nInputs, outputs: 1 },
    inferDTypes: uniformDTypeOutputs(name),
    inferShapes: (inShapes) => einsumInferShapes(makeEq(inShapes), inShapes),
    backward: (_s, outBox, ctx) => einsumBackward(eqFor(ctx), outBox, ctx),
    forward: (inSlot, inBox, ctx) => einsumForward(eqFor(ctx), inSlot, inBox, ctx),
    oracleDeps: (_s, outIndex, ctx) => einsumOracleDeps(eqFor(ctx), outIndex, ctx),
    flopsFor: (_s, outBox, ctx) => einsumFlops(eqFor(ctx), outBox, ctx),
  };
}

export const matmulOp = einsumSugar("matmul", 2, (sh) => {
  const [a, b] = sh;
  if (a.length === 2 && b.length === 2) return "mk,kn->mn";
  if (a.length === 3 && b.length === 3) return "bmk,bkn->bmn";
  if (a.length === 4 && b.length === 4) return "bhmk,bhkn->bhmn";
  throw new Error(`matmul: unsupported ranks ${a.length}/${b.length}`);
});

export const bmmOp = einsumSugar("bmm", 2, () => "bmk,bkn->bmn");

export const linearOp = einsumSugar("linear", 2, (sh) => {
  // x[..., K] @ W[N, K]^T
  const r = sh[0].length;
  if (r === 2) return "mk,nk->mn";
  if (r === 3) return "bsk,nk->bsn";
  throw new Error(`linear: unsupported input rank ${r}`);
});
