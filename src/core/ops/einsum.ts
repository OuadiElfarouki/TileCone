import { z } from "zod";
import { Box, Region, coversAxisFully, empty, fromBox, iv, canonicalize } from "../region";
import { Attrs, DependencyNoteDraft, DIAG_ENUM_CAP, OpCtx, OpSpec, uniformDTypeOutputs, NoteCtx } from "./types";
import { AxisNames } from "./types";
import { Sym } from "../shapes";

type ParsedEquation = { operands: string[][]; output: string[] };

function parseEquation(eq: string, nInputs?: number): ParsedEquation {
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

function einsumInferShapes(eq: string, inShapes: number[][]): number[][] {
  const pe = parseEquation(eq, inShapes.length);
  const ext = labelExtents(pe, inShapes);
  return [pe.output.map((L) => ext.get(L)!)];
}

/** @internal Exported for direct oracle-sized tests of diagonal semantics. */
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

function einsumForward(eq: string, inSlot: number, inBox: Box, ctx: OpCtx): Region[] {
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

function einsumOracleDeps(eq: string, outIndex: number[], ctx: OpCtx): number[][][] {
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

function einsumFlops(eq: string, outBox: Box, ctx: OpCtx): number {
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


/**
 * Contraction notes. A contracted label is one that appears in the operands but
 * not the output, so every element of the output sums over its whole extent.
 *
 * The note is only emitted when the cone *demonstrably* pulled that whole
 * extent: `backward` is free to return something narrower for a degenerate
 * selection, and claiming a full contraction that did not happen would be a
 * statement the picture does not support.
 */
function einsumDependencyNote(eq: string, ctx: NoteCtx): DependencyNoteDraft | null {
  let pe: ParsedEquation;
  try {
    pe = parseEquation(eq, ctx.inShapes.length);
  } catch {
    return null; // a malformed equation is diagnosed elsewhere; say nothing here
  }
  const outLabels = new Set(pe.output);
  const contracted: string[] = [];
  for (const operand of pe.operands)
    for (const label of operand)
      if (!outLabels.has(label) && !contracted.includes(label)) contracted.push(label);
  if (!contracted.length) return null;

  for (const label of contracted) {
    // every input slot carrying this label, with the axis it sits on
    const carriers: { slot: number; axis: number; extent: number }[] = [];
    pe.operands.forEach((operand, slot) => {
      const axis = operand.indexOf(label);
      if (axis >= 0) carriers.push({ slot, axis, extent: ctx.inShapes[slot][axis] });
    });
    if (!carriers.length) continue;
    const extent = carriers[0].extent;

    const pulledInFull = carriers.filter(({ slot, axis }) => {
      const region = ctx.inRegions[slot];
      return region && coversAxisFully(region, axis, ctx.inShapes[slot][axis]);
    });
    if (pulledInFull.length !== carriers.length) continue;

    // Prefer what the source called this axis over the einsum label, which is
    // internal to the equation. The axis's own name comes first, then the
    // dimension it was declared with: `emb` says what the axis is, `H*D` only
    // says how wide it is, and a produced operand has no declared shape at all.
    const named = carriers
      .map(({ slot, axis }) => ctx.inAxisNames[slot]?.[axis])
      .find((name) => name !== undefined);
    const declared = carriers
      .map(({ slot, axis }) => ctx.inDims[slot]?.[axis])
      .find((dim) => typeof dim === "string") as string | undefined;
    const axisLabel = named ?? declared ?? label;

    const names = carriers.map(({ slot }) => ctx.inNames[slot]);
    const listed =
      names.length === 2 ? `${names[0]} and ${names[1]}` : names.join(", ");
    return {
      // Equal labels and extents are not enough to make two contractions one
      // constraint: a later E-wide matmul is independent of an earlier E-wide
      // projection. Parallel projections merge when they contract the same
      // carrier (for example Q/K/V all reading X's embedding axis).
      key: `contract:${axisLabel}:${extent}:${ctx.inIds[carriers[0].slot]}`,
      subject: ctx.outNames[0],
      severity: 3,
      flags: carriers.map(({ slot, axis }) => ({
        tensorId: ctx.inIds[slot],
        text: `full ${axisLabel} on axis ${axis} : contracted, never tiled`,
      })),
      // Phrased without an article before the axis name on purpose: the label
      // is user-supplied, so "a K" / "an E" cannot be chosen ahead of time.
      text:
        `${ctx.outNames[0]} contracts ${axisLabel}=${extent} in full. One tile of ` +
        `${ctx.outNames[0]} pulls the complete ${axisLabel} extent of ${listed}. A fused ` +
        `kernel must either stage ${axisLabel} or accumulate across it.`,
    };
  }
  return null;
}

const eqOf = (attrs: Attrs) => attrs.equation as string;

/**
 * An einsum output axis is the same axis as any operand axis sharing its label,
 * so a label carries its name through the contraction. Labels that appear only
 * on the operands are contracted away and take their names with them.
 */
function einsumAxisNames(eq: string, inNames: AxisNames[], nInputs: number): AxisNames[] {
  const pe = parseEquation(eq, nInputs);
  return [
    pe.output.map((label) => {
      for (let slot = 0; slot < pe.operands.length; slot++) {
        const axis = pe.operands[slot].indexOf(label);
        if (axis >= 0 && inNames[slot]?.[axis] !== undefined) return inNames[slot][axis];
      }
      return undefined;
    }),
  ];
}

/** An output label's extent is the extent of any operand axis wearing it, so a
 * contraction carries `M` and `N` through even though `K` disappears. */
function einsumSymShape(eq: string, inSyms: Sym[][], ctx: OpCtx, nInputs: number): Sym[][] {
  const pe = parseEquation(eq, nInputs);
  return [
    pe.output.map((label, outAxis) => {
      for (let slot = 0; slot < pe.operands.length; slot++) {
        const axis = pe.operands[slot].indexOf(label);
        if (axis >= 0 && typeof inSyms[slot]?.[axis] === "string") return inSyms[slot][axis];
      }
      return ctx.outShapes[0][outAxis];
    }),
  ];
}

export const einsumOp: OpSpec = {
  name: "einsum",
  attrSchema: z.object({ equation: z.string() }),
  arity: { inputs: { min: 1 }, outputs: 1 },
  inferAxisNames: (inNames, ctx) =>
    einsumAxisNames(eqOf(ctx.attrs), inNames, ctx.inShapes.length),
  inferSymShapes: (inSyms, ctx) =>
    einsumSymShape(eqOf(ctx.attrs), inSyms, ctx, ctx.inShapes.length),
  validateArity: (inputCount, _outputCount, attrs) => {
    const parts = eqOf(attrs).replace(/\s+/g, "").split("->");
    if (parts.length !== 2) return; // equation syntax is diagnosed by shape inference
    const operandCount = parts[0].split(",").length;
    if (operandCount !== inputCount)
      throw new Error(
        `equation declares ${operandCount} operand${operandCount === 1 ? "" : "s"}, got ${inputCount} input${inputCount === 1 ? "" : "s"}`
      );
  },
  inferDTypes: uniformDTypeOutputs("einsum"),
  inferShapes: (inShapes, attrs) => einsumInferShapes(eqOf(attrs), inShapes),
  backward: (_slot, outBox, ctx) => einsumBackward(eqOf(ctx.attrs), outBox, ctx),
  forward: (inSlot, inBox, ctx) => einsumForward(eqOf(ctx.attrs), inSlot, inBox, ctx),
  oracleDeps: (_slot, outIndex, ctx) => einsumOracleDeps(eqOf(ctx.attrs), outIndex, ctx),
  flopsFor: (_slot, outBox, ctx) => einsumFlops(eqOf(ctx.attrs), outBox, ctx),
  dependencyNote: (ctx) => einsumDependencyNote(eqOf(ctx.attrs), ctx),
};

/** Sugar: build an OpSpec that lowers to a fixed-arity einsum with a shape-derived equation. */
function einsumSugar(
  name: string,
  nInputs: number,
  makeEq: (inShapes: number[][]) => string
): OpSpec {
  const eqFor = (ctx: OpCtx) => makeEq(ctx.inShapes);
  return {
    name,
    attrSchema: z.object({}),
    arity: { inputs: nInputs, outputs: 1 },
    inferAxisNames: (inNames, ctx) =>
      einsumAxisNames(makeEq(ctx.inShapes), inNames, ctx.inShapes.length),
    inferSymShapes: (inSyms, ctx) =>
      einsumSymShape(makeEq(ctx.inShapes), inSyms, ctx, ctx.inShapes.length),
    inferDTypes: uniformDTypeOutputs(name),
    inferShapes: (inShapes) => einsumInferShapes(makeEq(inShapes), inShapes),
    backward: (_s, outBox, ctx) => einsumBackward(eqFor(ctx), outBox, ctx),
    forward: (inSlot, inBox, ctx) => einsumForward(eqFor(ctx), inSlot, inBox, ctx),
    oracleDeps: (_s, outIndex, ctx) => einsumOracleDeps(eqFor(ctx), outIndex, ctx),
    flopsFor: (_s, outBox, ctx) => einsumFlops(eqFor(ctx), outBox, ctx),
    dependencyNote: (ctx) => einsumDependencyNote(eqFor(ctx), ctx),
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
