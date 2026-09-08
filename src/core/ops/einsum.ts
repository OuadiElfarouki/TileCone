import { z } from "zod";
import { Box, Region, coversAxisFully, empty, fromBox, iv, canonicalize } from "../region";
import { Attrs, DependencyNoteDraft, OpCtx, OpSpec, promotingDTypeOutputs, NoteCtx } from "./types";
import { limitsOf } from "./limits";
import { AxisNames } from "./types";
import { Sym } from "../shapes";

type ParsedEquation = { operands: string[][]; output: string[] };

/**
 * Equations are string constants on a node, re-read by `backward`, `forward`,
 * `flopsPerElement` and the note layer on every call. Parsing one is pure, so
 * the result is cached by text; a transformer walks hundreds of einsum nodes
 * per query and was spending most of its einsum time re-splitting the same
 * handful of strings.
 *
 * `nInputs` is deliberately not part of the key: it only gates a check on an
 * already-parsed equation, so caching on the text alone stays correct while
 * every caller still gets its own arity checked.
 */
const equationCache = new Map<string, ParsedEquation | Error>();

function parseEquation(eq: string, nInputs?: number): ParsedEquation {
  const hit = equationCache.get(eq);
  if (hit !== undefined) {
    if (hit instanceof Error) throw hit;
    checkOperandCount(hit, eq, nInputs);
    return hit;
  }
  // Equations are author-supplied, so an editor session can mint arbitrarily
  // many. Clearing wholesale on overflow keeps this a cache rather than a leak;
  // a re-parse is cheap and the working set of one graph is a few dozen.
  if (equationCache.size > 512) equationCache.clear();
  let parsed: ParsedEquation;
  try {
    parsed = parseEquationUncached(eq);
  } catch (e) {
    // Cache the failure too: a malformed equation is re-read just as often as
    // a valid one while the author is still typing it.
    equationCache.set(eq, e as Error);
    throw e;
  }
  // The parse is shared by every caller now, so it must not be writable by one.
  parsed.operands.forEach((labs) => Object.freeze(labs));
  Object.freeze(parsed.operands);
  Object.freeze(parsed.output);
  Object.freeze(parsed);
  equationCache.set(eq, parsed);
  checkOperandCount(parsed, eq, nInputs);
  return parsed;
}

function checkOperandCount(pe: ParsedEquation, eq: string, nInputs?: number): void {
  if (nInputs !== undefined && pe.operands.length !== nInputs)
    throw new Error(`einsum "${eq}" has ${pe.operands.length} operands but node has ${nInputs} inputs`);
}

function parseEquationUncached(eq: string): ParsedEquation {
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
  return { operands, output };
}

/**
 * Label extents depend only on the equation and the input shapes, and both are
 * fixed for the life of a node. `propagationPlan` builds one `OpCtx` per node
 * per resolved graph and reuses it, so the shapes array is a stable object and
 * can key the memo directly; a graph that is recompiled gets fresh arrays and
 * naturally misses.
 *
 * The map is shared, so it is frozen against a caller that would otherwise
 * mutate every future reader's copy.
 */
const extentCache = new WeakMap<
  ParsedEquation,
  WeakMap<number[][], ReadonlyMap<string, number>>
>();

function labelExtents(pe: ParsedEquation, inShapes: number[][]): ReadonlyMap<string, number> {
  // Both keys are objects, so the lookup is two pointer hashes and no string is
  // built. An earlier version keyed on the equation text and spent as much
  // rebuilding that key as it saved.
  let byShapes = extentCache.get(pe);
  if (byShapes) {
    const hit = byShapes.get(inShapes);
    if (hit) return hit;
  } else {
    byShapes = new WeakMap();
    extentCache.set(pe, byShapes);
  }
  const computed = labelExtentsUncached(pe, inShapes);
  byShapes.set(inShapes, computed);
  return computed;
}

function labelExtentsUncached(pe: ParsedEquation, inShapes: number[][]): ReadonlyMap<string, number> {
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
  return Object.freeze(ext);
}

function einsumInferShapes(eq: string, inShapes: number[][]): number[][] {
  const pe = parseEquation(eq, inShapes.length);
  const ext = labelExtents(pe, inShapes);
  return [pe.output.map((L) => ext.get(L)!)];
}

/**
 * The region of one operand implied by an assignment of intervals to labels.
 *
 * Shared by `backward` and `coaccess`, which differ only in where the label
 * intervals come from: an output box for one, another operand's box for the
 * other. The diagonal handling below is the whole reason this is worth
 * factoring; a repeated label constrains its axes to *equal* indices, and
 * getting that right in two places independently is how they drift.
 */
function operandRegion(
  labs: readonly string[],
  labelIv: ReadonlyMap<string, { lo: number; hi: number }>,
  diagEnum: number
): Region {
  const base: Box = labs.map((L) => ({ ...labelIv.get(L)! }));
  const counts = new Map<string, number>();
  labs.forEach((L) => counts.set(L, (counts.get(L) ?? 0) + 1));
  const repeated = [...counts.entries()].filter(([, c]) => c > 1).map(([L]) => L);
  if (repeated.length === 0) return fromBox(base);

  // Diagonal: the true set constrains repeated-label axes to equal indices.
  // Enumerate diagonal positions when small; otherwise block the staircase.
  let combos = 1;
  for (const L of repeated) {
    const I = labelIv.get(L)!;
    combos *= Math.max(0, I.hi - I.lo);
  }
  if (combos === 0) return empty(labs.length);
  if (combos > diagEnum) {
    // Too many diagonal positions to name individually, but the staircase is
    // still worth keeping: chunk each repeated label's range into blocks and
    // emit one box per combination of blocks. Each box is the sub-square a
    // block spans, which contains that block's diagonal cells, so the union is
    // a superset exactly as the single enclosing box was - and for a 300-wide
    // diagonal it is a few hundred elements rather than 90,000.
    const perLabel = Math.max(1, Math.floor(Math.pow(diagEnum, 1 / repeated.length)));
    const blocks = repeated.map((L) => {
      const I = labelIv.get(L)!;
      const width = I.hi - I.lo;
      const groups = Math.min(perLabel, width);
      const size = Math.ceil(width / groups);
      const out: { lo: number; hi: number }[] = [];
      for (let start = I.lo; start < I.hi; start += size)
        out.push(iv(start, Math.min(start + size, I.hi)));
      return out;
    });
    const boxes: Box[] = [];
    const walk = (li: number, assign: Map<string, { lo: number; hi: number }>) => {
      if (li === repeated.length) {
        boxes.push(labs.map((L, ax) => ({ ...(assign.get(L) ?? base[ax]) })));
        return;
      }
      for (const block of blocks[li]) {
        assign.set(repeated[li], block);
        walk(li + 1, assign);
      }
      assign.delete(repeated[li]);
    };
    walk(0, new Map());
    return canonicalize({ boxes, exact: false, reasons: ["diagonal einsum"] });
  }
  const boxes: Box[] = [];
  const rec = (li: number, assign: Map<string, number>) => {
    if (li === repeated.length) {
      boxes.push(
        labs.map((L, ax) =>
          assign.has(L) ? iv(assign.get(L)!, assign.get(L)! + 1) : { ...base[ax] }
        )
      );
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
}

/** @internal Exported for direct oracle-sized tests of diagonal semantics. */
export function einsumBackward(eq: string, outBox: Box, ctx: OpCtx): Region[] {
  const diagEnum = limitsOf(ctx).diagEnum;
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  const labelIv = new Map<string, { lo: number; hi: number }>();
  pe.output.forEach((L, ax) => labelIv.set(L, outBox[ax]));
  for (const [L, e] of ext) if (!labelIv.has(L)) labelIv.set(L, iv(0, e));
  return pe.operands.map((labs) => operandRegion(labs, labelIv, diagEnum));
}

/**
 * Which elements of operand `otherSlot` are multiplied against this box of
 * operand `slot`.
 *
 * A label shared by the two operands is the join between them: fixing it on one
 * side fixes it on the other, because a term of the contraction reads both at
 * the same value of that label. Labels the box does not mention stay full.
 *
 * This is strictly finer than composing `forward` with `backward`. That route
 * asks which *outputs* the box reaches and then what those outputs read, which
 * for `mk,kn->mn` is every element of the second operand: every `C[m,n]` in the
 * row band does read all of `B`. Entanglement asks the narrower question of
 * which elements are combined in the same term, and answers `B[k0:k1, :]`.
 */
function einsumCoaccess(
  eq: string,
  slot: number,
  box: Box,
  otherSlot: number,
  ctx: OpCtx
): Region {
  const diagEnum = limitsOf(ctx).diagEnum;
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  const labs = pe.operands[slot];

  // A label repeated within the operand is read on its diagonal, so the values
  // it can take are those its axes agree on.
  const labelIv = new Map<string, { lo: number; hi: number }>();
  labs.forEach((L, ax) => {
    const prev = labelIv.get(L);
    labelIv.set(
      L,
      prev
        ? iv(Math.max(prev.lo, box[ax].lo), Math.min(prev.hi, box[ax].hi))
        : { ...box[ax] }
    );
  });
  for (const I of labelIv.values())
    if (I.hi <= I.lo) return empty(pe.operands[otherSlot].length);
  for (const [L, e] of ext) if (!labelIv.has(L)) labelIv.set(L, iv(0, e));

  return operandRegion(pe.operands[otherSlot], labelIv, diagEnum);
}

/**
 * Ground truth for entanglement: one entry per term of the contraction, each
 * naming the element every operand contributes to that term.
 *
 * Derived from the definition rather than from `coaccess`, for the same reason
 * `oracleDeps` is derived from the definition rather than from `backward`.
 */
function einsumOracleTerms(eq: string, outIndex: number[], ctx: OpCtx): (number[] | null)[][] {
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  const assign = new Map<string, number>();
  pe.output.forEach((L, ax) => assign.set(L, outIndex[ax]));
  const contracted = [...ext.keys()].filter((L) => !assign.has(L));
  const terms: (number[] | null)[][] = [];
  const rec = (ci: number) => {
    if (ci === contracted.length) {
      terms.push(pe.operands.map((labs) => labs.map((L) => assign.get(L)!)));
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
  return terms;
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
  let vol = 1;
  for (const I of outBox) vol *= Math.max(0, I.hi - I.lo);
  return vol * einsumFlopsPerElement(eq, ctx);
}

/**
 * Cost of one output element of the equation **as written**: one fused
 * contraction, no intermediate materialized.
 *
 * Per contracted position the loop body multiplies the operands together and
 * accumulates, so the count is
 *
 *     (operands - 1) multiplies  +  one add per position that accumulates
 *
 * both scaled by the number of contracted positions. Each term is nil in a real
 * case, which is why they are counted separately rather than folded into one
 * factor: an outer product (`i,j->ij`) contracts nothing and only multiplies,
 * a plain reduction (`ij->i`) has one operand and only adds, and a transpose
 * (`ij->ji`) does neither. For the ordinary two-operand contraction it comes
 * out at the conventional `2 * K` per element, so a GEMM still reports 2MNK.
 *
 * The previous form, `2 * (operands - 1) * positions`, was right only for two
 * operands with something to contract. It charged an outer product two FLOPs
 * for a single multiply, and a three-operand contraction four per position
 * where the fused loop does three.
 *
 * This is deliberately not the cost of an optimal pairwise path. A three-
 * operand einsum written as one node is one fused contraction, and a pairwise
 * decomposition is a different program that happens to compute the same values
 * - usually far more cheaply. Reporting its cost here would answer a question
 * about a program the author did not write; writing the two einsums separately
 * reports it, because then that is what the graph says.
 */
function einsumFlopsPerElement(eq: string, ctx: OpCtx): number {
  const pe = parseEquation(eq, ctx.inShapes.length);
  const ext = labelExtents(pe, ctx.inShapes);
  let positions = 1;
  const outSet = new Set(pe.output);
  for (const [L, e] of ext) if (!outSet.has(L)) positions *= e;
  const multiplies = (pe.operands.length - 1) * positions;
  const adds = positions > 1 ? positions : 0;
  return multiplies + adds;
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
  inferDTypes: promotingDTypeOutputs("einsum"),
  inferShapes: (inShapes, attrs) => einsumInferShapes(eqOf(attrs), inShapes),
  backward: (_slot, outBox, ctx) => einsumBackward(eqOf(ctx.attrs), outBox, ctx),
  forward: (inSlot, inBox, ctx) => einsumForward(eqOf(ctx.attrs), inSlot, inBox, ctx),
  oracleDeps: (_slot, outIndex, ctx) => einsumOracleDeps(eqOf(ctx.attrs), outIndex, ctx),
  coaccess: (slot, box, otherSlot, ctx) =>
    einsumCoaccess(eqOf(ctx.attrs), slot, box, otherSlot, ctx),
  oracleTerms: (_slot, outIndex, ctx) => einsumOracleTerms(eqOf(ctx.attrs), outIndex, ctx),
  flopsFor: (_slot, outBox, ctx) => einsumFlops(eqOf(ctx.attrs), outBox, ctx),
  flopsPerElement: (_slot, ctx) => einsumFlopsPerElement(eqOf(ctx.attrs), ctx),
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
    inferDTypes: promotingDTypeOutputs(name),
    inferShapes: (inShapes) => einsumInferShapes(makeEq(inShapes), inShapes),
    backward: (_s, outBox, ctx) => einsumBackward(eqFor(ctx), outBox, ctx),
    forward: (inSlot, inBox, ctx) => einsumForward(eqFor(ctx), inSlot, inBox, ctx),
    oracleDeps: (_s, outIndex, ctx) => einsumOracleDeps(eqFor(ctx), outIndex, ctx),
    coaccess: (slot, box, otherSlot, ctx) => einsumCoaccess(eqFor(ctx), slot, box, otherSlot, ctx),
    oracleTerms: (_s, outIndex, ctx) => einsumOracleTerms(eqFor(ctx), outIndex, ctx),
    flopsFor: (_s, outBox, ctx) => einsumFlops(eqFor(ctx), outBox, ctx),
    flopsPerElement: (_slot, ctx) => einsumFlopsPerElement(eqFor(ctx), ctx),
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
