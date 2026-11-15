import { text } from "./kinds";
import { Box, Region, coversAxisFully, empty, fromBox } from "../region";
import {
  type IterationDomain,
  fibreOf,
  iterationBoxOfInput,
  iterationBoxOfOutput,
  iterationVolume,
  projectInput,
  projectOutput,
  spansReduced,
} from "../iteration";
import {
  Attrs,
  DependencyNoteDraft,
  OpCtx,
  OpSpec,
  promotingDTypeOutputs,
  NoteCtx,
  SymEquality,
} from "./types";
import { limitsOf } from "./limits";
import { AxisNames } from "./types";
import { type Tally, uniformTally } from "../tally";
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
 * The equation as an iteration domain: one loop axis per label, in the order
 * the labels first appear across the operands. `backward`, `forward`,
 * `coaccess`, `multiplicity` and the FLOP count below are all read off it, so
 * they cannot disagree about what a label means. The oracles are not: they are
 * derived from the definition, and are what the domain is checked against.
 *
 * Cached like the extents, on the parsed equation and the shapes array.
 */
const domainCache = new WeakMap<ParsedEquation, WeakMap<number[][], IterationDomain>>();

function domainOf(pe: ParsedEquation, inShapes: number[][]): IterationDomain {
  let byShapes = domainCache.get(pe);
  if (byShapes) {
    const hit = byShapes.get(inShapes);
    if (hit) return hit;
  } else {
    byShapes = new WeakMap();
    domainCache.set(pe, byShapes);
  }
  const ext = labelExtents(pe, inShapes);
  const labels = [...ext.keys()];
  const index = new Map(labels.map((L, i) => [L, i]));
  const outSet = new Set(pe.output);
  const domain: IterationDomain = Object.freeze({
    axes: Object.freeze(
      labels.map((L) =>
        Object.freeze({
          label: L,
          extent: ext.get(L)!,
          role: outSet.has(L) ? ("output" as const) : ("reduced" as const),
        })
      )
    ),
    inputs: Object.freeze(pe.operands.map((labs) => Object.freeze(labs.map((L) => index.get(L)!)))),
    output: Object.freeze(pe.output.map((L) => index.get(L)!)),
  });
  byShapes.set(inShapes, domain);
  return domain;
}

const einsumDomain = (eq: string, ctx: OpCtx): IterationDomain =>
  domainOf(parseEquation(eq, ctx.inShapes.length), ctx.inShapes);

/** @internal Exported for direct oracle-sized tests of diagonal semantics. */
export function einsumBackward(eq: string, outBox: Box, ctx: OpCtx): Region[] {
  const diagEnum = limitsOf(ctx).diagEnum;
  const d = einsumDomain(eq, ctx);
  const iterBox = iterationBoxOfOutput(d, outBox);
  return d.inputs.map((_, slot) => projectInput(d, slot, iterBox, diagEnum));
}

/**
 * How many outputs of the box read each element of operand `slot`.
 *
 * An output element reads an operand element when the two agree on every label
 * they share, so the output labels the operand does not carry are free: each
 * combination of them is a different output reading the same element. For
 * `mk,kn->mn` over an `m x n` box that is `n` for each element of `A` and `m`
 * for each of `B`. A contracted label fixes nothing on the output side, and an
 * output that meets the element in several terms still reads it once.
 *
 * The count is the same on every element read, so the region is `backward`'s,
 * diagonal handling and its fallback included.
 */
function einsumMultiplicity(eq: string, outBox: Box, slot: number, ctx: OpCtx): Tally {
  const d = einsumDomain(eq, ctx);
  const iterBox = iterationBoxOfOutput(d, outBox);
  return uniformTally(
    projectInput(d, slot, iterBox, limitsOf(ctx).diagEnum),
    fibreOf(d, slot, iterBox)
  );
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
  const d = einsumDomain(eq, ctx);
  const iterBox = iterationBoxOfInput(d, slot, box);
  if (!iterBox) return empty(d.inputs[otherSlot].length);
  return projectInput(d, otherSlot, iterBox, limitsOf(ctx).diagEnum);
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
  const d = einsumDomain(eq, ctx);
  // Elements off the diagonal of a repeated label are never read, so they
  // influence nothing.
  const iterBox = iterationBoxOfInput(d, inSlot, inBox);
  return [iterBox ? fromBox(projectOutput(d, iterBox)) : empty(d.output.length)];
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
 * The work of the terms in an iteration box. One term multiplies the operands
 * together and, where the equation contracts anything, accumulates once: the
 * count `einsumFlopsPerElement` states per output element, taken per term. The
 * boxes of a split reduction therefore add up to the whole.
 *
 * A box spanning the reduced axes is charged through `einsumFlops` on its
 * output box. The two products are the same integer, but past 2^53 the order
 * of multiplication decides the last digit, and a whole reduction must report
 * the figure `flopsFor` does.
 */
function einsumIterationFlops(eq: string, iterBox: Box, ctx: OpCtx): number {
  const d = einsumDomain(eq, ctx);
  if (spansReduced(d, iterBox)) return einsumFlops(eq, projectOutput(d, iterBox), ctx);
  let positions = 1;
  for (const axis of d.axes) if (axis.role === "reduced") positions *= axis.extent;
  const perTerm = d.inputs.length - 1 + (positions > 1 ? 1 : 0);
  return iterationVolume(iterBox) * perTerm;
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
      summary: `${ctx.outNames[0]} contracts ${axisLabel}=${extent} in full`,
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

/**
 * Every operand axis wearing a label must have the same extent: that is what
 * a shared label means. `labelled` says whether to name the label, which only
 * helps when the author wrote the equation; a matmul's labels are internal.
 *
 * Every pair is compared, not each against the first: a first operand whose
 * extent is unknown would otherwise leave two provably different later ones
 * unchecked, as in `einsum("i,i,i->i", X, A, B)` with A and B on M and N.
 */
function einsumSymEqualities(
  eq: string,
  inSyms: Sym[][],
  nInputs: number,
  labelled: boolean
): SymEquality[] {
  const pe = parseEquation(eq, nInputs);
  const seen = new Map<string, { slot: number; axis: number }[]>();
  const equalities: SymEquality[] = [];
  pe.operands.forEach((labels, slot) =>
    labels.forEach((label, axis) => {
      const earlier = seen.get(label) ?? [];
      const kind = pe.output.includes(label) ? "shared" : "contracted";
      for (const prior of earlier)
        equalities.push({
          left: inSyms[prior.slot][prior.axis],
          right: inSyms[slot][axis],
          what:
            `the ${kind} axes${labelled ? ` "${label}"` : ""} ` +
            `(input ${prior.slot} axis ${prior.axis}, input ${slot} axis ${axis})`,
          input: slot,
        });
      seen.set(label, [...earlier, { slot, axis }]);
    })
  );
  return equalities;
}

export const einsumOp: OpSpec = {
  name: "einsum",
  attrs: { equation: text() },
  arity: { inputs: { min: 1 }, outputs: 1 },
  inferAxisNames: (inNames, ctx) =>
    einsumAxisNames(eqOf(ctx.attrs), inNames, ctx.inShapes.length),
  inferSymShapes: (inSyms, ctx) =>
    einsumSymShape(eqOf(ctx.attrs), inSyms, ctx, ctx.inShapes.length),
  symEqualities: (inSyms, ctx) =>
    einsumSymEqualities(eqOf(ctx.attrs), inSyms, ctx.inShapes.length, true),
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
  multiplicity: (_s, outBox, inSlot, ctx) => einsumMultiplicity(eqOf(ctx.attrs), outBox, inSlot, ctx),
  coaccess: (slot, box, otherSlot, ctx) =>
    einsumCoaccess(eqOf(ctx.attrs), slot, box, otherSlot, ctx),
  coaccessSeparable: true,
  oracleTerms: (_slot, outIndex, ctx) => einsumOracleTerms(eqOf(ctx.attrs), outIndex, ctx),
  flopsFor: (_slot, outBox, ctx) => einsumFlops(eqOf(ctx.attrs), outBox, ctx),
  flopsPerElement: (_slot, ctx) => einsumFlopsPerElement(eqOf(ctx.attrs), ctx),
  iteration: (ctx) => einsumDomain(eqOf(ctx.attrs), ctx),
  flopsForIteration: (iterBox, ctx) => einsumIterationFlops(eqOf(ctx.attrs), iterBox, ctx),
  dependencyNote: (ctx) => einsumDependencyNote(eqOf(ctx.attrs), ctx),
};

/**
 * For an operation that is an einsum underneath, the first pair of axes that
 * must agree and do not, said by operand and axis rather than by the label of
 * an equation the author never wrote.
 */
function operandMismatch(eq: string, inShapes: number[][]): string | null {
  const [lhs, out] = eq.replace(/\s+/g, "").split("->");
  const seen = new Map<string, { input: number; axis: number; extent: number }>();
  const terms = lhs.split(",");
  for (let input = 0; input < terms.length; input++)
    for (let axis = 0; axis < terms[input].length; axis++) {
      const label = terms[input][axis];
      const extent = inShapes[input]?.[axis];
      if (extent === undefined) continue;
      const first = seen.get(label);
      if (!first) {
        seen.set(label, { input, axis, extent });
        continue;
      }
      if (first.extent === extent) continue;
      const role = out.includes(label) ? "a batch axis" : "the contracted axis";
      return (
        `${role} differs: axis ${first.axis} of input ${first.input} is ${first.extent}, ` +
        `axis ${axis} of input ${input} is ${extent}`
      );
    }
  return null;
}

/** Sugar: build an OpSpec that lowers to a fixed-arity einsum with a shape-derived equation. */
function einsumSugar(
  name: string,
  nInputs: number,
  makeEq: (inShapes: number[][]) => string
): OpSpec {
  const eqFor = (ctx: OpCtx) => makeEq(ctx.inShapes);
  return {
    name,
    attrs: {},
    arity: { inputs: nInputs, outputs: 1 },
    inferAxisNames: (inNames, ctx) =>
      einsumAxisNames(makeEq(ctx.inShapes), inNames, ctx.inShapes.length),
    inferSymShapes: (inSyms, ctx) =>
      einsumSymShape(makeEq(ctx.inShapes), inSyms, ctx, ctx.inShapes.length),
    symEqualities: (inSyms, ctx) =>
      einsumSymEqualities(makeEq(ctx.inShapes), inSyms, ctx.inShapes.length, false),
    inferDTypes: promotingDTypeOutputs(name),
    inferShapes: (inShapes) => {
      const eq = makeEq(inShapes);
      const mismatch = operandMismatch(eq, inShapes);
      if (mismatch) throw new Error(mismatch);
      return einsumInferShapes(eq, inShapes);
    },
    backward: (_s, outBox, ctx) => einsumBackward(eqFor(ctx), outBox, ctx),
    forward: (inSlot, inBox, ctx) => einsumForward(eqFor(ctx), inSlot, inBox, ctx),
    oracleDeps: (_s, outIndex, ctx) => einsumOracleDeps(eqFor(ctx), outIndex, ctx),
    multiplicity: (_s, outBox, inSlot, ctx) => einsumMultiplicity(eqFor(ctx), outBox, inSlot, ctx),
    coaccess: (slot, box, otherSlot, ctx) => einsumCoaccess(eqFor(ctx), slot, box, otherSlot, ctx),
    coaccessSeparable: true,
    oracleTerms: (_s, outIndex, ctx) => einsumOracleTerms(eqFor(ctx), outIndex, ctx),
    flopsFor: (_s, outBox, ctx) => einsumFlops(eqFor(ctx), outBox, ctx),
    flopsPerElement: (_slot, ctx) => einsumFlopsPerElement(eqFor(ctx), ctx),
    iteration: (ctx) => einsumDomain(eqFor(ctx), ctx),
    flopsForIteration: (iterBox, ctx) => einsumIterationFlops(eqFor(ctx), iterBox, ctx),
    dependencyNote: (ctx) => einsumDependencyNote(eqFor(ctx), ctx),
  };
}

export const matmulOp = einsumSugar("matmul", 2, (sh) => {
  const [a, b] = sh;
  if (a.length === 2 && b.length === 2) return "mk,kn->mn";
  if (a.length === 3 && b.length === 3) return "bmk,bkn->bmn";
  if (a.length === 4 && b.length === 4) return "bhmk,bhkn->bhmn";
  throw new Error(
    `input 0 has rank ${a.length} and input 1 rank ${b.length}; both must have rank 2, 3 or 4`
  );
});

export const linearOp = einsumSugar("linear", 2, (sh) => {
  // x[..., K] @ W[N, K]^T
  const r = sh[0].length;
  if (r === 2) return "mk,nk->mn";
  if (r === 3) return "bsk,nk->bsn";
  throw new Error(`input 0 has rank ${r}; it must have rank 2 or 3`);
});
