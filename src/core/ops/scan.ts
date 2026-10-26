import { axis, bool } from "./kinds";
import { Box, Region, canonicalize, count, fromBox, iv } from "../region";
import { normAxis } from "./reduce";
import { DependencyNoteDraft, NoteCtx, OpCtx, OpSpec, uniformDTypeOutputs } from "./types";
import { sameAxisNames } from "./axis-names";
import { axisWord } from "./axis-names";
import { sameSymShape } from "./sym-shape";
import { limitsOf } from "./limits";
import { separableTally, type Step } from "../tally";

type ScanAttrs = { axis: number; reverse: boolean };

function cfg(ctx: OpCtx) {
  const a = ctx.attrs as ScanAttrs;
  return { axis: normAxis(a.axis, ctx.inShapes[0].length), reverse: a.reverse };
}

function scanFlops(outRegion: Region, ctx: OpCtx): number {
  const { axis, reverse } = cfg(ctx);
  const extent = ctx.inShapes[0][axis];
  const prefix = canonicalize({
    boxes: outRegion.boxes.map((box) =>
      box.map((interval, i) =>
        i === axis
          ? reverse
            ? iv(interval.lo, extent)
            : iv(0, interval.hi)
          : { ...interval }
      )
    ),
    exact: outRegion.exact,
    reasons: outRegion.reasons,
  });
  return count(prefix);
}

/** Triangular dependency cone: output o depends on inputs [0, o] (or [o, n) reversed). */
/**
 * A scan is neither pointwise nor a clean reduction: the dependency is
 * triangular, so tiles along the scan axis are ordered rather than independent.
 */
function cumsumDependencyNote(ctx: NoteCtx): DependencyNoteDraft | null {
  if (!ctx.inRegions[0]) return null;
  const rank = ctx.inShapes[0].length;
  const axis = normAxis(ctx.attrs.axis as number, rank);
  const reverse = Boolean(ctx.attrs.reverse);
  const extent = ctx.inShapes[0][axis];
  return {
    key: `scan:${axis}:${reverse}`,
    subject: ctx.outNames[0],
    severity: 1,
    flags: [
      {
        tensorId: ctx.inIds[0],
        text: `triangular cone - prefix over axis ${axisWord(ctx.inAxisNames[0], ctx.inDims[0], axis)}`,
      },
    ],
    text:
      `cumulative_sum is a prefix scan along axis ${axisWord(ctx.inAxisNames[0], ctx.inDims[0], axis)} ` +
      `(${extent} wide), so element i of ${ctx.outNames[0]} depends on ` +
      `${reverse ? "every later" : "every earlier"} element of ${ctx.inNames[0]} - the cone is ` +
      `triangular, not rectangular. Tiles along axis ${axisWord(ctx.inAxisNames[0], ctx.inDims[0], axis)} must ` +
      `run in order and carry the running value across the boundary.`,
  };
}

export const cumsumOp: OpSpec = {
  name: "cumulative_sum",
  attrs: { axis: axis(), reverse: bool({ default: false }) },
  arity: { inputs: 1, outputs: 1 },
  inferAxisNames: sameAxisNames,
  inferSymShapes: sameSymShape,
  dependencyNote: cumsumDependencyNote,
  inferDTypes: uniformDTypeOutputs("cumulative_sum"),
  inferShapes: (inShapes, attrs) => {
    normAxis(attrs.axis as number, inShapes[0].length);
    return [inShapes[0].slice()];
  },
  backward: (_s, outBox, ctx) => {
    const { axis, reverse } = cfg(ctx);
    const n = ctx.inShapes[0][axis];
    const b: Box = outBox.map((I, ax) =>
      ax === axis ? (reverse ? iv(I.lo, n) : iv(0, I.hi)) : { ...I }
    );
    return [fromBox(b)];
  },
  forward: (_s, inBox, ctx) => {
    const { axis, reverse } = cfg(ctx);
    const n = ctx.outShapes[0][axis];
    const b: Box = inBox.map((I, ax) =>
      ax === axis ? (reverse ? iv(0, I.hi) : iv(I.lo, n)) : { ...I }
    );
    return [fromBox(b)];
  },
  oracleDeps: (_s, outIndex, ctx) => {
    const { axis, reverse } = cfg(ctx);
    const n = ctx.inShapes[0][axis];
    const deps: number[][] = [];
    const [lo, hi] = reverse ? [outIndex[axis], n] : [0, outIndex[axis] + 1];
    for (let v = lo; v < hi; v++) {
      const idx = outIndex.slice();
      idx[axis] = v;
      deps.push(idx);
    }
    return [deps];
  },
  /* Triangular in counts as well as in extent: output i reads the prefix up to
     i, so an earlier element is read by every output of the box at or after it
     and the count falls by one per step through the box. Build disjoint steps
     directly, grouping the ramp at its maximum count before allocating when
     it exceeds the budget. Work then depends on the cap, not the extent. */
  multiplicity: (_s, outBox, _inSlot, ctx) => {
    const { axis, reverse } = cfg(ctx);
    const n = ctx.inShapes[0][axis];
    const budget = Math.max(1, limitsOf(ctx).tallyPieces);
    let exact = true;
    const perAxis: Step[][] = outBox.map((I, ax) => {
      if (ax !== axis) return [{ lo: I.lo, hi: I.hi, count: 1 }];
      const readers = I.hi - I.lo;
      if (readers <= 0) return [];
      const capped = readers > budget;
      exact = !capped;
      const width = Math.max(1, Math.ceil(readers / budget));
      const steps: Step[] = [];
      for (let lo = I.lo; lo < I.hi; lo += width) {
        const hi = Math.min(I.hi, lo + width);
        // Fold the constant prefix/suffix into the adjacent ramp step.
        steps.push({
          lo: !reverse && lo === I.lo ? 0 : lo,
          hi: reverse && hi === I.hi ? n : hi,
          count: reverse ? hi - I.lo : I.hi - lo,
        });
      }
      return steps;
    });
    return separableTally(perAxis, budget, exact, exact ? [] : ["multiplicity cap"]);
  },
  flopsFor: (_s, outBox, ctx) => scanFlops(fromBox(outBox), ctx),
  flopsForRegion: (_s, outRegion, ctx) => scanFlops(outRegion, ctx),
};
