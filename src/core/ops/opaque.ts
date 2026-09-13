import { z } from "zod";
import { fromBox, Region } from "../region";
import { resolveShape, Shape } from "../shapes";
import { promoteDTypes, DTYPES, DType } from "../dtypes";
import { OpSpec, Attrs } from "./types";

/**
 * An operation whose shapes are known and whose semantics are not.
 *
 * This is the escape hatch an importer needs. Without it a model containing one
 * unrecognised operation cannot be loaded at all: `resolveGraphCollecting`
 * prunes the failing node and everything downstream of it, which is right for a
 * line of hand-written DSL - one error, no invented cascade - and wrong for a
 * four-hundred-node graph, where it silently answers a question about a
 * different, shorter model. A cone that stops early is a *subset* of the truth,
 * and a subset is the one thing this engine must never produce.
 *
 * So an unknown operation becomes a barrier rather than a wall: every output
 * element is assumed to read every input element, and every input element to
 * reach every output. That is the weakest true statement about an operation
 * nobody has described, it is a superset by construction, and it is marked
 * inexact with the original name as its reason, so the paint hatches it and the
 * figures above it read as bounds. Narrowing it is then just implementing the
 * operation for real; nothing downstream has to change when that happens.
 *
 * What it cannot do is invent work. `flopsFor` is zero because the arithmetic
 * of an undescribed operation is unknown, so FLOPs under a barrier are a floor,
 * not the bound the other figures are. The dependency note says so where a
 * reader will see it.
 */

const shapeSchema = z.array(z.union([z.string(), z.number().int().min(1)]));

export type OpaqueAttrs = {
  /** The operation this stands in for, in its source vocabulary. */
  op: string;
  /** One shape per output: an importer knows them even when it knows nothing else. */
  shapes: Shape[];
  /** Outputs' dtype when it is not the promotion of the inputs (argmax, cast-like ops). */
  dtype?: DType;
};

const attrSchema = z.object({
  op: z.string().min(1),
  shapes: z.array(shapeSchema).min(1),
  dtype: z.enum(DTYPES).optional(),
});

const attrsOf = (attrs: Attrs) => attrs as unknown as OpaqueAttrs;

/** Every element of a tensor, as the one region that cannot understate. */
const whole = (shape: number[], reason: string): Region => ({
  ...fromBox(shape.map((extent) => ({ lo: 0, hi: extent }))),
  exact: false,
  reasons: [reason],
});

const reasonFor = (attrs: Attrs) => `opaque op "${attrsOf(attrs).op}"`;

export const opaqueOp: OpSpec = {
  name: "opaque",
  attrSchema,
  arity: { inputs: { min: 1 }, outputs: { min: 1 } },
  displayName: (attrs) => attrsOf(attrs).op,

  validateArity(_inputCount, outputCount, attrs) {
    const declared = attrsOf(attrs).shapes.length;
    if (declared !== outputCount)
      throw new Error(`declares ${declared} output shape(s) for ${outputCount} output(s)`);
  },

  inferShapes: (_inShapes, attrs, params) =>
    attrsOf(attrs).shapes.map((shape) => resolveShape(shape, params ?? {})),

  /* Neither axis names nor symbolic extents cross a barrier: an operation that
     cannot say what it computes cannot claim an output axis is the same axis as
     an input one. Both hooks are therefore deliberately absent. */

  inferDTypes: (inDTypes, attrs, outShapes) => {
    const declared = attrsOf(attrs).dtype;
    if (declared) return outShapes.map(() => declared);
    if (!inDTypes.length) throw new Error("opaque: expected at least one input dtype");
    return outShapes.map(() => promoteDTypes(inDTypes));
  },

  backward: (_outSlot, _outBox, ctx) =>
    ctx.inShapes.map((shape) => whole(shape, reasonFor(ctx.attrs))),

  forward: (_inSlot, _inBox, ctx) =>
    ctx.outShapes.map((shape) => whole(shape, reasonFor(ctx.attrs))),

  /* The oracle's ground truth is the definition this op is given, not a guess
     at the operation it stands for: every output element reads every input
     element. The regions above equal that set rather than exceeding it, and are
     still marked inexact - they bound an unknown operation, and saying "exact"
     would assert that a real batch norm genuinely reads its whole input. */
  oracleDeps: (_outSlot, _outIndex, ctx) =>
    ctx.inShapes.map((shape) => {
      const deps: number[][] = [];
      const walk = (axis: number, index: number[]) => {
        if (axis === shape.length) return void deps.push(index.slice());
        for (let v = 0; v < shape[axis]; v++) walk(axis + 1, [...index, v]);
      };
      walk(0, []);
      return deps;
    }),

  /** Unknown work, reported as none. See the note in the header. */
  flopsFor: () => 0,
  flopsPerElement: () => 0,

  dependencyNote: (ctx) => {
    const { op } = attrsOf(ctx.attrs);
    return {
      text:
        ` ${op} is carried as a barrier: its semantics are not modelled, so one output element` +
        ` is assumed to read every input element in full. Everything through it is a bound, and` +
        ` its own arithmetic is not counted.`,
      key: `opaque:${op}`,
      subject: ctx.outIds[0] ?? ctx.inIds[0],
      severity: 3,
      flags: ctx.inIds.map((tensorId) => ({ tensorId, text: `read in full by ${op}` })),
    };
  },
};
