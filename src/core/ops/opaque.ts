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
  /**
   * One dtype per output, where they are not the promotion of the inputs.
   *
   * Per output, not one for all of them, because real operations do not have a
   * single output type: `TopK` returns f32 values beside i64 indices, `MaxPool`
   * an optional i64 index tensor beside its values, `LayerNormalization` a mean
   * and an inverse standard deviation beside the normalised result. One dtype
   * for all outputs could describe none of those, and byte estimates are
   * measured per tensor from that tensor's own dtype, so getting it wrong
   * misreports the footprint of every one of them.
   *
   * Parallel to `shapes` rather than a list of `{ shape, dtype }` records,
   * which is what this wants to be: the DSL has no record literal, and adding
   * one to the grammar to serve a single attribute is the kind of change the
   * import plan already declined to make for the printer's sake. The lengths
   * are checked against the output count instead, so the pairing cannot come
   * apart unnoticed.
   *
   * A hole is an output that takes the promotion of the inputs. That is a
   * convenience for a hand-authored barrier, where the author knows the result
   * type is ordinary. A converter should not use it: an unknown source
   * operation's result type cannot be inferred safely from its inputs, so
   * anything generating barriers is expected to state one dtype per output.
   */
  dtypes?: (DType | null)[];
  /**
   * Where the operation came from, kept because the name alone is ambiguous.
   *
   * `domain` distinguishes a vendor's `Attention` from anyone else's, `opset`
   * says which version of the operation was meant - the same name means
   * different things across versions, and what is a barrier at one opset may be
   * mappable at another - and `sourceName` is the node's own name in the model,
   * which survives even when the converter has to rename the node to keep ids
   * unique. None of it changes what the barrier claims; all of it is what a
   * reader needs to decide whether implementing this operation is worth it.
   */
  domain?: string;
  opset?: number;
  sourceName?: string;
};

const attrSchema = z.object({
  op: z.string().min(1),
  shapes: z.array(shapeSchema).min(1),
  dtypes: z.array(z.enum(DTYPES).nullable()).optional(),
  domain: z.string().optional(),
  opset: z.number().int().min(1).optional(),
  sourceName: z.string().optional(),
});

/** ONNX's default domain, written either way, where an operation needs no qualifier. */
const DEFAULT_DOMAINS = ["", "ai.onnx"];

const attrsOf = (attrs: Attrs) => attrs as unknown as OpaqueAttrs;

/** Every element of a tensor, as the one region that cannot understate. */
const whole = (shape: number[], reason: string): Region => ({
  ...fromBox(shape.map((extent) => ({ lo: 0, hi: extent }))),
  exact: false,
  reasons: [reason],
});

/**
 * The operation's name, qualified by its domain only where that says something.
 *
 * One helper for the card, the approximation reason and the note, so a reader
 * who sees `com.microsoft.Attention` on a tensor's reasons finds the same words
 * on the operation that put it there.
 */
const qualifiedName = (attrs: Attrs): string => {
  const { op, domain } = attrsOf(attrs);
  return domain && !DEFAULT_DOMAINS.includes(domain) ? `${domain}.${op}` : op;
};

const reasonFor = (attrs: Attrs) => `opaque op "${qualifiedName(attrs)}"`;

export const opaqueOp: OpSpec = {
  name: "opaque",
  attrSchema,
  /* Zero inputs is allowed, not encouraged. A node that reads nothing and
     produces a value is constant-like, and a converter is nearly always better
     off importing it as a declared tensor - which is a thing the graph
     understands, rather than an operation it does not. But the metadata has to
     be able to say it: refusing the arity would mean an importer meeting one
     had no representation for it at all, and the fallback from "no
     representation" is dropping the node. */
  arity: { inputs: { min: 0 }, outputs: { min: 1 } },
  // Qualified only where the qualifier carries information. Every ordinary
  // operation is in the default domain, and prefixing all of them would cost
  // card width to say nothing.
  displayName: qualifiedName,

  validateArity(_inputCount, outputCount, attrs) {
    const { shapes, dtypes } = attrsOf(attrs);
    if (shapes.length !== outputCount)
      throw new Error(`declares ${shapes.length} output shape(s) for ${outputCount} output(s)`);
    // Checked here rather than trusted, because the two lists are what a record
    // would have paired structurally.
    if (dtypes && dtypes.length !== outputCount)
      throw new Error(`declares ${dtypes.length} output dtype(s) for ${outputCount} output(s)`);
  },

  inferShapes: (_inShapes, attrs, params) =>
    attrsOf(attrs).shapes.map((shape) => resolveShape(shape, params ?? {})),

  /* Neither axis names nor symbolic extents cross a barrier: an operation that
     cannot say what it computes cannot claim an output axis is the same axis as
     an input one. Both hooks are therefore deliberately absent. */

  inferDTypes: (inDTypes, attrs, outShapes) => {
    const declared = attrsOf(attrs).dtypes;
    return outShapes.map((_shape, slot) => {
      const stated = declared?.[slot];
      if (stated) return stated;
      // Nothing to promote and nothing declared: the barrier would have to
      // invent a type for a tensor whose bytes every footprint below it counts.
      if (!inDTypes.length)
        throw new Error(
          `output ${slot} has no declared dtype and no inputs to promote from`
        );
      return promoteDTypes(inDTypes);
    });
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

  /* Unknown work, reported as none - and declared as unknown, so that "none"
     is never summed into a total as if it were a measurement. See the note in
     the header. */
  unknownWork: true,
  flopsFor: () => 0,
  flopsPerElement: () => 0,

  dependencyNote: (ctx) => {
    const attrs = attrsOf(ctx.attrs);
    const op = qualifiedName(ctx.attrs);
    // A barrier that reads nothing bounds nothing: there are no inputs to widen
    // to, so the claim is not an over-approximation and saying "read in full"
    // would describe operands it does not have. What a reader needs to know is
    // the other half - that its arithmetic is still uncounted.
    if (!ctx.inIds.length)
      return {
        text:
          ` ${op} is carried as a barrier and reads no tensor in this graph, so nothing upstream` +
          ` constrains it. Its own arithmetic is not counted.`,
        key: `opaque-source:${op}`,
        subject: ctx.outIds[0],
        severity: 1,
      };
    return {
      text:
        ` ${op} is carried as a barrier: its semantics are not modelled, so one output element` +
        ` is assumed to read every input element in full. Everything through it is a bound, and` +
        ` its own arithmetic is not counted.` +
        (attrs.opset !== undefined ? ` Source opset ${attrs.opset}.` : ""),
      key: `opaque:${op}`,
      subject: ctx.outIds[0] ?? ctx.inIds[0],
      severity: 3,
      flags: ctx.inIds.map((tensorId) => ({ tensorId, text: `read in full by ${op}` })),
    };
  },
};
