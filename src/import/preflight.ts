import { Graph } from "../core/graph";
import { DTYPES, DType } from "../core/dtypes";
import { dimSymbols, resolveShape, Sym } from "../core/shapes";
import { ImportDiagnostic, ImportResult } from "./types";
import { validateImportReport } from "./validate";

/**
 * Everything wrong with a converted model, before anything installs it.
 *
 * `resolveGraph` already validates a graph thoroughly, and this does not
 * duplicate it. It exists for two things resolution cannot do.
 *
 * The first is quantity. Resolution throws on the first problem it meets, which
 * is right for one line of hand-written DSL and useless for a four-hundred-node
 * model: a converter author fixes one node, re-runs, and meets the next. This
 * reports every problem at once, each naming its node.
 *
 * The second is the distinction resolution has no way to draw. A model that
 * leaves its batch axis free is not broken - a dynamic dimension is the
 * ordinary case in an exported graph - and the useful response is to ask for a
 * value, not to report an unbound symbol as a bad shape. So a free dimension
 * comes back as a request for a binding, separately from the errors, and a
 * caller can prompt for it and try again.
 *
 * The rule underneath is the barrier's own safety condition: a barrier is safe
 * only because its output metadata is concrete. An output whose rank, extents
 * or element type the file did not supply makes it unsafe, and there is no
 * conservative guess available - a shape is not something that can be widened
 * in the direction that stays honest.
 */

export type PreflightResult = {
  /**
   * Dimensions the file left free, in first-seen order.
   *
   * Not errors: a caller is expected to bind these - from the user, or from a
   * default it states - and import again. Reported here so that asking is
   * possible at all.
   */
  unbound: string[];
  /** Problems no binding can fix, each naming the node or tensor it is about. */
  errors: ImportDiagnostic[];
};

const isSupportedDType = (dtype: unknown): dtype is DType =>
  typeof dtype === "string" && (DTYPES as readonly string[]).includes(dtype);

/** An output shape a node declared for itself, where the operation carries one. */
function declaredShapes(attrs: Record<string, unknown>): Sym[][] | null {
  const shapes = attrs.shapes;
  if (!Array.isArray(shapes)) return null;
  return shapes.every(Array.isArray) ? (shapes as Sym[][]) : null;
}

export function preflightImport(result: ImportResult): PreflightResult {
  const graph: Graph = result.graph;
  const errors: ImportDiagnostic[] = validateImportReport(result);
  const unbound: string[] = [];
  const seenUnbound = new Set<string>();
  const params = graph.params ?? {};
  const produced = new Set(graph.nodes.flatMap((node) => node.outputs));

  const noteSymbols = (shape: Sym[]): void => {
    for (const dim of shape)
      for (const name of dimSymbols(dim))
        if (!(name in params) && !seenUnbound.has(name)) {
          seenUnbound.add(name);
          unbound.push(name);
        }
  };

  /**
   * Check one shape that is meant to be concrete.
   *
   * Symbols are collected before evaluating, so a shape that is only waiting on
   * a binding is reported as waiting rather than as broken. Anything still
   * failing after that is a real defect: a zero extent, arithmetic that does
   * not come out whole, a dimension that does not parse.
   */
  const checkShape = (
    shape: Sym[],
    subject: ImportDiagnostic["subject"],
    what: string
  ): void => {
    noteSymbols(shape);
    if (shape.some((dim) => dimSymbols(dim).some((name) => !(name in params)))) return;
    try {
      resolveShape(shape, params);
    } catch (e) {
      errors.push({ severity: "error", message: `${what}: ${(e as Error).message}`, subject });
    }
  };

  for (const [id, tensor] of Object.entries(graph.tensors)) {
    if (!isSupportedDType(tensor.dtype))
      errors.push({
        severity: "error",
        message:
          `tensor "${tensor.name}" has element type "${String(tensor.dtype)}",` +
          ` which this engine does not hold`,
        subject: { kind: "tensor", id },
      });
    // A produced tensor's shape is inferred from its producer, so an empty one
    // there is the placeholder it is supposed to be. A declared tensor with no
    // shape is a rank-0 scalar, which is legitimate, so neither case is checked
    // for emptiness - only for extents that cannot be resolved.
    if (!produced.has(id))
      checkShape(tensor.shape, { kind: "tensor", id }, `tensor "${tensor.name}"`);
  }

  for (const node of graph.nodes) {
    const attrs = (node.attrs ?? {}) as Record<string, unknown>;
    if (node.op !== "opaque") continue;

    // The barrier's whole safety argument is that its outputs are concrete.
    // Without the metadata there is nothing to be conservative with: a region
    // needs an extent before it can be widened to the whole of one.
    const shapes = declaredShapes(attrs);
    if (!shapes)
      errors.push({
        severity: "error",
        message:
          `node "${node.id}" is a barrier with no declared output shapes;` +
          ` a barrier is safe only because its output metadata is concrete`,
        subject: { kind: "node", id: node.id, attribute: "shapes" },
      });
    else {
      if (shapes.length !== node.outputs.length)
        errors.push({
          severity: "error",
          message:
            `node "${node.id}" declares ${shapes.length} output shape(s)` +
            ` for ${node.outputs.length} output(s)`,
          subject: { kind: "node", id: node.id, attribute: "shapes" },
        });
      shapes.forEach((shape, slot) =>
        checkShape(shape, { kind: "node", id: node.id, attribute: "shapes" },
          `node "${node.id}" output ${slot}`)
      );
    }

    const dtypes = attrs.dtypes;
    if (!Array.isArray(dtypes))
      errors.push({
        severity: "error",
        message:
          dtypes === undefined
            ? `node "${node.id}" is a barrier with no declared output dtypes;` +
              ` imported barriers must state one concrete dtype per output`
            : `node "${node.id}" declares output dtypes that are not a list`,
        subject: { kind: "node", id: node.id, attribute: "dtypes" },
      });
    else {
      if (dtypes.length !== node.outputs.length)
        errors.push({
          severity: "error",
          message:
            `node "${node.id}" declares ${dtypes.length} output dtype(s)` +
            ` for ${node.outputs.length} output(s)`,
          subject: { kind: "node", id: node.id, attribute: "dtypes" },
        });
      dtypes.forEach((dtype, slot) => {
        if (dtype === null) {
          errors.push({
            severity: "error",
            message:
              `node "${node.id}" output ${slot} has no concrete dtype;` +
              ` an unknown operation's result type cannot be inferred from its inputs`,
            subject: { kind: "node", id: node.id, attribute: "dtypes" },
          });
          return;
        }
        if (isSupportedDType(dtype)) return;
        errors.push({
          severity: "error",
          message:
            `node "${node.id}" output ${slot} has element type "${String(dtype)}",` +
            ` which this engine does not hold`,
          subject: { kind: "node", id: node.id, attribute: "dtypes" },
        });
      });
    }
  }

  return { unbound, errors };
}
