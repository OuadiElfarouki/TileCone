/** Graph IR: types, validation, topological sort, symbolic shape resolution. */

import { ZodObject, ZodType, ZodIssue } from "zod";
import { getOp } from "./ops/index";
import { GraphError, resolveShape, Shape } from "./shapes";
import { DTYPES, DType } from "./dtypes";
import type { AxisNames, Cardinality } from "./ops/types";

export type { Shape, Sym } from "./shapes";
export type { DType } from "./dtypes";

export type Tensor = {
  id: string;
  name: string;
  shape: Shape;
  resolved?: number[]; // populated by resolveGraph
  /** Inputs declare this; produced tensors are canonicalized by `inferDTypes` during resolution. */
  dtype: DType;
  /** One name per axis, parallel to `shape`; holes are axes with no name. */
  axisNames?: AxisNames;
  /**
   * Symbolic extents for every axis, populated by resolution: a declared
   * tensor's own dimensions, and for a produced tensor whatever its operation
   * could carry across. An axis with no symbol holds its literal extent, so
   * this is always full length and always evaluates to `resolved`.
   */
  symShape?: Shape;
  /** Display-only: distinguishes a learned parameter from an activation. */
  role?: "activation" | "weight";
  producer?: { nodeId: string; slot: number }; // absent => graph input
};

export type Node = {
  id: string;
  op: string;
  inputs: string[]; // tensor ids
  outputs: string[]; // tensor ids
  attrs: Record<string, unknown>;
  label?: string;
};

export type Graph = {
  nodes: Node[];
  tensors: Record<string, Tensor>;
  params: Record<string, number>;
};

export type ResolvedGraph = Graph & {
  topo: Node[]; // topological order
  consumers: Record<string, { nodeId: string; slot: number }[]>;
  shapesOf: (ids: string[]) => number[][];
};

function validateCardinality(
  node: Node,
  side: "input" | "output",
  count: number,
  contract: Cardinality
): void {
  const min = typeof contract === "number" ? contract : contract.min;
  const max = typeof contract === "number" ? contract : contract.max;
  if (count >= min && (max === undefined || count <= max)) return;
  const expected =
    typeof contract === "number"
      ? `${contract}`
      : max === undefined
        ? `at least ${min}`
        : `${min} to ${max}`;
  const singular =
    typeof contract === "number"
      ? contract === 1
      : contract.max === undefined && contract.min === 1;
  throw new GraphError(
    `node "${node.id}" (${node.op}): expected ${expected} ${side}${singular ? "" : "s"}, got ${count}`,
    "GRAPH_ARITY",
    { kind: "node", id: node.id }
  );
}

/**
 * Clone the canonical, serializable portion of a graph.
 *
 * Graph resolution annotates tensors and normalizes attrs. Keeping that work on
 * a private copy makes compilation referentially transparent: callers may cache,
 * diff, or recompile their source graph without resolution leaking state into it.
 */
function cloneGraph(source: Graph): Graph {
  const cloneValue = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(cloneValue);
    if (value && typeof value === "object")
      return Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, cloneValue(item)])
      );
    return value;
  };

  const tensors: Record<string, Tensor> = {};
  for (const [id, tensor] of Object.entries(source.tensors)) {
    tensors[id] = {
      id: tensor.id,
      name: tensor.name,
      shape: tensor.shape.slice(),
      dtype: tensor.dtype,
      ...(tensor.axisNames ? { axisNames: tensor.axisNames.slice() } : {}),
      ...(tensor.symShape ? { symShape: tensor.symShape.slice() } : {}),
      ...(tensor.role ? { role: tensor.role } : {}),
    };
  }
  return {
    nodes: source.nodes.map((node) => ({
      id: node.id,
      op: node.op,
      inputs: node.inputs.slice(),
      outputs: node.outputs.slice(),
      attrs: cloneValue(node.attrs) as Record<string, unknown>,
      ...(node.label !== undefined ? { label: node.label } : {}),
    })),
    tensors,
    params: { ...source.params },
  };
}

/** Validate structure, resolve shapes, infer intermediate/output shapes, topo sort. */
/**
 * The tensors nothing consumes — what the graph is for. A produced tensor with
 * no consumer is a result; an unused *input* is a loose end, not an output, so
 * it is not one of these.
 */
export function graphOutputs(graph: ResolvedGraph): Tensor[] {
  return Object.values(graph.tensors).filter(
    (tensor) => tensor.producer && !graph.consumers[tensor.id]?.length
  );
}

/** An input's names as a full-length array, so an op's mapping can index by
 * axis without first checking whether the tensor was named at all. */
function axisNamesOf(tensor: Tensor, rank: number): AxisNames {
  const declared = tensor.axisNames ?? [];
  return Array.from({ length: rank }, (_, axis) => declared[axis]);
}

/**
 * Keep only the symbolic extents that are actually true of this tensor.
 *
 * An operation proposes; the resolver checks. Every axis is evaluated against
 * the bound parameters and kept only if it comes out at the extent inference
 * actually produced — anything else falls back to the literal. A symbol is read
 * as a claim about the graph, so a plausible-but-wrong one misinforms in a way
 * a plain number cannot, and no mapping is trusted far enough to do that.
 */
function verifiedSymShape(
  proposed: Shape | undefined,
  extents: number[],
  params: Record<string, number>
): Shape {
  return extents.map((extent, axis) => {
    const sym = proposed?.[axis];
    if (sym === undefined) return extent;
    try {
      return resolveShape([sym], params)[0] === extent ? sym : extent;
    } catch {
      return extent;
    }
  });
}

/** Narrow an object attribute schema so unknown keys are rejected instead of
 * stripped. Done here rather than in each `OpSpec` so a new operation cannot
 * forget it; see the note on `OpSpec.attrSchema` for the escape hatch. */
function strictAttrs(schema: ZodType<unknown>): ZodType<unknown> {
  return schema instanceof ZodObject ? schema.strict() : schema;
}

function editDistance(a: string, b: string): number {
  const row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let diagonal = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const above = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] === b[j - 1] ? 0 : 1));
      diagonal = above;
    }
  }
  return row[b.length];
}

/** The nearest attribute the operation actually declares, when the unknown one
 * is close enough to be a misspelling rather than a different idea. */
function nearestKey(unknown: string, known: string[]): string | null {
  let best: string | null = null;
  let bestDistance = Infinity;
  for (const candidate of known) {
    const d = editDistance(unknown.toLowerCase(), candidate.toLowerCase());
    if (d < bestDistance) {
      best = candidate;
      bestDistance = d;
    }
  }
  return best !== null && bestDistance <= 2 && bestDistance < unknown.length ? best : null;
}

/** An unrecognized attribute is the failure worth explaining well: `keepdims`
 * for `keepdim` used to resolve to a different shape without a word. */
function describeAttrIssue(issue: ZodIssue, schema: ZodType<unknown>): string {
  if (issue.code === "unrecognized_keys") {
    const known = schema instanceof ZodObject ? Object.keys(schema.shape as object) : [];
    return issue.keys
      .map((key) => {
        const suggestion = nearestKey(key, known);
        if (suggestion) return `unknown attribute "${key}" (did you mean "${suggestion}"?)`;
        return known.length
          ? `unknown attribute "${key}"; this op takes ${known.join(", ")}`
          : `unknown attribute "${key}"; this op takes no attributes`;
      })
      .join("; ");
  }
  const path = issue.path.join(".");
  return path ? `${path}: ${issue.message}` : issue.message;
}

export function resolveGraph(source: Graph): ResolvedGraph {
  const g = cloneGraph(source);
  for (const [name, value] of Object.entries(g.params))
    if (!Number.isSafeInteger(value) || value <= 0)
      throw new GraphError(
        `parameter "${name}": bad binding ${name}=${value}`,
        "GRAPH_SHAPE",
        { kind: "parameter", id: name }
      );
  const tensorIds = new Set(Object.keys(g.tensors));
  const producers = new Map<string, { nodeId: string; slot: number }>();
  const nodeIds = new Set<string>();

  for (const n of g.nodes) {
    if (nodeIds.has(n.id))
      throw new GraphError(`duplicate node id "${n.id}"`, "GRAPH_DEFINITION", {
        kind: "node",
        id: n.id,
      });
    nodeIds.add(n.id);
    const spec = getOp(n.op);
    if (!spec)
      throw new GraphError(`node "${n.id}": unknown op "${n.op}"`, "GRAPH_UNKNOWN_OP", {
        kind: "node",
        id: n.id,
      });
    for (const t of [...n.inputs, ...n.outputs])
      if (!tensorIds.has(t))
        throw new GraphError(
          `node "${n.id}" references missing tensor "${t}"`,
          "GRAPH_DEFINITION",
          { kind: "node", id: n.id }
        );
    validateCardinality(n, "input", n.inputs.length, spec.arity.inputs);
    validateCardinality(n, "output", n.outputs.length, spec.arity.outputs);
    const parsed = strictAttrs(spec.attrSchema).safeParse(n.attrs ?? {});
    if (!parsed.success)
      throw new GraphError(
        `node "${n.id}" (${n.op}): bad attrs: ${parsed.error.issues
          .map((issue) => describeAttrIssue(issue, spec.attrSchema))
          .join("; ")}`,
        "GRAPH_INVALID_ATTRIBUTES",
        { kind: "node", id: n.id }
      );
    n.attrs = parsed.data as Record<string, unknown>;
    try {
      spec.validateArity?.(n.inputs.length, n.outputs.length, n.attrs);
    } catch (e) {
      throw new GraphError(
        `node "${n.id}" (${n.op}): ${(e as Error).message}`,
        "GRAPH_ARITY",
        { kind: "node", id: n.id }
      );
    }
    for (let s = 0; s < n.outputs.length; s++) {
      const t = n.outputs[s];
      if (producers.has(t))
        throw new GraphError(`tensor "${t}" has multiple producers`, "GRAPH_DEFINITION", {
          kind: "tensor",
          id: t,
        });
      producers.set(t, { nodeId: n.id, slot: s });
    }
  }

  for (const [id, t] of Object.entries(g.tensors)) {
    if (t.id !== id)
      throw new GraphError(`tensor key "${id}" != id "${t.id}"`, "GRAPH_DEFINITION", {
        kind: "tensor",
        id,
      });
    t.producer = producers.get(id);
    if (t.producer) {
      // Produced names are derived metadata. Ignore any serialized copy and
      // recompute it from the operation, just like resolved shapes and dtypes.
      delete t.axisNames;
    } else if (t.axisNames) {
      if (t.axisNames.length !== t.shape.length)
        throw new GraphError(
          `tensor "${id}": ${t.axisNames.length} axis names for rank ${t.shape.length}`,
          "GRAPH_DEFINITION",
          { kind: "tensor", id }
        );
      if (t.axisNames.some((name) => name !== undefined && typeof name !== "string"))
        throw new GraphError(
          `tensor "${id}": axis names must be strings`,
          "GRAPH_DEFINITION",
          { kind: "tensor", id }
        );
      const named = t.axisNames.filter((name): name is string => name !== undefined);
      if (named.length && named.length !== t.axisNames.length)
        throw new GraphError(
          `tensor "${id}": name every axis or none`,
          "GRAPH_DEFINITION",
          { kind: "tensor", id }
        );
      if (new Set(named).size !== named.length)
        throw new GraphError(
          `tensor "${id}": duplicate axis name`,
          "GRAPH_DEFINITION",
          { kind: "tensor", id }
        );
      // An all-hole wire representation is equivalent to no names at all.
      if (!named.length) delete t.axisNames;
    }
    if (!DTYPES.includes(t.dtype))
      throw new GraphError(`tensor "${id}": invalid dtype "${String(t.dtype)}"`, "GRAPH_DTYPE", {
        kind: "tensor",
        id,
      });
  }

  // Topo sort (Kahn) over nodes via tensor producer edges.
  const nodeOf = new Map(g.nodes.map((n) => [n.id, n]));
  const indeg = new Map<string, number>();
  const dependents = new Map<string, string[]>();
  for (const n of g.nodes) {
    let d = 0;
    for (const t of n.inputs) {
      const p = producers.get(t);
      if (p) {
        d++;
        const arr = dependents.get(p.nodeId) ?? [];
        arr.push(n.id);
        dependents.set(p.nodeId, arr);
      }
    }
    indeg.set(n.id, d);
  }
  const queue = g.nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
  queue.sort();
  const topo: Node[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    topo.push(nodeOf.get(id)!);
    for (const dep of dependents.get(id) ?? []) {
      const d = indeg.get(dep)! - 1;
      indeg.set(dep, d);
      if (d === 0) queue.push(dep);
    }
  }
  if (topo.length !== g.nodes.length) throw new GraphError("graph has a cycle", "GRAPH_CYCLE");

  // Resolve declared shapes, then infer through the DAG.
  for (const t of Object.values(g.tensors)) {
    if (t.producer) continue;
    try {
      t.resolved = resolveShape(t.shape, g.params);
      t.symShape = verifiedSymShape(t.shape, t.resolved, g.params);
    } catch (e) {
      throw new GraphError(
        `tensor "${t.id}": shape resolution failed: ${(e as Error).message}`,
        "GRAPH_SHAPE",
        { kind: "tensor", id: t.id }
      );
    }
  }

  for (const n of topo) {
    const spec = getOp(n.op)!;
    const inShapes = n.inputs.map((t) => {
      const r = g.tensors[t].resolved;
      if (!r)
        throw new GraphError(
          `node "${n.id}": input "${t}" has unresolved shape`,
          "GRAPH_SHAPE",
          { kind: "node", id: n.id }
        );
      return r;
    });
    let outShapes: number[][];
    try {
      outShapes = spec.inferShapes(inShapes, n.attrs, g.params);
    } catch (e) {
      throw new GraphError(
        `node "${n.id}" (${n.op}): shape inference failed: ${(e as Error).message}`,
        "GRAPH_SHAPE",
        { kind: "node", id: n.id }
      );
    }
    if (outShapes.length !== n.outputs.length)
      throw new GraphError(
        `node "${n.id}": inferShapes returned ${outShapes.length} shapes for ${n.outputs.length} outputs`,
        "GRAPH_SHAPE",
        { kind: "node", id: n.id }
      );
    for (let slot = 0; slot < outShapes.length; slot++) {
      const inferred = outShapes[slot];
      if (!Array.isArray(inferred))
        throw new GraphError(
          `node "${n.id}" (${n.op}): inferred output ${slot} is not a shape`,
          "GRAPH_SHAPE",
          { kind: "node", id: n.id }
        );
      for (let axis = 0; axis < inferred.length; axis++) {
        const extent = inferred[axis];
        if (!Number.isSafeInteger(extent) || extent < 0)
          throw new GraphError(
            `node "${n.id}" (${n.op}): inferred output ${slot}, axis ${axis} has invalid extent ${String(extent)}`,
            "GRAPH_SHAPE",
            { kind: "node", id: n.id }
          );
      }
    }
    let outNames: AxisNames[] | null = null;
    if (spec.inferAxisNames) {
      // A bad mapping is an operation bug, not a bad graph, so it is reported
      // the same way a bad `inferShapes` is rather than silently dropping names.
      try {
        outNames = spec.inferAxisNames(
          n.inputs.map((id, slot) => axisNamesOf(g.tensors[id], inShapes[slot].length)),
          { inShapes, outShapes, attrs: n.attrs }
        );
      } catch (e) {
        throw new GraphError(
          `node "${n.id}" (${n.op}): axis-name inference failed: ${(e as Error).message}`,
          "GRAPH_INVALID",
          { kind: "node", id: n.id }
        );
      }
      if (outNames.length !== n.outputs.length)
        throw new GraphError(
          `node "${n.id}" (${n.op}): inferAxisNames returned ${outNames.length} name sets for ${n.outputs.length} outputs`,
          "GRAPH_INVALID",
          { kind: "node", id: n.id }
        );
      for (let slot = 0; slot < outNames.length; slot++) {
        const names = outNames[slot];
        if (!Array.isArray(names) || names.length !== outShapes[slot].length)
          throw new GraphError(
            `node "${n.id}" (${n.op}): inferAxisNames returned ` +
              `${Array.isArray(names) ? names.length : "a non-array"} names for output ${slot} ` +
              `of rank ${outShapes[slot].length}`,
            "GRAPH_INVALID",
            { kind: "node", id: n.id }
          );
        if (names.some((name) => name !== undefined && typeof name !== "string"))
          throw new GraphError(
            `node "${n.id}" (${n.op}): inferAxisNames returned an invalid name for output ${slot}`,
            "GRAPH_INVALID",
            { kind: "node", id: n.id }
          );
      }
    }

    let outSyms: Shape[] | null = null;
    if (spec.inferSymShapes) {
      try {
        outSyms = spec.inferSymShapes(
          n.inputs.map((id) => g.tensors[id].symShape ?? g.tensors[id].resolved!),
          { inShapes, outShapes, attrs: n.attrs }
        );
      } catch (e) {
        throw new GraphError(
          `node "${n.id}" (${n.op}): symbolic shape inference failed: ${(e as Error).message}`,
          "GRAPH_INVALID",
          { kind: "node", id: n.id }
        );
      }
      if (!Array.isArray(outSyms) || outSyms.length !== n.outputs.length)
        throw new GraphError(
          `node "${n.id}" (${n.op}): inferSymShapes returned ` +
            `${Array.isArray(outSyms) ? outSyms.length : "a non-array"} shapes for ${n.outputs.length} outputs`,
          "GRAPH_INVALID",
          { kind: "node", id: n.id }
        );
      for (let slot = 0; slot < outSyms.length; slot++) {
        const syms = outSyms[slot];
        if (!Array.isArray(syms) || syms.length !== outShapes[slot].length)
          throw new GraphError(
            `node "${n.id}" (${n.op}): inferSymShapes returned ` +
              `${Array.isArray(syms) ? syms.length : "a non-array"} dimensions for output ${slot} ` +
              `of rank ${outShapes[slot].length}`,
            "GRAPH_INVALID",
            { kind: "node", id: n.id }
          );
        if (syms.some((sym) => typeof sym !== "string" && typeof sym !== "number"))
          throw new GraphError(
            `node "${n.id}" (${n.op}): inferSymShapes returned an invalid dimension for output ${slot}`,
            "GRAPH_INVALID",
            { kind: "node", id: n.id }
          );
      }
    }

    let outDTypes: DType[];
    try {
      outDTypes = spec.inferDTypes(
        n.inputs.map((id) => g.tensors[id].dtype),
        n.attrs,
        outShapes
      );
    } catch (e) {
      throw new GraphError(
        `node "${n.id}" (${n.op}): dtype inference failed: ${(e as Error).message}`,
        "GRAPH_DTYPE",
        { kind: "node", id: n.id }
      );
    }
    if (outDTypes.length !== n.outputs.length)
      throw new GraphError(
        `node "${n.id}" (${n.op}): inferDTypes returned ${outDTypes.length} dtypes for ${n.outputs.length} outputs`,
        "GRAPH_DTYPE",
        { kind: "node", id: n.id }
      );
    for (let slot = 0; slot < outDTypes.length; slot++)
      if (!DTYPES.includes(outDTypes[slot]))
        throw new GraphError(
          `node "${n.id}" (${n.op}): inferred output ${slot} has invalid dtype "${String(outDTypes[slot])}"`,
          "GRAPH_DTYPE",
          { kind: "node", id: n.id }
        );
    for (let s = 0; s < n.outputs.length; s++) {
      const t = g.tensors[n.outputs[s]];
      const inferred = outShapes[s];
      if (t.shape.length) {
        // If a shape was declared, check consistency where resolvable.
        try {
          const declared = resolveShape(t.shape, g.params);
          if (declared.length !== inferred.length || declared.some((d, i) => d !== inferred[i]))
            throw new GraphError(
              `tensor "${t.id}": declared shape [${declared}] != inferred [${inferred}]`,
              "GRAPH_SHAPE",
              { kind: "tensor", id: t.id }
            );
        } catch (e) {
          if (e instanceof GraphError && e.code === "GRAPH_UNBOUND_SYMBOL") {
            /* declared with unbound syms: accept inferred */
          } else throw e;
        }
      }
      t.resolved = inferred;
      t.dtype = outDTypes[s];
      t.symShape = verifiedSymShape(outSyms?.[s], inferred, g.params);
      const names = outNames?.[s];
      if (names && names.some((v) => v !== undefined))
        t.axisNames = names.slice();
      else delete t.axisNames;
    }
  }

  const consumers: Record<string, { nodeId: string; slot: number }[]> = {};
  for (const id of tensorIds) consumers[id] = [];
  for (const n of g.nodes)
    n.inputs.forEach((t, slot) => consumers[t].push({ nodeId: n.id, slot }));

  return {
    ...g,
    topo,
    consumers,
    shapesOf: (ids: string[]) => ids.map((id) => g.tensors[id].resolved!),
  };
}
