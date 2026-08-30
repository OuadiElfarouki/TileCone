/** Graph IR: types, validation, topological sort, symbolic shape resolution. */

import { getOp } from "./ops/index";
import { GraphError, resolveDim, resolveShape, Shape, Sym } from "./shapes";

export { GraphError, resolveDim, resolveShape };
export type { Shape, Sym };

export type DType = "f32" | "f16" | "bf16" | "f8" | "i32" | "i8" | "bool";

export const DTYPE_BYTES: Record<DType, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  f8: 1,
  i32: 4,
  i8: 1,
  bool: 1,
};

export type Tensor = {
  id: string;
  name: string;
  shape: Shape;
  resolved?: number[]; // populated by resolveGraph
  dtype: DType;
  axisNames?: string[];
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

/** Validate structure, resolve shapes, infer intermediate/output shapes, topo sort. */
export function resolveGraph(g: Graph): ResolvedGraph {
  const tensorIds = new Set(Object.keys(g.tensors));
  const producers = new Map<string, { nodeId: string; slot: number }>();
  const nodeIds = new Set<string>();

  for (const n of g.nodes) {
    if (nodeIds.has(n.id)) throw new GraphError(`duplicate node id "${n.id}"`);
    nodeIds.add(n.id);
    const spec = getOp(n.op);
    if (!spec) throw new GraphError(`node "${n.id}": unknown op "${n.op}"`);
    for (const t of [...n.inputs, ...n.outputs])
      if (!tensorIds.has(t)) throw new GraphError(`node "${n.id}" references missing tensor "${t}"`);
    if (spec.arity.inputs !== "variadic" && n.inputs.length !== spec.arity.inputs)
      throw new GraphError(`node "${n.id}" (${n.op}): expected ${spec.arity.inputs} inputs, got ${n.inputs.length}`);
    if (spec.arity.outputs !== "variadic" && n.outputs.length !== spec.arity.outputs)
      throw new GraphError(`node "${n.id}" (${n.op}): expected ${spec.arity.outputs} outputs, got ${n.outputs.length}`);
    const parsed = spec.attrSchema.safeParse(n.attrs ?? {});
    if (!parsed.success)
      throw new GraphError(`node "${n.id}" (${n.op}): bad attrs: ${parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ")}`);
    n.attrs = parsed.data as Record<string, unknown>;
    for (let s = 0; s < n.outputs.length; s++) {
      const t = n.outputs[s];
      if (producers.has(t)) throw new GraphError(`tensor "${t}" has multiple producers`);
      producers.set(t, { nodeId: n.id, slot: s });
    }
  }

  for (const [id, t] of Object.entries(g.tensors)) {
    if (t.id !== id) throw new GraphError(`tensor key "${id}" != id "${t.id}"`);
    t.producer = producers.get(id);
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
  if (topo.length !== g.nodes.length) throw new GraphError("graph has a cycle");

  // Resolve declared shapes, then infer through the DAG.
  for (const t of Object.values(g.tensors))
    if (!t.producer) t.resolved = resolveShape(t.shape, g.params);

  for (const n of topo) {
    const spec = getOp(n.op)!;
    const inShapes = n.inputs.map((t) => {
      const r = g.tensors[t].resolved;
      if (!r) throw new GraphError(`node "${n.id}": input "${t}" has unresolved shape`);
      return r;
    });
    let outShapes: number[][];
    try {
      outShapes = spec.inferShapes(inShapes, n.attrs, g.params);
    } catch (e) {
      throw new GraphError(`node "${n.id}" (${n.op}): shape inference failed: ${(e as Error).message}`);
    }
    if (outShapes.length !== n.outputs.length)
      throw new GraphError(`node "${n.id}": inferShapes returned ${outShapes.length} shapes for ${n.outputs.length} outputs`);
    for (let s = 0; s < n.outputs.length; s++) {
      const t = g.tensors[n.outputs[s]];
      const inferred = outShapes[s];
      if (t.shape.length) {
        // If a shape was declared, check consistency where resolvable.
        try {
          const declared = resolveShape(t.shape, g.params);
          if (declared.length !== inferred.length || declared.some((d, i) => d !== inferred[i]))
            throw new GraphError(
              `tensor "${t.id}": declared shape [${declared}] != inferred [${inferred}]`
            );
        } catch (e) {
          if (e instanceof GraphError && /unbound/.test(e.message)) {
            /* declared with unbound syms: accept inferred */
          } else throw e;
        }
      }
      t.resolved = inferred;
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
