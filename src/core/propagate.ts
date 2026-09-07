import { ResolvedGraph } from "./graph";
import { getOp } from "./ops/index";
import { OpCtx, OpSpec } from "./ops/types";
import type { Limits } from "./ops/limits";
import { Box, Region, canonicalize, isEmpty, sortRegion, union } from "./region";

export type Selection = { tensorId: string; region: Region };

export type TensorResult = {
  region: Region;
  /** shortest hop distance from the selected tensor (0 = the selection itself) */
  depth: number;
};

export type PropResult = {
  direction: "backward" | "forward";
  /**
   * The seed. On a merged result this is the first contributing propagation's
   * seed and nothing more - `roots` is the field that stays true after a merge,
   * and anything asking "was this tensor selected?" must use that one.
   */
  selection: Selection;
  /** Every tensor this cone was seeded from: one entry per propagation merged. */
  roots: string[];
  tensors: Map<string, TensorResult>;
  /** all inexactness reasons encountered anywhere */
  reasons: string[];
};

type PropagationStep = {
  spec: OpSpec;
  ctx: OpCtx;
  fromIds: string[];
  toIds: string[];
};

type PropagationPlan = {
  backward: PropagationStep[];
  forward: PropagationStep[];
};

/** Resolve operation dispatch and shape context once per compiled graph.
 * Interactive selection, per-box attribution, contribution probes, and reuse
 * sampling all execute the same graph repeatedly; none should rebuild this
 * immutable node context for every probe. */
const planMemo = new WeakMap<ResolvedGraph, PropagationPlan>();

function propagationPlan(graph: ResolvedGraph, limits?: Limits): PropagationPlan {
  // Only the default plan is memoized. Custom limits come from tests driving
  // the conservative branches, and caching one graph's plan under whichever
  // limits happened to be asked for first would hand the next caller a context
  // built for someone else's thresholds.
  const cached = limits ? undefined : planMemo.get(graph);
  if (cached) return cached;
  const context = new Map<string, { spec: OpSpec; ctx: OpCtx }>();
  for (const node of graph.topo) {
    context.set(node.id, {
      spec: getOp(node.op)!,
      ctx: {
        inShapes: graph.shapesOf(node.inputs),
        outShapes: graph.shapesOf(node.outputs),
        attrs: node.attrs,
        ...(limits ? { limits } : {}),
      },
    });
  }
  const steps = (direction: "backward" | "forward") => {
    const nodes = direction === "backward" ? [...graph.topo].reverse() : graph.topo;
    return nodes.map((node): PropagationStep => ({
      ...context.get(node.id)!,
      fromIds: direction === "backward" ? node.outputs : node.inputs,
      toIds: direction === "backward" ? node.inputs : node.outputs,
    }));
  };
  const plan = { backward: steps("backward"), forward: steps("forward") };
  if (!limits) planMemo.set(graph, plan);
  return plan;
}

function propagate(
  graph: ResolvedGraph,
  sel: Selection,
  dir: "backward" | "forward",
  limits?: Limits
): PropResult {
  if (!graph.tensors[sel.tensorId]) throw new Error(`unknown tensor "${sel.tensorId}"`);
  const acc = new Map<string, TensorResult>();
  const seed = canonicalize(sel.region);
  acc.set(sel.tensorId, { region: seed, depth: 0 });

  for (const { spec, ctx, fromIds, toIds } of propagationPlan(graph, limits)[dir]) {
    // Most nodes in a wide graph may be unrelated to this seed. Test reachability
    // before allocating pending regions; structural context is already cached.
    const sources = fromIds.map((id) => acc.get(id));
    if (!sources.some((src) => src && !isEmpty(src.region))) continue;

    // Union of contributions per destination tensor, canonicalized ONCE per node.
    const pending: { boxes: Box[]; exact: boolean; reasons: Set<string> }[] = toIds.map(() => ({
      boxes: [],
      exact: true,
      reasons: new Set(),
    }));
    let sourceDepth = Infinity;
    let touched = false;

    for (let slot = 0; slot < sources.length; slot++) {
      const src = sources[slot];
      if (!src || isEmpty(src.region)) continue;
      touched = true;
      sourceDepth = Math.min(sourceDepth, src.depth);
      for (const b of src.region.boxes) {
        const results: Region[] =
          dir === "backward" ? spec.backward(slot, b, ctx) : spec.forward(slot, b, ctx);
        results.forEach((r, ti) => {
          pending[ti].boxes.push(...r.boxes);
          // Inexactness flows through: a superset selection yields superset deps.
          if (!r.exact || !src.region.exact) pending[ti].exact = false;
          r.reasons.forEach((x) => pending[ti].reasons.add(x));
          src.region.reasons.forEach((x) => pending[ti].reasons.add(x));
        });
      }
    }
    if (!touched) continue;

    toIds.forEach((id, ti) => {
      const p = pending[ti];
      if (p.boxes.length === 0) return;
      const r = canonicalize({ boxes: p.boxes, exact: p.exact, reasons: [...p.reasons] });
      if (isEmpty(r)) return;
      const prev = acc.get(id);
      const next: TensorResult = prev
        ? { region: union(prev.region, r), depth: Math.min(prev.depth, sourceDepth + 1) }
        : { region: r, depth: sourceDepth + 1 };
      acc.set(id, next);
    });
  }

  const reasons = new Set<string>();
  for (const [id, tr] of acc) {
    acc.set(id, { region: sortRegion(tr.region), depth: tr.depth });
    tr.region.reasons.forEach((r) => reasons.add(r));
  }
  return { direction: dir, selection: sel, roots: [sel.tensorId], tensors: acc, reasons: [...reasons].sort() };
}

export function propagateBackward(
  graph: ResolvedGraph,
  sel: Selection,
  limits?: Limits
): PropResult {
  return propagate(graph, sel, "backward", limits);
}

export function propagateForward(
  graph: ResolvedGraph,
  sel: Selection,
  limits?: Limits
): PropResult {
  return propagate(graph, sel, "forward", limits);
}

/**
 * Combine cones seeded from different tensors into one readout.
 *
 * A cone is a per-tensor region, so combining is a per-tensor union - the same
 * operation the propagator already performs when two paths reconverge on one
 * tensor, applied one level up. Depth is the shortest hop to *any* seed, which
 * keeps the dim-by-distance rendering monotone: a tensor two hops from one
 * selection and one hop from another reads as one hop, because it is.
 *
 * Union never under-approximates, so the merged region is a valid dependency
 * claim whenever every input was, and inexactness propagates through
 * `union`'s own `exact` handling rather than being reasoned about again here.
 */
export function mergeProps(parts: PropResult[]): PropResult | null {
  if (parts.length === 0) return null;
  if (parts.length === 1) return parts[0];
  const tensors = new Map<string, TensorResult>();
  for (const part of parts)
    for (const [id, tr] of part.tensors) {
      const prev = tensors.get(id);
      tensors.set(
        id,
        prev
          ? { region: union(prev.region, tr.region), depth: Math.min(prev.depth, tr.depth) }
          : tr
      );
    }
  const reasons = new Set<string>();
  for (const [id, tr] of tensors) {
    tensors.set(id, { region: sortRegion(tr.region), depth: tr.depth });
    tr.region.reasons.forEach((r) => reasons.add(r));
  }
  const roots: string[] = [];
  for (const part of parts) for (const r of part.roots) if (!roots.includes(r)) roots.push(r);
  return {
    direction: parts[0].direction,
    selection: parts[0].selection,
    roots,
    tensors,
    reasons: [...reasons].sort(),
  };
}
