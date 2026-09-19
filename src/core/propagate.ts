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

/** No tensor blocked: the walk every transitive cone takes. */
const OPEN: ReadonlySet<string> = new Set();

/**
 * The one traversal behind both kinds of cone.
 *
 * A `blocked` tensor may still be reached - its region is recorded like any
 * other - but it is never read as a source, so no dependency path continues
 * through it. Blocking is applied here rather than in `propagationPlan`, which
 * is memoized per graph and must stay the same plan for every caller.
 */
function walk(
  graph: ResolvedGraph,
  sel: Selection,
  dir: "backward" | "forward",
  limits: Limits | undefined,
  blocked: ReadonlySet<string>
): { tensors: Map<string, TensorResult>; reasons: string[] } {
  if (!graph.tensors[sel.tensorId]) throw new Error(`unknown tensor "${sel.tensorId}"`);
  const acc = new Map<string, TensorResult>();
  const seed = canonicalize(sel.region);
  acc.set(sel.tensorId, { region: seed, depth: 0 });

  for (const { spec, ctx, fromIds, toIds } of propagationPlan(graph, limits)[dir]) {
    // Most nodes in a wide graph may be unrelated to this seed. Test reachability
    // before allocating pending regions; structural context is already cached.
    const sources = fromIds.map((id) => (blocked.has(id) ? undefined : acc.get(id)));
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
  return { tensors: acc, reasons: [...reasons].sort() };
}

function propagate(
  graph: ResolvedGraph,
  sel: Selection,
  dir: "backward" | "forward",
  limits?: Limits
): PropResult {
  const { tensors, reasons } = walk(graph, sel, dir, limits, OPEN);
  return { direction: dir, selection: sel, roots: [sel.tensorId], tensors, reasons };
}

/**
 * A cone stopped at a frontier: what one stage demands of its boundary, rather
 * than everything the seed transitively reaches.
 *
 * It relates the seed to the elements joined to it by a dependency path whose
 * *interior* avoids the frontier. The endpoints of a path - the seed, and the
 * tensor a region is reported on - may lie on it. So a frontier tensor is
 * reached and carries its demand, but nothing is followed through it, and a
 * tensor also reachable around the frontier keeps exactly the part that route
 * supplies.
 *
 * Deliberately not a `PropResult`. Every consumer of that type - metrics,
 * notes, contribution, input sharing, the canvas - reads absence as "no
 * dependency" and treats a tensor without a producer as a graph input. A cone
 * that stops early breaks both readings: graph inputs past the frontier are
 * missing, and the frontier tensors that stand in for them have producers.
 * Handed to `computeMetrics`, it would understate input bytes while presenting
 * the result as a bound. The two records differ in shape so the compiler, not
 * a runtime check, keeps them apart.
 */
export type BoundedCone = {
  direction: "backward" | "forward";
  seed: Selection;
  /** The frontier as requested: sorted, each tensor once. */
  frontier: string[];
  /**
   * The frontier tensors this cone reached, other than its seed: where its
   * demand lands on the boundary. Sorted.
   */
  stoppedAt: string[];
  tensors: Map<string, TensorResult>;
  reasons: string[];
};

/**
 * Propagate from `seed`, stopping at every `frontier` tensor it reaches.
 *
 * The seed is never blocked, even when it is on the frontier, because it is
 * where every path starts rather than a point inside one. That is what lets a
 * single boundary serve both directions: a stage's inputs and outputs together
 * bound a backward walk from one of its outputs and a forward walk from one of
 * its inputs, with no need to say which side the seed is on.
 */
export function propagateWithin(
  graph: ResolvedGraph,
  seed: Selection,
  direction: "backward" | "forward",
  frontier: readonly string[],
  limits?: Limits
): BoundedCone {
  const stop = [...new Set(frontier)].sort();
  for (const id of stop) if (!graph.tensors[id]) throw new Error(`unknown frontier tensor "${id}"`);
  const blocked = new Set(stop.filter((id) => id !== seed.tensorId));
  const { tensors, reasons } = walk(graph, seed, direction, limits, blocked);
  return {
    direction,
    seed,
    frontier: stop,
    stoppedAt: stop.filter((id) => blocked.has(id) && tensors.has(id)),
    tensors,
    reasons,
  };
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
