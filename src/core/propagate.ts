import { ResolvedGraph } from "./graph";
import { getOp } from "./ops/index";
import { OpCtx } from "./ops/types";
import { Box, Region, canonicalize, isEmpty, sortRegion, union } from "./region";

export type Selection = { tensorId: string; region: Region };

export type TensorResult = {
  region: Region;
  /** shortest hop distance from the selected tensor (0 = the selection itself) */
  depth: number;
};

export type PropResult = {
  direction: "backward" | "forward";
  selection: Selection;
  tensors: Map<string, TensorResult>;
  /** all inexactness reasons encountered anywhere */
  reasons: string[];
};

function propagate(graph: ResolvedGraph, sel: Selection, dir: "backward" | "forward"): PropResult {
  if (!graph.tensors[sel.tensorId]) throw new Error(`unknown tensor "${sel.tensorId}"`);
  const acc = new Map<string, TensorResult>();
  const seed = canonicalize(sel.region);
  acc.set(sel.tensorId, { region: seed, depth: 0 });

  const nodes = dir === "backward" ? [...graph.topo].reverse() : graph.topo;

  for (const node of nodes) {
    const spec = getOp(node.op)!;
    const ctx: OpCtx = {
      inShapes: graph.shapesOf(node.inputs),
      outShapes: graph.shapesOf(node.outputs),
      attrs: node.attrs,
    };
    const fromIds = dir === "backward" ? node.outputs : node.inputs;
    const toIds = dir === "backward" ? node.inputs : node.outputs;
    const toShapes = dir === "backward" ? ctx.inShapes : ctx.outShapes;

    // Union of contributions per destination tensor, canonicalized ONCE per node.
    const pending: { boxes: Box[]; exact: boolean; reasons: Set<string> }[] = toIds.map(() => ({
      boxes: [],
      exact: true,
      reasons: new Set(),
    }));
    let sourceDepth = Infinity;
    let touched = false;

    for (let slot = 0; slot < fromIds.length; slot++) {
      const src = acc.get(fromIds[slot]);
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
      void toShapes;
    });
  }

  const reasons = new Set<string>();
  for (const [id, tr] of acc) {
    acc.set(id, { region: sortRegion(tr.region), depth: tr.depth });
    tr.region.reasons.forEach((r) => reasons.add(r));
  }
  return { direction: dir, selection: sel, tensors: acc, reasons: [...reasons].sort() };
}

export function propagateBackward(graph: ResolvedGraph, sel: Selection): PropResult {
  return propagate(graph, sel, "backward");
}

export function propagateForward(graph: ResolvedGraph, sel: Selection): PropResult {
  return propagate(graph, sel, "forward");
}
