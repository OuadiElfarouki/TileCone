/**
 * Entanglement: what a tile is *combined with*, as opposed to what it depends
 * on or feeds.
 *
 * Upstream and downstream answer questions about a path through the graph.
 * Entanglement answers a question about a single operation: given a block of
 * one operand, which elements of the *other* operands meet it in the same term
 * of the computation. Neither of the other two relations contains it.
 *
 * The distinction is easiest to see on a matmul. For `C = A @ B` and a block
 * `A[0:4, 0:4]`, the downstream cone is `C[0:4, :]`, and what that cone reads
 * of `B` is all of `B` - correctly, since every `C[m,n]` in the band does read
 * every row of `B`. But the block of `A` is only ever *multiplied* against
 * `B[0:4, :]`, because `A[m,k]` meets `B[k,n]` and nothing else. That is what a
 * kernel author is asking when they ask what has to be resident alongside a
 * tile, and composing the existing relations cannot answer it: the correlation
 * between which input element produced which output element is discarded at the
 * region boundary, where a set of elements becomes a shape.
 *
 * Scope is one operation. Two tensors that meet several hops apart are related
 * through a chain of these rather than directly, and following that chain would
 * mean carrying the correlation across propagation - a materially harder
 * problem, and not the one the single-op answer leaves unsolved.
 */

import { ResolvedGraph } from "./graph";
import { getOp } from "./ops/index";
import { OpCtx } from "./ops/types";
import { Region, canonicalize, isEmpty, markInexact, sortRegion, union } from "./region";

export type Entanglement = {
  /** The operation where the two tensors meet. */
  nodeId: string;
  op: string;
  /** The tensor entangled with the selection, and its slot on that node. */
  tensorId: string;
  slot: number;
  /** The slot the selection itself occupies. A tensor may appear twice. */
  fromSlot: number;
  region: Region;
};

/**
 * Every tensor entangled with `region` on `tensorId`, one entry per (node,
 * other-slot) pair.
 *
 * A tensor read twice by one node - `matmul(A, A)` - yields an entry per slot
 * pairing, because "what does the row band meet" and "what does the column band
 * meet" are different questions with different answers, and merging them would
 * report their union as though it were either.
 */
export function entangledWith(
  graph: ResolvedGraph,
  tensorId: string,
  region: Region
): Entanglement[] {
  if (!graph.tensors[tensorId]) throw new Error(`unknown tensor "${tensorId}"`);
  const out: Entanglement[] = [];
  if (isEmpty(region)) return out;

  for (const node of graph.topo) {
    const spec = getOp(node.op)!;
    if (node.inputs.length < 2) continue;
    const ctx: OpCtx = {
      inShapes: graph.shapesOf(node.inputs),
      outShapes: graph.shapesOf(node.outputs),
      attrs: node.attrs,
    };

    node.inputs.forEach((id, fromSlot) => {
      if (id !== tensorId) return;
      node.inputs.forEach((otherId, slot) => {
        // A slot is not entangled with itself. The *same tensor* in another
        // slot is, and that is the interesting case rather than an edge one.
        if (slot === fromSlot) return;
        let result: Region | null = null;
        if (spec.coaccess) {
          for (const b of region.boxes) {
            const r = spec.coaccess(fromSlot, b, slot, ctx);
            result = result ? union(result, r) : r;
          }
          // Inexactness in the selection carries into what it is entangled with.
          if (result && !region.exact) result = markInexact(result, ...region.reasons);
        } else {
          result = fallbackCoaccess(graph, node.id, fromSlot, slot, region, ctx);
        }
        if (!result || isEmpty(result)) return;
        out.push({
          nodeId: node.id,
          op: node.op,
          tensorId: otherId,
          slot,
          fromSlot,
          region: sortRegion(canonicalize(result)),
        });
      });
    });
  }
  return out;
}

/**
 * What an operation with no `coaccess` can still be said to be entangled with:
 * forward to the outputs the block reaches, then back to the other operand.
 *
 * A superset of the truth, never a subset, because every term combining the two
 * does land in some output the block reaches. It is marked inexact for the
 * reason it is loose - the composition forgets which element produced which
 * output - so a reader is told this is a bound rather than the set.
 */
function fallbackCoaccess(
  graph: ResolvedGraph,
  nodeId: string,
  fromSlot: number,
  slot: number,
  region: Region,
  ctx: OpCtx
): Region | null {
  const node = graph.topo.find((n) => n.id === nodeId)!;
  const spec = getOp(node.op)!;
  let result: Region | null = null;
  for (const b of region.boxes) {
    const outputs = spec.forward(fromSlot, b, ctx);
    outputs.forEach((outRegion, outSlot) => {
      for (const ob of outRegion.boxes) {
        const back = spec.backward(outSlot, ob, ctx)[slot];
        if (!back) continue;
        const tainted = outRegion.exact ? back : markInexact(back, ...outRegion.reasons);
        result = result ? union(result, tainted) : tainted;
      }
    });
  }
  return result && markInexact(result, "composed from forward and backward");
}
