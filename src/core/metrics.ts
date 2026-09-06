import { ResolvedGraph } from "./graph";
import { DTYPE_BYTES } from "./dtypes";
import { getOp } from "./ops/index";
import { OpCtx } from "./ops/types";
import { PropResult } from "./propagate";
import { Region, count, disjointify, formatBoxIndices, regionOverlap } from "./region";

export type TensorReadout = {
  tensorId: string;
  name: string;
  depth: number;
  elements: number;
  totalElements: number;
  bytes: number;
  boxCount: number;
  /** Elements this cone reads more than once, because two boxes share them.
   * `elements` already counts them once; this is the difference between that
   * and the sum of the listed boxes, so the slice expressions add up. */
  overlap: number;
  exact: boolean;
  reasons: string[];
  isInput: boolean;
  /** Copyable `name[i, lo:hi]` lines, plus a trailing `#` comment when the
   * region is an over-approximation. The syntax is index notation both NumPy
   * and torch accept, so there is one list rather than one per framework. */
  sliceExprs: string[];
};

export type AggregateReadout = {
  flops: number;
  inputBytes: number;
  intermediateBytes: number;
  outputBytes: number;
  intensity: number; // FLOPs / input bytes
  tensors: TensorReadout[];
};

/** One line per box, in the boxes' own terms. They may overlap: two operand
 * slots reading one tensor produce two bands that share a corner, and naming
 * the bands is the point. The element total is measured on the union, so it is
 * smaller than these lines summed whenever they do overlap. */
function regionSliceExprs(name: string, r: Region): string[] {
  const lines = r.boxes.map((b) => `${name}[${formatBoxIndices(b)}]`);
  const suffix = r.exact ? [] : [`# over-approximation: ${r.reasons.join(", ")}`];
  return [...lines, ...suffix];
}

/**
 * What one cone touches, per tensor, ordered by distance from the seed.
 *
 * Direction-neutral on purpose: a backward cone's rows say what a box reads and
 * a forward cone's rows say what it feeds, but they are the same measurement of
 * the same region algebra and the panel shows them side by side. `depth` is
 * therefore steps *along the cone*, not steps upstream.
 */
export function coneReadout(graph: ResolvedGraph, prop: PropResult): TensorReadout[] {
  const tensors: TensorReadout[] = [];
  for (const [tid, tr] of prop.tensors) {
    const t = graph.tensors[tid];
    const elements = count(tr.region);
    const exprs = regionSliceExprs(t.name, tr.region);
    tensors.push({
      tensorId: tid,
      name: t.name,
      depth: tr.depth,
      elements,
      totalElements: (t.resolved ?? []).reduce((a, b) => a * b, 1),
      bytes: elements * DTYPE_BYTES[t.dtype],
      boxCount: tr.region.boxes.length,
      overlap: regionOverlap(tr.region).summed - elements,
      exact: tr.region.exact,
      reasons: tr.region.reasons,
      isInput: !t.producer,
      sliceExprs: exprs,
    });
  }
  tensors.sort((a, b) => a.depth - b.depth || a.name.localeCompare(b.name));
  return tensors;
}

export function computeMetrics(
  graph: ResolvedGraph,
  back: PropResult,
  countIntermediates = false
): AggregateReadout {
  let flops = 0;
  for (const node of graph.topo) {
    const spec = getOp(node.op)!;
    const ctx: OpCtx = {
      inShapes: graph.shapesOf(node.inputs),
      outShapes: graph.shapesOf(node.outputs),
      attrs: node.attrs,
    };
    node.outputs.forEach((tid, slot) => {
      const tr = back.tensors.get(tid);
      if (!tr) return;
      if (spec.flopsForRegion) flops += spec.flopsForRegion(slot, tr.region, ctx);
      else if (spec.flopsPerElement)
        flops += count(tr.region) * spec.flopsPerElement(slot, ctx);
      // Per-box costs are summed, so they must be summed over a partition.
      // Tiles may overlap, and a shared element would otherwise be paid for
      // once per box that covers it.
      else for (const b of disjointify(tr.region).boxes) flops += spec.flopsFor(slot, b, ctx);
    });
  }

  const tensors = coneReadout(graph, back);
  let inputBytes = 0;
  let intermediateBytes = 0;
  let outputBytes = 0;
  for (const t of tensors) {
    if (t.isInput) inputBytes += t.bytes;
    else if (back.roots.includes(t.tensorId)) outputBytes += t.bytes;
    else intermediateBytes += t.bytes;
  }
  const denom = inputBytes + (countIntermediates ? intermediateBytes : 0);
  return {
    flops,
    inputBytes,
    intermediateBytes,
    outputBytes,
    intensity: denom > 0 ? flops / denom : 0,
    tensors,
  };
}
