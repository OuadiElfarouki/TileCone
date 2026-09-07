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
  /**
   * Whether every figure above is exact.
   *
   * Each of them is measured over the cone's regions, so each inherits any
   * over-approximation in them: a superset region contributes bytes it does not
   * really need and FLOPs for work that is not really done, which makes these
   * upper bounds rather than counts. The per-tensor rows have always carried
   * their own `exact` flag, and every other layer refuses to let an
   * approximation pass as truth - this field is what extends that rule to the
   * totals, which were the one place a bound was printed as a number.
   *
   * False means "no more than this", never "this". It is never a *lower* bound,
   * because a region is never a subset of the truth.
   */
  exact: boolean;
  /** Why the totals are bounds, deduplicated across the contributing rows. */
  reasons: string[];
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
  // Any inexact row taints every total, because each total sums over all of
  // them. Taking the reasons from the rows rather than from `back.reasons`
  // keeps the explanation to the regions these figures were actually measured
  // on: propagation may have recorded a reason on a tensor this cone reached
  // but no metric counted.
  const inexact = tensors.filter((tensor) => !tensor.exact);
  const reasons = [...new Set(inexact.flatMap((tensor) => tensor.reasons))].sort();
  return {
    flops,
    inputBytes,
    intermediateBytes,
    outputBytes,
    intensity: denom > 0 ? flops / denom : 0,
    tensors,
    exact: inexact.length === 0,
    reasons,
  };
}
