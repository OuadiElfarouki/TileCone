import { ResolvedGraph } from "./graph";
import { DTYPE_BYTES } from "./dtypes";
import { getOp, opLabel } from "./ops/index";
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
  /** The byte count with source-dtype widening and region widening applied. */
  byteFigure: Figure;
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

/**
 * What is known about a figure, in the same vocabulary regions already use.
 *
 * A number on its own cannot say which direction it is wrong in, and these
 * figures are wrong in three different directions:
 *
 * - `exact` is a count.
 * - `upper` is "no more than this". Every byte figure measured over a widened
 *   region is one: a superset contributes bytes that are not really needed, and
 *   a region is never a subset of the truth, so the figure is never understated.
 * - `approximate` is a number that moved in an unknown direction. Only ratios
 *   are this: widening a region raises the numerator *and* the denominator, so
 *   an intensity can land either side of the truth. Printing `≤` on one would
 *   claim a direction it does not have.
 * - `unknown` has no number at all. A barrier contributes zero FLOPs while the
 *   mapped operations around it contribute upper bounds over widened regions,
 *   so their sum is neither a ceiling nor a floor. `value` is `null` rather
 *   than a partial total, because a partial total is exactly the thing a reader
 *   would take for the answer.
 */
export type FigureStatus = "exact" | "upper" | "approximate" | "unknown";

export type Figure =
  | { value: number; status: "exact" | "upper" | "approximate"; reasons: string[] }
  | { value: null; status: "unknown"; reasons: string[] };

/** Worst-case ordering: a sum is only as good as its weakest contribution. */
const STATUS_ORDER: Record<FigureStatus, number> = {
  exact: 0,
  upper: 1,
  approximate: 2,
  unknown: 3,
};

const worst = (a: FigureStatus, b: FigureStatus): FigureStatus =>
  STATUS_ORDER[a] >= STATUS_ORDER[b] ? a : b;

export function figure(value: number, status: FigureStatus, reasons: string[] = []): Figure {
  const sorted = [...new Set(reasons)].sort();
  return status === "unknown"
    ? { value: null, status, reasons: sorted }
    : { value, status, reasons: sorted };
}

/** Sum two figures, keeping the weaker claim of the two. */
export function addFigures(a: Figure, b: Figure): Figure {
  const status = worst(a.status, b.status);
  const reasons = [...a.reasons, ...b.reasons];
  if (status === "unknown" || a.value === null || b.value === null)
    return figure(0, "unknown", reasons);
  return figure(a.value + b.value, status, reasons);
}

export const sumFigures = (figures: Figure[]): Figure =>
  figures.reduce(addFigures, figure(0, "exact"));

/**
 * A ratio of two figures.
 *
 * Unknown whenever either side is, because a ratio of an unknown quantity is
 * not a smaller unknown. Otherwise `approximate` as soon as either side is
 * inexact: both moved, so the quotient has no direction, which is what the `~`
 * everywhere else in this app already means.
 */
export function ratioFigure(numerator: Figure, denominator: Figure): Figure {
  const reasons = [...numerator.reasons, ...denominator.reasons];
  if (numerator.value === null || denominator.value === null)
    return figure(0, "unknown", reasons);
  if (denominator.value === 0) return figure(0, numerator.status, reasons);
  const inexact = numerator.status !== "exact" || denominator.status !== "exact";
  return figure(numerator.value / denominator.value, inexact ? "approximate" : "exact", reasons);
}

export type AggregateReadout = {
  /**
   * Arithmetic over the cone, or `unknown` where a barrier is in it.
   *
   * The one figure here that can have no number. `flopsFor` returns zero for an
   * operation nobody described - the only honest answer, since inventing work
   * would be worse - so a total spanning one is a sum of upper bounds and a
   * zero, which bounds nothing in either direction.
   */
  flops: Figure;
  inputBytes: Figure;
  intermediateBytes: Figure;
  outputBytes: Figure;
  /** Ideal op-by-op traffic: distinct reads per operation plus its writes.
   * No cross-operation cache reuse; views are modeled as materialized ops. */
  unfusedBytes: Figure;
  /**
   * FLOPs per byte of memory traffic, under the two fusion assumptions a
   * kernel author actually chooses between.
   *
   * `fused` charges the cone's graph inputs and its output: one kernel, with
   * every intermediate held in registers or shared memory and never written
   * out. `unfused` sums distinct input reads and output writes per operation,
   * with no cross-operation cache reuse. Views are assumed materialized.
   * These are idealized scenarios, not bounds on measured hardware traffic.
   *
   * Both denominators include `outputBytes`. The tile has to be written
   * somewhere in either world, and leaving it out overstated the ratio on
   * producer-output query.
   */
  fusedIntensity: Figure;
  unfusedIntensity: Figure;
  tensors: TensorReadout[];
  /**
   * Whether every figure above is exact.
   *
   * Derived from the figures rather than tracked beside them, on the same
   * principle their own statuses follow: two places that can disagree about
   * whether a number is a count eventually will. It stays because callers
   * routinely want the one-line answer - does anything here need qualifying -
   * without inspecting seven statuses to find out.
   */
  exact: boolean;
  /** Why the totals are qualified, deduplicated across every figure. */
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
    const bytes = elements * DTYPE_BYTES[t.dtype];
    const byteReasons = [
      ...tr.region.reasons,
      ...(t.dtypeWidening ? [t.dtypeWidening.note] : []),
    ];
    tensors.push({
      tensorId: tid,
      name: t.name,
      depth: tr.depth,
      elements,
      totalElements: (t.resolved ?? []).reduce((a, b) => a * b, 1),
      bytes,
      byteFigure: figure(
        bytes,
        tr.region.exact && !t.dtypeWidening ? "exact" : "upper",
        byteReasons
      ),
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

export function computeMetrics(graph: ResolvedGraph, back: PropResult): AggregateReadout {
  let flops = 0;
  let flopsExact = true;
  const flopsReasons = new Set<string>();
  // Nodes in this cone whose arithmetic nobody described. One of them is enough
  // to make the FLOP total meaningless; they are collected rather than counted
  // so the reason can name them.
  const unknownWork = new Set<string>();
  let unfusedBytes = 0;
  let trafficExact = true;
  const trafficReasons = new Set<string>();
  for (const node of graph.topo) {
    const spec = getOp(node.op)!;
    const ctx: OpCtx = {
      inShapes: graph.shapesOf(node.inputs),
      outShapes: graph.shapesOf(node.outputs),
      attrs: node.attrs,
    };
    // Keep reads separate across consumers, but deduplicate repeated operand
    // slots and overlapping output dependencies within a single operation.
    const reads = new Map<string, Region>();
    node.outputs.forEach((tid, slot) => {
      const tr = back.tensors.get(tid);
      if (!tr) return;
      unfusedBytes += count(tr.region) * DTYPE_BYTES[graph.tensors[tid].dtype];
      for (const b of tr.region.boxes) {
        spec.backward(slot, b, ctx).forEach((region, inputSlot) => {
          trafficExact &&= region.exact;
          if (!region.exact) region.reasons.forEach((reason) => trafficReasons.add(reason));
          const inputId = node.inputs[inputSlot];
          const prev = reads.get(inputId);
          reads.set(inputId, prev ? {
            boxes: [...prev.boxes, ...region.boxes],
            exact: prev.exact && region.exact,
            reasons: [...prev.reasons, ...region.reasons],
          } : region);
        });
      }
      // An operation that cannot say what it computes cannot say what it costs.
      // Its `flopsFor` returns zero, which is the only honest answer and not a
      // contribution to a total: recording the node here is what stops that
      // zero from being summed in beside real upper bounds as if it were one.
      if (spec.unknownWork) {
        unknownWork.add(opLabel(node));
        return;
      }
      if (!tr.region.exact) {
        flopsExact = false;
        tr.region.reasons.forEach((reason) => flopsReasons.add(reason));
      }
      if (spec.flopsForRegion) flops += spec.flopsForRegion(slot, tr.region, ctx);
      else if (spec.flopsPerElement)
        flops += count(tr.region) * spec.flopsPerElement(slot, ctx);
      // Per-box costs are summed, so they must be summed over a partition.
      // Tiles may overlap, and a shared element would otherwise be paid for
      // once per box that covers it.
      else for (const b of disjointify(tr.region).boxes) flops += spec.flopsFor(slot, b, ctx);
    });
    for (const [tid, region] of reads)
      unfusedBytes += count(region) * DTYPE_BYTES[graph.tensors[tid].dtype];
  }

  const tensors = coneReadout(graph, back);
  // Each byte bucket carries only the rows it actually summed. A widened weight
  // does not make the output-byte figure a bound, and saying it did would put a
  // `\u2264` on a number that is a count.
  const bucketOf = (t: TensorReadout) =>
    t.isInput ? "input" : back.roots.includes(t.tensorId) ? "output" : "intermediate";
  const bytesIn = (bucket: string): Figure => {
    const rows = tensors.filter((t) => bucketOf(t) === bucket);
    return sumFigures(rows.map((t) => t.byteFigure));
  };

  const inputBytes = bytesIn("input");
  const intermediateBytes = bytesIn("intermediate");
  const outputBytes = bytesIn("output");

  // Unknown beats every other claim: a total that skipped an operation's
  // arithmetic entirely is not an upper bound on the work, and a reader who
  // saw one number would have no way to tell.
  const flopsFigure: Figure = unknownWork.size
    ? figure(0, "unknown", [...unknownWork].map((op) => `unknown work in ${op}`))
    : figure(flops, flopsExact ? "exact" : "upper", [...flopsReasons]);

  const unfusedFigure = figure(
    unfusedBytes,
    trafficExact ? "exact" : "upper",
    [...trafficReasons]
  );

  // Traffic, not just what is read: the tile is written in both worlds.
  const fusedBytes = addFigures(inputBytes, outputBytes);
  const figures = [
    flopsFigure,
    inputBytes,
    intermediateBytes,
    outputBytes,
    unfusedFigure,
  ];
  return {
    flops: flopsFigure,
    inputBytes,
    intermediateBytes,
    outputBytes,
    unfusedBytes: unfusedFigure,
    fusedIntensity: ratioFigure(flopsFigure, fusedBytes),
    unfusedIntensity: ratioFigure(flopsFigure, unfusedFigure),
    tensors,
    exact: figures.every((f) => f.status === "exact"),
    reasons: [...new Set(figures.flatMap((f) => f.reasons))].sort(),
  };
}
