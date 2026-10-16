/**
 * Read multiplicity: how many output elements of each operation in a cone read
 * each element of its operands.
 *
 * A footprint says which elements a tile needs; it does not say how often each
 * is used. For `C = A @ B` and an `m x n` tile of `C`, the footprint of `A` is
 * `m x K` and every one of those elements is read by `n` outputs - the reuse a
 * kernel can exploit, which no union of regions can show because a union keeps
 * whether an element is reached and drops how many times.
 *
 * Counted per operation, over the outputs the cone needs of it. That is local
 * and exact wherever the cone's regions are: an operation's output region in a
 * backward cone is what the tile needs of it, so its operands' counts follow
 * from the operation alone. Composing counts across several operations is a
 * different question - multiplying them along paths counts a residual branch
 * twice - and is not answered here.
 *
 * Every figure keeps the one-sided contract regions have. An inexact count is
 * an upper bound at every element, so the reads, the elements read and the
 * elements read more than once are upper bounds; the elements read exactly once
 * and the mean move in no fixed direction and are approximate.
 */

import { DTYPE_BYTES } from "./dtypes";
import { ResolvedGraph, Tensor } from "./graph";
import { Figure, figure, ratioFigure } from "./metrics";
import { getOp, opLabel } from "./ops/index";
import { boxElements, OpCtx, OpSpec } from "./ops/types";
import { Region, disjointify } from "./region";
import { Tally, addTallies, markTallyInexact, measureTally, uniformTally } from "./tally";

/**
 * The count for one operand over an output region of one operation.
 *
 * The region is split into disjoint boxes first: an output element counted in
 * two boxes would read its operands twice. A split past its budget is a
 * coarser superset of outputs, and more outputs only raise a count, so the
 * result stays an upper bound.
 */
export function operandTally(
  spec: OpSpec,
  outSlot: number,
  outRegion: Region,
  inSlot: number,
  ctx: OpCtx
): Tally {
  const parts = disjointify(outRegion);
  const tally = addTallies(
    parts.boxes.map((b) => {
      if (spec.multiplicity) return spec.multiplicity(outSlot, b, inSlot, ctx);
      const read = spec.backward(outSlot, b, ctx)[inSlot];
      if (spec.readsOnce) return uniformTally(read, 1);
      // No element is read by more outputs than the box holds.
      const outputs = boxElements(b);
      const bound = uniformTally(read, outputs);
      return outputs === 1 ? bound : markTallyInexact(bound, `${opLabel({ op: spec.name })} reads not counted`);
    })
  );
  return parts.exact ? tally : markTallyInexact(tally, ...parts.reasons);
}

/** One operation reading one tensor, through every operand slot that names it. */
export type OperandReads = {
  nodeId: string;
  /** The operation as the reader knows it. */
  op: string;
  /** The outputs of the operation the cone needs. Two operations of one kind
   *  share a name, and these are what tell them apart. */
  outputIds: string[];
  tensorId: string;
  /** Operand slots holding this tensor. A tensor read through two counts both. */
  slots: number[];
  /** Pairs of an output element and an element it reads. */
  reads: Figure;
  /** Elements read at least once. */
  elements: Figure;
  /** Elements read by exactly one output. */
  once: Figure;
  /** Elements read by more than one output. */
  reused: Figure;
  /** The fewest and most outputs reading one element. */
  min: Figure;
  max: Figure;
  /** `reads / elements`: outputs per element read, on average. */
  mean: Figure;
  /** `once` and `reused` as bytes of this tensor. */
  onceBytes: Figure;
  reusedBytes: Figure;
  exact: boolean;
  reasons: string[];
};

/** Elements as bytes, keeping the figure's status; a widened dtype makes a count a bound. */
function asBytes(t: Tensor, f: Figure): Figure {
  if (f.value === null) return f;
  const widened = t.dtypeWidening;
  const status = widened && f.status === "exact" ? "upper" : f.status;
  const bytes = f.value * DTYPE_BYTES[t.dtype];
  if (!Number.isSafeInteger(bytes))
    return figure(0, "unknown", [...f.reasons, "read multiplicity exceeds safe integer range"]);
  return figure(bytes, status, [
    ...f.reasons,
    ...(widened ? [widened.note] : []),
  ]);
}

/** The figures of one tally, under the statuses its exactness earns. */
function readsOf(t: Tally): Pick<OperandReads, "reads" | "elements" | "once" | "reused" | "min" | "max" | "mean"> {
  const m = measureTally(t);
  const reasons = [...t.reasons];
  const bound = t.exact ? "exact" : "upper";
  const loose = t.exact ? "exact" : "approximate";
  const measured = (value: number | null, status: "exact" | "upper" | "approximate"): Figure =>
    value === null
      ? figure(0, "unknown", [...reasons, "read multiplicity exceeds safe integer range"])
      : figure(value, status, reasons);
  const reads = measured(m.total, bound);
  const elements = measured(m.elements, bound);
  return {
    reads,
    elements,
    once: measured(m.once, loose),
    reused: measured(m.repeated, bound),
    min: measured(m.min, loose),
    max: measured(m.max, bound),
    mean: ratioFigure(reads, elements),
  };
}

/**
 * One operation's tallies over the outputs `regions` holds of it: per tensor
 * it reads, the slots holding it and how many of those outputs read each of
 * its elements. Null when the operation computes nothing in `regions`.
 */
function operationTallies(
  graph: ResolvedGraph,
  node: ResolvedGraph["topo"][number],
  regions: ReadonlyMap<string, { region: Region }>
): { outputIds: string[]; byTensor: Map<string, { slots: number[]; tally: Tally }> } | null {
  const needed = node.outputs.flatMap((id, slot) => {
    const region = regions.get(id)?.region;
    return region ? [{ slot, region }] : [];
  });
  if (!needed.length) return null;
  const spec = getOp(node.op)!;
  const ctx: OpCtx = {
    inShapes: graph.shapesOf(node.inputs),
    outShapes: graph.shapesOf(node.outputs),
    attrs: node.attrs,
  };
  const collected = new Map<string, { slots: number[]; tallies: Tally[] }>();
  node.inputs.forEach((tensorId, inSlot) => {
    const entry = collected.get(tensorId) ?? { slots: [], tallies: [] };
    entry.slots.push(inSlot);
    for (const { slot, region } of needed) entry.tallies.push(operandTally(spec, slot, region, inSlot, ctx));
    collected.set(tensorId, entry);
  });
  const byTensor = new Map<string, { slots: number[]; tally: Tally }>();
  for (const [tensorId, { slots, tallies }] of collected) byTensor.set(tensorId, { slots, tally: addTallies(tallies) });
  return { outputIds: needed.map(({ slot }) => node.outputs[slot]), byTensor };
}

/**
 * Every operation computing a tensor in `regions`, with how many of the outputs
 * it computes there read each element of each tensor it reads. In graph order,
 * then operand order; an operation reading one tensor twice gives one row for it.
 *
 * `regions` holds what is computed of each tensor, the input `flopsOver` takes:
 * a backward cone's tensors, or the tile and untiled tensors of a plan task.
 * A tensor without a producer is read, never computed, and names no operation.
 */
export function operandReads(
  graph: ResolvedGraph,
  regions: ReadonlyMap<string, { region: Region }>
): OperandReads[] {
  const rows: OperandReads[] = [];
  for (const node of graph.topo) {
    const tallies = operationTallies(graph, node, regions);
    if (!tallies) continue;
    for (const [tensorId, { slots, tally }] of tallies.byTensor) {
      if (!tally.pieces.length) continue;
      const tensor = graph.tensors[tensorId];
      const figures = readsOf(tally);
      rows.push({
        nodeId: node.id,
        op: opLabel(node),
        outputIds: tallies.outputIds,
        tensorId,
        slots,
        ...figures,
        onceBytes: asBytes(tensor, figures.once),
        reusedBytes: asBytes(tensor, figures.reused),
        exact: tally.exact,
        reasons: [...tally.reasons],
      });
    }
  }
  return rows;
}

/**
 * The tally behind one `operandReads` row: how many of the outputs `nodeId`
 * computes in `regions` read each element of `tensorId`. The same count the
 * row's figures measure, per element, for drawing it. Null when the operation
 * computes nothing there or does not read the tensor.
 */
export function operandTallyOf(
  graph: ResolvedGraph,
  regions: ReadonlyMap<string, { region: Region }>,
  nodeId: string,
  tensorId: string
): Tally | null {
  const node = graph.topo.find((candidate) => candidate.id === nodeId);
  const tally = node ? operationTallies(graph, node, regions)?.byTensor.get(tensorId)?.tally : undefined;
  return tally && tally.pieces.length ? tally : null;
}
