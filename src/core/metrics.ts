import { ResolvedGraph } from "./graph";
import { DTYPE_BYTES } from "./dtypes";
import { getOp } from "./ops/index";
import { OpCtx } from "./ops/types";
import { PropResult } from "./propagate";
import { Region, count } from "./region";

export type TensorReadout = {
  tensorId: string;
  name: string;
  depth: number;
  elements: number;
  totalElements: number;
  bytes: number;
  boxCount: number;
  exact: boolean;
  reasons: string[];
  isInput: boolean;
  sliceExprsNumpy: string[];
  sliceExprsTorch: string[];
};

export type AggregateReadout = {
  flops: number;
  inputBytes: number;
  intermediateBytes: number;
  outputBytes: number;
  intensity: number; // FLOPs / input bytes
  tensors: TensorReadout[];
};

export function regionSliceExprs(name: string, r: Region): { numpy: string[]; torch: string[] } {
  const mk = (b: { lo: number; hi: number }[]) =>
    `${name}[` + b.map((I) => (I.hi - I.lo === 1 ? `${I.lo}` : `${I.lo}:${I.hi}`)).join(", ") + "]";
  const lines = r.boxes.map(mk);
  const suffix = r.exact ? [] : [`# over-approximation: ${r.reasons.join(", ")}`];
  return { numpy: [...lines, ...suffix], torch: [...lines, ...suffix] };
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
      else for (const b of tr.region.boxes) flops += spec.flopsFor(slot, b, ctx);
    });
  }

  let inputBytes = 0;
  let intermediateBytes = 0;
  let outputBytes = 0;
  const tensors: TensorReadout[] = [];
  for (const [tid, tr] of back.tensors) {
    const t = graph.tensors[tid];
    const elements = count(tr.region);
    const bytes = elements * DTYPE_BYTES[t.dtype];
    const isInput = !t.producer;
    if (isInput) inputBytes += bytes;
    else if (tid === back.selection.tensorId) outputBytes += bytes;
    else intermediateBytes += bytes;
    const exprs = regionSliceExprs(t.name, tr.region);
    tensors.push({
      tensorId: tid,
      name: t.name,
      depth: tr.depth,
      elements,
      totalElements: (t.resolved ?? []).reduce((a, b) => a * b, 1),
      bytes,
      boxCount: tr.region.boxes.length,
      exact: tr.region.exact,
      reasons: tr.region.reasons,
      isInput,
      sliceExprsNumpy: exprs.numpy,
      sliceExprsTorch: exprs.torch,
    });
  }
  tensors.sort((a, b) => a.depth - b.depth || a.name.localeCompare(b.name));
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
