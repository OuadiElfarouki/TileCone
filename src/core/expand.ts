/** Composite-op expansion: rewrite softmax / normalize nodes into primitive subgraphs.
 * The dependency result must be identical either way (tested in composite.test.ts). */

import { Graph, Node, Tensor } from "./graph";
import { normAxes, normAxis } from "./ops/reduce";

export function isExpandable(op: string): boolean {
  return op === "softmax" || op === "normalize";
}

function clone(g: Graph): Graph {
  return JSON.parse(JSON.stringify({ nodes: g.nodes, tensors: g.tensors, params: g.params }));
}

export function expandNode(g: Graph, nodeId: string): Graph {
  const out = clone(g);
  const idx = out.nodes.findIndex((n) => n.id === nodeId);
  if (idx < 0) throw new Error(`no node "${nodeId}"`);
  const node = out.nodes[idx];
  const mk = (suffix: string, shape: (string | number)[], like: Tensor): Tensor => {
    const id = `${nodeId}$${suffix}`;
    const t: Tensor = { id, name: id, shape, dtype: like.dtype };
    out.tensors[id] = t;
    return t;
  };
  const nodes: Node[] = [];
  const op = (suffix: string, opName: string, inputs: string[], outputs: string[], attrs: Record<string, unknown>) =>
    nodes.push({ id: `${nodeId}$${suffix}`, op: opName, inputs, outputs, attrs });

  const x = out.tensors[node.inputs[0]];
  const y = node.outputs[0];
  const rank = (x.resolved ?? x.shape).length;

  if (node.op === "softmax") {
    const axis = normAxis(node.attrs.axis as number, rank);
    const m = mk("max", [], x);
    const xm = mk("sub", [], x);
    const e = mk("exp", [], x);
    const s = mk("sum", [], x);
    op("n0", "reduce", [x.id], [m.id], { fn: "max", axes: [axis], keepdim: true });
    op("n1", "elementwise", [x.id, m.id], [xm.id], { fn: "sub", nary: 2 });
    op("n2", "elementwise", [xm.id], [e.id], { fn: "exp", nary: 1 });
    op("n3", "reduce", [e.id], [s.id], { fn: "sum", axes: [axis], keepdim: true });
    op("n4", "elementwise", [e.id, s.id], [y], { fn: "div", nary: 2 });
  } else if (node.op === "normalize") {
    const kind = node.attrs.kind as string;
    const axes = normAxes(node.attrs.axes as number[], rank);
    // elementwise broadcasting is trailing-aligned, so expansion requires trailing axes
    const trailing = axes.every((a, i) => a === rank - axes.length + i);
    if (!trailing) throw new Error("expand normalize: only trailing axes supported");
    const hasWeight = node.attrs.hasWeight as boolean;
    const hasBias = node.attrs.hasBias as boolean;
    const wid = hasWeight ? node.inputs[1] : undefined;
    const bid = hasBias ? node.inputs[hasWeight ? 2 : 1] : undefined;

    let xn: Tensor;
    if (kind === "layernorm") {
      const mean = mk("mean", [], x);
      const xc = mk("xc", [], x);
      const sq = mk("sq", [], x);
      const vr = mk("var", [], x);
      const rstd = mk("rstd", [], x);
      xn = mk("xn", [], x);
      op("n0", "reduce", [x.id], [mean.id], { fn: "mean", axes, keepdim: true });
      op("n1", "elementwise", [x.id, mean.id], [xc.id], { fn: "sub", nary: 2 });
      op("n2", "elementwise", [xc.id, xc.id], [sq.id], { fn: "mul", nary: 2 });
      op("n3", "reduce", [sq.id], [vr.id], { fn: "mean", axes, keepdim: true });
      op("n4", "elementwise", [vr.id], [rstd.id], { fn: "rsqrt", nary: 1 });
      op("n5", "elementwise", [xc.id, rstd.id], [xn.id], { fn: "mul", nary: 2 });
    } else {
      const sq = mk("sq", [], x);
      const ms = mk("ms", [], x);
      const rstd = mk("rstd", [], x);
      xn = mk("xn", [], x);
      op("n0", "elementwise", [x.id, x.id], [sq.id], { fn: "mul", nary: 2 });
      op("n1", "reduce", [sq.id], [ms.id], { fn: "mean", axes, keepdim: true });
      op("n2", "elementwise", [ms.id], [rstd.id], { fn: "rsqrt", nary: 1 });
      op("n3", "elementwise", [x.id, rstd.id], [xn.id], { fn: "mul", nary: 2 });
    }
    let cur = xn.id;
    if (wid) {
      const scaled = hasBias ? mk("scaled", [], x) : undefined;
      const target = scaled ? scaled.id : y;
      op("n6", "elementwise", [cur, wid], [target], { fn: "mul", nary: 2 });
      cur = target;
    }
    if (bid) op("n7", "elementwise", [cur, bid], [y], { fn: "add", nary: 2 });
    if (!wid && !bid) op("n8", "identity", [cur], [y], {});
  } else {
    throw new Error(`op "${node.op}" is not expandable`);
  }

  out.nodes.splice(idx, 1, ...nodes);
  // strip stale resolved shapes on new tensors; resolveGraph recomputes
  for (const t of Object.values(out.tensors)) delete t.producer;
  return out;
}
