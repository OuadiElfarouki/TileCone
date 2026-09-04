import dagre from "dagre";
import type { ResolvedGraph, Tensor } from "../core/graph";
import {
  cubicPath,
  curvedEdge,
  fanSpacing,
  fannedFlowMark,
  type Rect,
  WORLD_MARGIN,
} from "./graph-geometry";
import type { TensorOffsets } from "./tensor-layout";

export type GraphNodeKind = "tensor" | "op";

export type PlacedGraphNode = Rect & {
  id: string;
  kind: GraphNodeKind;
};

export type GraphLink = {
  from: string;
  to: string;
  tensorId: string;
  opId: string;
};

export type RoutedGraphEdge = GraphLink & {
  key: string;
  path: string;
  mark: string;
};

export type BaseGraphLayout = {
  nodes: PlacedGraphNode[];
  links: GraphLink[];
  width: number;
  height: number;
};

export type GraphScene = {
  nodes: PlacedGraphNode[];
  edges: RoutedGraphEdge[];
  /** World-space origin of the full scene bounds. User-moved cards may make
   * either value negative; node coordinates themselves stay unnormalised so a
   * live drag never shifts its peers. */
  left: number;
  top: number;
  width: number;
  height: number;
};

export type TensorMeasure = (tensor: Tensor) => Pick<Rect, "w" | "h">;

const graphId = (node: PlacedGraphNode): string =>
  `${node.kind === "tensor" ? "t" : "n"}:${node.id}`;

/**
 * Run structural graph layout exactly once for a resolved graph and a tensor
 * measurement policy. Query results, highlighting, viewport state, and user
 * offsets are deliberately absent from this contract.
 */
export function buildBaseGraphLayout(
  resolved: ResolvedGraph,
  measureTensor: TensorMeasure
): BaseGraphLayout {
  const graph = new dagre.graphlib.Graph();
  graph.setGraph({ rankdir: "LR", nodesep: 26, ranksep: 46, marginx: 20, marginy: 20 });
  graph.setDefaultEdgeLabel(() => ({}));

  for (const tensor of Object.values(resolved.tensors)) {
    const { w, h } = measureTensor(tensor);
    graph.setNode(`t:${tensor.id}`, { width: w, height: h });
  }
  for (const node of resolved.nodes) {
    const label = node.label ?? node.op;
    graph.setNode(`n:${node.id}`, {
      width: Math.max(64, label.length * 8 + 22),
      height: 30,
    });
    for (const tensorId of node.inputs) graph.setEdge(`t:${tensorId}`, `n:${node.id}`);
    for (const tensorId of node.outputs) graph.setEdge(`n:${node.id}`, `t:${tensorId}`);
  }

  dagre.layout(graph);
  const nodes: PlacedGraphNode[] = [];
  for (const id of graph.nodes()) {
    const placed = graph.node(id);
    if (!placed) continue;
    nodes.push({
      id: id.slice(2),
      kind: id.startsWith("t:") ? "tensor" : "op",
      x: placed.x - placed.width / 2,
      y: placed.y - placed.height / 2,
      w: placed.width,
      h: placed.height,
    });
  }

  // Dagre keys edges by endpoint pair, so repeated operands such as
  // `matmul(X, X)` collapse inside its graph. Rebuild the render links from the
  // ordered operand lists: layout needs connectivity, rendering needs arity.
  const links: GraphLink[] = [];
  for (const node of resolved.nodes) {
    for (const tensorId of node.inputs)
      links.push({
        from: `t:${tensorId}`,
        to: `n:${node.id}`,
        tensorId,
        opId: node.id,
      });
    for (const tensorId of node.outputs)
      links.push({
        from: `n:${node.id}`,
        to: `t:${tensorId}`,
        tensorId,
        opId: node.id,
      });
  }

  const bounds = graph.graph();
  return {
    nodes,
    links,
    width: bounds.width ?? 800,
    height: bounds.height ?? 600,
  };
}

/**
 * Apply user movement and route the live connectors without invoking Dagre.
 * This is intentionally cheap enough to run for every tensor drag frame.
 */
export function buildGraphScene(
  base: BaseGraphLayout,
  tensorOffsets: Readonly<TensorOffsets>
): GraphScene {
  const nodes = base.nodes.map((node) => {
    if (node.kind !== "tensor") return node;
    const offset = tensorOffsets[node.id];
    return offset
      ? { ...node, x: node.x + offset.dx, y: node.y + offset.dy }
      : node;
  });
  const nodesByGraphId = new Map(nodes.map((node) => [graphId(node), node]));

  // Parallel connectors fan around their shared centre line in operand order.
  const pairCounts = new Map<string, number>();
  for (const link of base.links) {
    const key = `${link.from}|${link.to}`;
    pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
  }
  const pairSlots = new Map<string, number>();
  const edges: RoutedGraphEdge[] = [];
  for (const link of base.links) {
    const pairKey = `${link.from}|${link.to}`;
    const slot = pairSlots.get(pairKey) ?? 0;
    pairSlots.set(pairKey, slot + 1);
    const from = nodesByGraphId.get(link.from);
    const to = nodesByGraphId.get(link.to);
    if (!from || !to) continue;
    const count = pairCounts.get(pairKey)!;
    const spacing = fanSpacing(from, to, count);
    const curve = curvedEdge(from, to, (slot - (count - 1) / 2) * spacing);
    edges.push({
      ...link,
      key: `${pairKey}#${slot}`,
      path: cubicPath(curve),
      mark: fannedFlowMark(curve, slot, count, spacing),
    });
  }

  const left = Math.min(0, ...nodes.map((node) => node.x - WORLD_MARGIN));
  const top = Math.min(0, ...nodes.map((node) => node.y - WORLD_MARGIN));
  const right = Math.max(
    base.width,
    ...nodes.map((node) => node.x + node.w + WORLD_MARGIN)
  );
  const bottom = Math.max(
    base.height,
    ...nodes.map((node) => node.y + node.h + WORLD_MARGIN)
  );

  return {
    nodes,
    edges,
    left,
    top,
    width: right - left,
    height: bottom - top,
  };
}
