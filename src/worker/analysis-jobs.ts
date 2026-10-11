import {
  hydrateResolvedGraph,
  resolvedGraphData,
  type ResolvedGraph,
  type ResolvedGraphData,
} from "../core/graph";
import { planReport, taskSupplyOf, type PlanReport } from "../core/plan/interfaces";
import { familyOf, tilePlan } from "../core/plan/plan";
import {
  estimateInputReuseSweep,
  type ReuseSurface,
  type ReuseSweep,
} from "../core/reuse";
import { fromBox, type Box } from "../core/region";
import { tryCompileDSL, type CompilerDiagnostic } from "../parse/compiler";
import type { CompileArtifact, PlanRestoreArtifact, PlanRestoreRequest } from "./analysis-protocol";
import { cardSize } from "../view/tensor/card-size";
import { buildBaseGraphLayout } from "../view/graph/graph-scene";
import { shapeLabel, symbolicExtentLabel } from "../view/tensor/shape-label";
import { graphScale, planeExtents } from "../view/tensor/tiling";
import { viewAxes } from "../view/tensor/tensor-view";

const graphPlanes = (resolved: ResolvedGraph): { rows: number; cols: number }[] =>
  Object.values(resolved.tensors).map((tensor) => {
    const shape = tensor.resolved!;
    const { rowAxis, colAxis } = viewAxes(shape);
    return planeExtents(shape, rowAxis, colAxis);
  });

/** Structural placement for a graph's chosen tensor planes. This job uses no
 * selection or viewport state and can run independently of query analysis. */
export function layoutArtifact(
  graph: ResolvedGraph | ResolvedGraphData,
  graphPx: number,
  axes: Record<string, [number, number]> = {},
) {
  const resolved = "shapesOf" in graph ? graph : hydrateResolvedGraph(graph);
  return buildBaseGraphLayout(resolved, (tensor) =>
    cardSize(tensor.resolved!, graphPx, tensor.name, [
      shapeLabel(tensor, "symbolic"), symbolicExtentLabel(tensor),
    ], axes[tensor.id] ? { axes: axes[tensor.id] } : undefined, tensor.axisNames)
  );
}

/** Compile, infer, and structurally lay out a graph. Safe to call in a Worker
 * because every returned field is structured-cloneable. */
export function compileArtifact(source: string):
  | { ok: true; artifact: CompileArtifact; resolved: ResolvedGraph }
  | { ok: false; diagnostics: CompilerDiagnostic[] } {
  const result = tryCompileDSL(source);
  if (!result.ok) return result;
  const { graph, resolved } = result.program;
  const graphPx = graphScale(graphPlanes(resolved));
  const layout = layoutArtifact(resolved, graphPx);
  return {
    ok: true,
    resolved,
    artifact: { graph, resolved: resolvedGraphData(resolved), graphPx, layout },
  };
}

export function planArtifact(
  graph: ResolvedGraph | ResolvedGraphData,
  tiles: Record<string, number[]>
): PlanReport {
  const resolved = "shapesOf" in graph ? graph : hydrateResolvedGraph(graph);
  return planReport(tilePlan(resolved, tiles));
}

/** Evaluate the complete restore before publishing its graph. The task uses
 * coordinate sets rather than an unbounded array of producer witnesses. */
export function planRestoreArtifact(graph: ResolvedGraph, request: PlanRestoreRequest): PlanRestoreArtifact {
  const plan = tilePlan(graph, request.tiles);
  if (request.task) familyOf(plan, request.task);
  return {
    supply: request.task ? taskSupplyOf(plan, request.task) : null,
    reports: request.kept.map((tiles) => planReport(tilePlan(graph, tiles))),
  };
}

export function reuseArtifact(
  graph: ResolvedGraph | ResolvedGraphData,
  tensorId: string,
  box: Box,
  surfaces?: ReuseSurface[]
): ReuseSweep {
  const resolved = "shapesOf" in graph ? graph : hydrateResolvedGraph(graph);
  return estimateInputReuseSweep(resolved, { tensorId, region: fromBox(box) }, { surfaces });
}
