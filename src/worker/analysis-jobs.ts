import {
  hydrateResolvedGraph,
  resolvedGraphData,
  type ResolvedGraph,
  type ResolvedGraphData,
} from "../core/graph";
import { planReport, type PlanReport } from "../core/plan/interfaces";
import { tilePlan } from "../core/plan/plan";
import {
  estimateInputReuseSweep,
  type ReuseSurface,
  type ReuseSweep,
} from "../core/reuse";
import { fromBox, type Box } from "../core/region";
import { tryCompileDSL, type CompilerDiagnostic } from "../parse/compiler";
import type { CompileArtifact } from "./analysis-protocol";
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

/** Compile, infer, and structurally lay out a graph. Safe to call in a Worker
 * because every returned field is structured-cloneable. */
export function compileArtifact(source: string):
  | { ok: true; artifact: CompileArtifact; resolved: ResolvedGraph }
  | { ok: false; diagnostics: CompilerDiagnostic[] } {
  const result = tryCompileDSL(source);
  if (!result.ok) return result;
  const { graph, resolved } = result.program;
  const graphPx = graphScale(graphPlanes(resolved));
  const layout = buildBaseGraphLayout(resolved, (tensor) =>
    cardSize(tensor.resolved!, graphPx, tensor.name, [
      shapeLabel(tensor, "symbolic"),
      symbolicExtentLabel(tensor),
    ])
  );
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

export function reuseArtifact(
  graph: ResolvedGraph | ResolvedGraphData,
  tensorId: string,
  box: Box,
  surfaces?: ReuseSurface[]
): ReuseSweep {
  const resolved = "shapesOf" in graph ? graph : hydrateResolvedGraph(graph);
  return estimateInputReuseSweep(resolved, { tensorId, region: fromBox(box) }, { surfaces });
}
