import {
  hydrateResolvedGraph,
  resolvedGraphData,
  type ResolvedGraph,
  type ResolvedGraphData,
} from "../core/graph";
import { interfaceOf, type InterfaceReport } from "../core/plan/interfaces";
import { tilePlan } from "../core/plan/plan";
import { tryCompileDSL, type CompilerDiagnostic } from "../parse/compiler";
import type { CompileArtifact } from "./analysis-protocol";
import { cardSize } from "./card-size";
import { buildBaseGraphLayout } from "./graph-scene";
import { shapeLabel, symbolicExtentLabel } from "./shape-label";
import { graphScale, planeExtents } from "./tiling";
import { viewAxes } from "./tensor-view";

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

export function familyArtifact(
  graph: ResolvedGraph | ResolvedGraphData,
  tiles: Record<string, number[]>,
  tensorId: string
): InterfaceReport {
  const resolved = "shapesOf" in graph ? graph : hydrateResolvedGraph(graph);
  return interfaceOf(tilePlan(resolved, tiles), tensorId);
}
