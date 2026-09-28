import type { Graph, ResolvedGraphData } from "../core/graph";
import type { PlanReport } from "../core/plan/interfaces";
import type { ReuseSurface, ReuseSweep } from "../core/reuse";
import type { Box } from "../core/region";
import type { CompilerDiagnostic } from "../parse/compiler";
import type { BaseGraphLayout } from "../view/graph/graph-scene";

export type CompileArtifact = {
  graph: Graph;
  resolved: ResolvedGraphData;
  graphPx: number;
  layout: BaseGraphLayout;
};

export type CompileJobResult =
  | { ok: true; graphId: number; artifact: CompileArtifact }
  | { ok: false; diagnostics: CompilerDiagnostic[] };

export type AnalysisRequest =
  | { id: number; kind: "compile"; source: string }
  | { id: number; kind: "register"; graphId: number; graph: ResolvedGraphData }
  | {
      id: number;
      kind: "plan";
      graphId: number | null;
      graph?: ResolvedGraphData;
      tiles: Record<string, number[]>;
    }
  | {
      id: number;
      kind: "reuse";
      graphId: number | null;
      graph?: ResolvedGraphData;
      tensorId: string;
      box: Box;
      /** The relations to report per probe; see `ReuseSurface`. */
      surfaces: ReuseSurface[];
    };

export type AnalysisResponse =
  | { id: number; kind: "compile"; result: CompileJobResult }
  | { id: number; kind: "registered"; graphId: number }
  | { id: number; kind: "plan"; result: PlanReport }
  | { id: number; kind: "reuse"; result: ReuseSweep }
  | { id: number; kind: "error"; message: string };
