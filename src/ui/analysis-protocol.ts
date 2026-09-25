import type { Graph, ResolvedGraphData } from "../core/graph";
import type { InterfaceReport } from "../core/plan/interfaces";
import type { ReuseSweep } from "../core/reuse";
import type { Box } from "../core/region";
import type { CompilerDiagnostic } from "../parse/compiler";
import type { BaseGraphLayout } from "./graph-scene";

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
      kind: "family";
      graphId: number | null;
      graph?: ResolvedGraphData;
      tiles: Record<string, number[]>;
      tensorId: string;
    }
  | {
      id: number;
      kind: "reuse";
      graphId: number | null;
      graph?: ResolvedGraphData;
      tensorId: string;
      box: Box;
    };

export type AnalysisResponse =
  | { id: number; kind: "compile"; result: CompileJobResult }
  | { id: number; kind: "registered"; graphId: number }
  | { id: number; kind: "family"; result: InterfaceReport }
  | { id: number; kind: "reuse"; result: ReuseSweep }
  | { id: number; kind: "error"; message: string };
