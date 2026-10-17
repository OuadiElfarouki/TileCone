import type { Graph, ResolvedGraphData } from "../core/graph";
import type { PlanReport, TaskSupply } from "../core/plan/interfaces";
import type { TaskRef } from "../core/plan/plan";
import type { FamilyReplay } from "../core/plan/replay";
import type { CompilerDiagnostic } from "../parse/compiler";
import type { BaseGraphLayout } from "../view/graph/graph-scene";
import type { CardShapeView } from "../view/tensor/card-size";

export type CompileArtifact = {
  graph: Graph;
  resolved: ResolvedGraphData;
  graphPx: number;
  layout: BaseGraphLayout;
  /** The program's warnings, with their source spans. */
  warnings: CompilerDiagnostic[];
};

export type CompileJobResult =
  | { ok: true; graphId: number; artifact: CompileArtifact }
  | { ok: false; diagnostics: CompilerDiagnostic[] };

export type PlanRestoreRequest = {
  tiles: Record<string, number[]>;
  task: TaskRef | null;
  kept: Record<string, number[]>[];
};

export type PlanRestoreArtifact = { supply: TaskSupply | null; reports: PlanReport[] };
export type RestoreJobResult =
  | { ok: true; graphId: number; artifact: CompileArtifact; restoredPlan: PlanRestoreArtifact }
  | { ok: false; diagnostics: CompilerDiagnostic[] };

export type AnalysisRequest =
  | { id: number; kind: "compile"; source: string }
  | { id: number; kind: "restore"; source: string; plan: PlanRestoreRequest }
  | { id: number; kind: "register"; graphId: number; graph: ResolvedGraphData }
  | {
      id: number;
      kind: "layout";
      graphId: number | null;
      graph?: ResolvedGraphData;
      graphPx: number;
      views: Record<string, CardShapeView>;
    }
  | {
      id: number;
      kind: "plan";
      graphId: number | null;
      graph?: ResolvedGraphData;
      tiles: Record<string, number[]>;
    }
  | {
      id: number;
      kind: "replay";
      graphId: number | null;
      graph?: ResolvedGraphData;
      tiles: Record<string, number[]>;
      task: TaskRef;
      /** Levels of the plan's tasks to follow upstream; see `familyReplay`. */
      depth: number;
    };

export type AnalysisResponse =
  | { id: number; kind: "compile"; result: CompileJobResult }
  | { id: number; kind: "restore"; result: RestoreJobResult }
  | { id: number; kind: "registered"; graphId: number }
  | { id: number; kind: "layout"; result: BaseGraphLayout }
  | { id: number; kind: "plan"; result: PlanReport }
  | { id: number; kind: "replay"; result: FamilyReplay }
  | { id: number; kind: "error"; message: string };
