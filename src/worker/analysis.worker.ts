import { hydrateResolvedGraph, type ResolvedGraph } from "../core/graph";
import { compileArtifact, layoutArtifact, planArtifact, planRestoreArtifact, replayArtifact } from "./analysis-jobs";
import type { AnalysisRequest, AnalysisResponse } from "./analysis-protocol";

type WorkerHost = {
  onmessage: ((event: MessageEvent<AnalysisRequest>) => void) | null;
  postMessage: (message: AnalysisResponse) => void;
};

const host = globalThis as unknown as WorkerHost;
const graphs = new Map<number, ResolvedGraph>();

host.onmessage = ({ data }) => {
  try {
    if (data.kind === "register") {
      graphs.clear();
      graphs.set(data.graphId, hydrateResolvedGraph(data.graph));
      host.postMessage({ id: data.id, kind: "registered", graphId: data.graphId });
      return;
    }

    if (data.kind === "compile" || data.kind === "restore") {
      const compiled = compileArtifact(data.source);
      if (!compiled.ok) {
        host.postMessage({ id: data.id, kind: data.kind, result: compiled });
        return;
      }
      if (data.kind === "restore")
        host.postMessage({ id: data.id, kind: "restore", result: {
          ok: true, graphId: data.id, artifact: compiled.artifact,
          restoredPlan: planRestoreArtifact(compiled.resolved, data.plan),
        } });
      else host.postMessage({ id: data.id, kind: "compile", result: { ok: true, graphId: data.id, artifact: compiled.artifact } });
      return;
    }

    let graph = data.graphId === null ? undefined : graphs.get(data.graphId);
    if (!graph && data.graph) {
      graph = hydrateResolvedGraph(data.graph);
      if (data.graphId !== null) {
        graphs.clear();
        graphs.set(data.graphId, graph);
      }
    }
    if (!graph) throw new Error("the graph is no longer available to the analysis worker");
    if (data.kind === "layout") {
      host.postMessage({ id: data.id, kind: "layout", result: layoutArtifact(graph, data.graphPx, data.axes) });
    } else if (data.kind === "plan") {
      host.postMessage({
        id: data.id,
        kind: "plan",
        result: planArtifact(graph, data.tiles),
      });
    } else {
      host.postMessage({
        id: data.id,
        kind: "replay",
        result: replayArtifact(graph, data.tiles, data.task),
      });
    }
  } catch (error) {
    host.postMessage({
      id: data.id,
      kind: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
