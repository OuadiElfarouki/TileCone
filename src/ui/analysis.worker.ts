import { hydrateResolvedGraph, type ResolvedGraph } from "../core/graph";
import { compileArtifact, familyArtifact, reuseArtifact } from "./analysis-jobs";
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

    if (data.kind === "compile") {
      const compiled = compileArtifact(data.source);
      if (!compiled.ok) {
        host.postMessage({ id: data.id, kind: "compile", result: compiled });
        return;
      }
      host.postMessage({
        id: data.id,
        kind: "compile",
        result: { ok: true, graphId: data.id, artifact: compiled.artifact },
      });
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
    if (data.kind === "family") {
      host.postMessage({
        id: data.id,
        kind: "family",
        result: familyArtifact(graph, data.tiles, data.tensorId),
      });
    } else {
      host.postMessage({
        id: data.id,
        kind: "reuse",
        result: reuseArtifact(graph, data.tensorId, data.box, data.surfaces),
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
