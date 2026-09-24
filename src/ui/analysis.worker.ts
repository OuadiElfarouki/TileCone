import { hydrateResolvedGraph, type ResolvedGraph } from "../core/graph";
import { compileArtifact, familyArtifact } from "./analysis-jobs";
import type { AnalysisRequest, AnalysisResponse } from "./analysis-protocol";

type WorkerHost = {
  onmessage: ((event: MessageEvent<AnalysisRequest>) => void) | null;
  postMessage: (message: AnalysisResponse) => void;
};

const host = globalThis as unknown as WorkerHost;
const graphs = new Map<number, ResolvedGraph>();

host.onmessage = ({ data }) => {
  try {
    if (data.kind === "compile") {
      const compiled = compileArtifact(data.source);
      if (!compiled.ok) {
        host.postMessage({ id: data.id, kind: "compile", result: compiled });
        return;
      }
      graphs.clear();
      graphs.set(data.id, compiled.resolved);
      host.postMessage({
        id: data.id,
        kind: "compile",
        result: { ok: true, graphId: data.id, artifact: compiled.artifact },
      });
      return;
    }

    const graph = data.graphId === null
      ? data.graph && hydrateResolvedGraph(data.graph)
      : graphs.get(data.graphId);
    if (!graph) throw new Error("the graph is no longer available to the analysis worker");
    host.postMessage({
      id: data.id,
      kind: "family",
      result: familyArtifact(graph, data.tiles, data.tensorId),
    });
  } catch (error) {
    host.postMessage({
      id: data.id,
      kind: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
