import { resolvedGraphData, type ResolvedGraph } from "../core/graph";
import type { InterfaceReport } from "../core/plan/interfaces";
import type {
  AnalysisRequest,
  AnalysisResponse,
  CompileJobResult,
} from "./analysis-protocol";

type Pending = {
  resolve: (response: AnalysisResponse) => void;
  reject: (error: Error) => void;
};

type AnalysisRequestBody =
  | Omit<Extract<AnalysisRequest, { kind: "compile" }>, "id">
  | Omit<Extract<AnalysisRequest, { kind: "family" }>, "id">;

let worker: Worker | null = null;
let nextRequestId = 1;
const pending = new Map<number, Pending>();

export const analysisWorkerAvailable = (): boolean => typeof Worker !== "undefined";

function analysisWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./analysis.worker.ts", import.meta.url), {
    type: "module",
    name: "tilecone-analysis",
  });
  worker.onmessage = ({ data }: MessageEvent<AnalysisResponse>) => {
    const request = pending.get(data.id);
    if (!request) return;
    pending.delete(data.id);
    if (data.kind === "error") request.reject(new Error(data.message));
    else request.resolve(data);
  };
  worker.onerror = (event) => {
    const error = new Error(event.message || "analysis worker failed");
    for (const request of pending.values()) request.reject(error);
    pending.clear();
    worker?.terminate();
    worker = null;
  };
  return worker;
}

function send(request: AnalysisRequestBody): Promise<AnalysisResponse> {
  const id = nextRequestId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    analysisWorker().postMessage({ ...request, id } as AnalysisRequest);
  });
}

export async function compileInWorker(source: string): Promise<CompileJobResult> {
  const response = await send({ kind: "compile", source });
  if (response.kind !== "compile") throw new Error("analysis worker returned the wrong response");
  return response.result;
}

export async function familyInWorker(args: {
  graphId: number | null;
  graph: ResolvedGraph;
  tiles: Record<string, number[]>;
  tensorId: string;
}): Promise<InterfaceReport> {
  const response = await send({
    kind: "family",
    graphId: args.graphId,
    ...(args.graphId === null ? { graph: resolvedGraphData(args.graph) } : {}),
    tiles: args.tiles,
    tensorId: args.tensorId,
  });
  if (response.kind !== "family") throw new Error("analysis worker returned the wrong response");
  return response.result;
}
