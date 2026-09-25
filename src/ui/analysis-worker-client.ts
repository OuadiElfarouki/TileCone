import {
  resolvedGraphData,
  type ResolvedGraph,
  type ResolvedGraphData,
} from "../core/graph";
import type { InterfaceReport } from "../core/plan/interfaces";
import type { ReuseSweep } from "../core/reuse";
import type { Box } from "../core/region";
import type {
  AnalysisRequest,
  AnalysisResponse,
  CompileJobResult,
} from "./analysis-protocol";

type Pending = {
  resolve: (response: AnalysisResponse) => void;
  reject: (error: Error) => void;
};

type WorkerLane = {
  worker: Worker | null;
  pending: Map<number, Pending>;
  graphIds: Set<number>;
};

type AnalysisRequestBody =
  | Omit<Extract<AnalysisRequest, { kind: "compile" }>, "id">
  | Omit<Extract<AnalysisRequest, { kind: "family" }>, "id">
  | Omit<Extract<AnalysisRequest, { kind: "reuse" }>, "id">;

let nextRequestId = 1;
const compileLane: WorkerLane = {
  worker: null,
  pending: new Map(),
  graphIds: new Set(),
};
const queryLane: WorkerLane = {
  worker: null,
  pending: new Map(),
  graphIds: new Set(),
};

export const analysisWorkerAvailable = (): boolean => typeof Worker !== "undefined";

class AnalysisCancelledError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AnalysisCancelledError";
  }
}

export function isAnalysisCancelled(error: unknown): boolean {
  return error instanceof Error && error.name === "AnalysisCancelledError";
}

function discardLane(lane: WorkerLane, error: Error): void {
  const current = lane.worker;
  lane.worker = null;
  lane.graphIds.clear();
  for (const request of lane.pending.values()) request.reject(error);
  lane.pending.clear();
  current?.terminate();
}

function createCompileWorker(): Worker {
  return new Worker(new URL("./analysis.worker.ts", import.meta.url), {
    type: "module",
    name: "tilecone-compile",
  });
}

function createQueryWorker(): Worker {
  return new Worker(new URL("./analysis.worker.ts", import.meta.url), {
    type: "module",
    name: "tilecone-query",
  });
}

function workerFor(lane: WorkerLane): Worker {
  if (lane.worker) return lane.worker;
  const created = lane === compileLane ? createCompileWorker() : createQueryWorker();
  lane.worker = created;
  created.onmessage = ({ data }: MessageEvent<AnalysisResponse>) => {
    if (lane.worker !== created) return;
    const request = lane.pending.get(data.id);
    if (!request) return;
    lane.pending.delete(data.id);
    if (data.kind === "error") request.reject(new Error(data.message));
    else request.resolve(data);
  };
  created.onerror = (event) => {
    if (lane.worker !== created) return;
    const error = new Error(event.message || "analysis worker failed");
    discardLane(lane, error);
  };
  return created;
}

function send(lane: WorkerLane, request: AnalysisRequestBody): Promise<AnalysisResponse> {
  const id = nextRequestId++;
  return new Promise((resolve, reject) => {
    lane.pending.set(id, { resolve, reject });
    try {
      workerFor(lane).postMessage({ ...request, id } as AnalysisRequest);
    } catch (error) {
      discardLane(
        lane,
        error instanceof Error ? error : new Error("could not contact analysis worker"),
      );
    }
  });
}

function cancelPending(lane: WorkerLane, task: string): void {
  if (lane.pending.size === 0) return;
  discardLane(lane, new AnalysisCancelledError(`superseded ${task}`));
}

function registerQueryGraph(graphId: number, graph: ResolvedGraphData): void {
  discardLane(queryLane, new AnalysisCancelledError("the compiled graph changed"));
  const id = nextRequestId++;
  try {
    workerFor(queryLane).postMessage({
      id,
      kind: "register",
      graphId,
      graph,
    } satisfies AnalysisRequest);
    queryLane.graphIds.add(graphId);
  } catch (error) {
    discardLane(
      queryLane,
      error instanceof Error ? error : new Error("could not initialize analysis worker"),
    );
  }
}

/**
 * The graph fields of one query: the id alone when this lane's worker is known
 * to hold that graph, and the whole graph otherwise.
 *
 * Only correct after the lane has been cleared for this query. Superseding the
 * lane discards its worker, and with it every graph that worker had cached, so
 * reading the cache first would claim a graph nobody holds and the worker
 * would refuse the query. `takeQueryLane` does both in that order.
 */
function graphForQuery(
  graphId: number | null,
  graph: ResolvedGraph,
): { graphId: number | null; graph?: ReturnType<typeof resolvedGraphData> } {
  if (graphId !== null && queryLane.graphIds.has(graphId)) return { graphId };
  if (graphId !== null) queryLane.graphIds.add(graphId);
  return { graphId, graph: resolvedGraphData(graph) };
}

/** Supersede whatever the query lane was doing and describe the graph the
 *  replacement needs. Latest-wins: the lane answers one question at a time. */
function takeQueryLane(
  graphId: number | null,
  graph: ResolvedGraph,
): { graphId: number | null; graph?: ReturnType<typeof resolvedGraphData> } {
  cancelPending(queryLane, "graph analysis");
  return graphForQuery(graphId, graph);
}

export async function compileInWorker(source: string): Promise<CompileJobResult> {
  cancelPending(compileLane, "compilation");
  const response = await send(compileLane, { kind: "compile", source });
  if (response.kind !== "compile") throw new Error("analysis worker returned the wrong response");
  if (response.result.ok) {
    registerQueryGraph(response.result.graphId, response.result.artifact.resolved);
  }
  return response.result;
}

export async function familyInWorker(args: {
  graphId: number | null;
  graph: ResolvedGraph;
  tiles: Record<string, number[]>;
  tensorId: string;
}): Promise<InterfaceReport> {
  const request = takeQueryLane(args.graphId, args.graph);
  const response = await send(queryLane, {
    kind: "family",
    ...request,
    tiles: args.tiles,
    tensorId: args.tensorId,
  });
  if (response.kind !== "family") throw new Error("analysis worker returned the wrong response");
  return response.result;
}

export async function reuseInWorker(args: {
  graphId: number | null;
  graph: ResolvedGraph;
  tensorId: string;
  box: Box;
}): Promise<ReuseSweep> {
  const request = takeQueryLane(args.graphId, args.graph);
  const response = await send(queryLane, {
    kind: "reuse",
    ...request,
    tensorId: args.tensorId,
    box: args.box,
  });
  if (response.kind !== "reuse") throw new Error("analysis worker returned the wrong response");
  return response.result;
}
