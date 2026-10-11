import {
  resolvedGraphData,
  type ResolvedGraph,
  type ResolvedGraphData,
} from "../core/graph";
import type { PlanReport } from "../core/plan/interfaces";
import type { ReuseSurface, ReuseSweep } from "../core/reuse";
import type { Box } from "../core/region";
import type { BaseGraphLayout } from "../view/graph/graph-scene";
import type {
  AnalysisRequest,
  AnalysisResponse,
  CompileJobResult,
  PlanRestoreRequest,
  RestoreJobResult,
} from "../worker/analysis-protocol";

type Pending = {
  resolve: (response: AnalysisResponse) => void;
  reject: (error: Error) => void;
};

type WorkerLane = {
  createWorker: () => Worker;
  worker: Worker | null;
  pending: Map<number, Pending>;
  graphIds: Set<number>;
};

type AnalysisRequestBody =
  | Omit<Extract<AnalysisRequest, { kind: "compile" }>, "id">
  | Omit<Extract<AnalysisRequest, { kind: "restore" }>, "id">
  | Omit<Extract<AnalysisRequest, { kind: "layout" }>, "id">
  | Omit<Extract<AnalysisRequest, { kind: "plan" }>, "id">
  | Omit<Extract<AnalysisRequest, { kind: "reuse" }>, "id">;

function createLane(createWorker: () => Worker): WorkerLane {
  return { createWorker, worker: null, pending: new Map(), graphIds: new Set() };
}

let nextRequestId = 1;
// Vite requires static Worker URLs and options. The factories differ only in
// their debugging names; caching, cancellation, and messaging stay shared.
const compileLane = createLane(() => new Worker(new URL("../worker/analysis.worker.ts", import.meta.url), {
  type: "module", name: "tilecone-compile",
}));
const queryLane = createLane(() => new Worker(new URL("../worker/analysis.worker.ts", import.meta.url), {
  type: "module", name: "tilecone-query",
}));
// Layout must not cancel a plan/reuse query, or wait behind it. It changes
// independently when a tensor's displayed axes change.
const layoutLane = createLane(() => new Worker(new URL("../worker/analysis.worker.ts", import.meta.url), {
  type: "module", name: "tilecone-layout",
}));

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

function workerFor(lane: WorkerLane): Worker {
  if (lane.worker) return lane.worker;
  const created = lane.createWorker();
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
  discardLane(layoutLane, new AnalysisCancelledError("the compiled graph changed"));
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
 * The graph fields of one job: the id alone when this lane's worker is known
 * to hold that graph, and the whole graph otherwise.
 *
 * Only correct after pending work in the lane has been cancelled. Superseding the
 * lane discards its worker, and with it every graph that worker had cached, so
 * reading the cache first would claim a graph nobody holds and the worker
 * would refuse the job. Both layout and query requests cancel before calling.
 */
function graphForLane(
  lane: WorkerLane,
  graphId: number | null,
  graph: ResolvedGraph,
): { graphId: number | null; graph?: ReturnType<typeof resolvedGraphData> } {
  if (graphId !== null && lane.graphIds.has(graphId)) return { graphId };
  if (graphId !== null) {
    // The worker replaces its single cached graph when it receives graph data.
    lane.graphIds.clear();
    lane.graphIds.add(graphId);
  }
  return { graphId, graph: resolvedGraphData(graph) };
}

export async function layoutInWorker(args: {
  graphId: number | null;
  graph: ResolvedGraph;
  graphPx: number;
  axes: Record<string, [number, number]>;
}): Promise<BaseGraphLayout> {
  cancelPending(layoutLane, "graph layout");
  const response = await send(layoutLane, {
    kind: "layout", ...graphForLane(layoutLane, args.graphId, args.graph),
    graphPx: args.graphPx, axes: args.axes,
  });
  if (response.kind !== "layout") throw new Error("analysis worker returned the wrong response");
  return response.result;
}

/** Retiring a remap also cancels its work, including when the default layout
 * is already available and no replacement request will be sent. */
export function cancelLayout(): void {
  cancelPending(layoutLane, "graph layout");
}

/** Supersede whatever the query lane was doing and describe the graph the
 *  replacement needs. Latest-wins: the lane answers one question at a time. */
function takeQueryLane(
  graphId: number | null,
  graph: ResolvedGraph,
): { graphId: number | null; graph?: ReturnType<typeof resolvedGraphData> } {
  cancelPending(queryLane, "graph analysis");
  return graphForLane(queryLane, graphId, graph);
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

/** A restore is one latest-wins compile-lane transaction. Inspector retries
 * remain on the query lane and cannot interrupt its kept-plan measurements. */
export async function restoreInWorker(source: string, plan: PlanRestoreRequest): Promise<RestoreJobResult> {
  cancelPending(compileLane, "workspace restore");
  const response = await send(compileLane, { kind: "restore", source, plan });
  if (response.kind !== "restore") throw new Error("analysis worker returned the wrong response");
  if (response.result.ok) registerQueryGraph(response.result.graphId, response.result.artifact.resolved);
  return response.result;
}

export async function planInWorker(args: {
  graphId: number | null;
  graph: ResolvedGraph;
  tiles: Record<string, number[]>;
}): Promise<PlanReport> {
  const request = takeQueryLane(args.graphId, args.graph);
  const response = await send(queryLane, {
    kind: "plan",
    ...request,
    tiles: args.tiles,
  });
  if (response.kind !== "plan") throw new Error("analysis worker returned the wrong response");
  return response.result;
}

export async function reuseInWorker(args: {
  graphId: number | null;
  graph: ResolvedGraph;
  tensorId: string;
  box: Box;
  surfaces: ReuseSurface[];
}): Promise<ReuseSweep> {
  const request = takeQueryLane(args.graphId, args.graph);
  const response = await send(queryLane, {
    kind: "reuse",
    ...request,
    tensorId: args.tensorId,
    box: args.box,
    surfaces: args.surfaces,
  });
  if (response.kind !== "reuse") throw new Error("analysis worker returned the wrong response");
  return response.result;
}
