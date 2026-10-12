import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { hydrateResolvedGraph } from "../../src/core/graph";
import { compileArtifact, layoutArtifact, planArtifact, replayArtifact } from "../../src/worker/analysis-jobs";
import type { AnalysisRequest, AnalysisResponse } from "../../src/worker/analysis-protocol";

const CHAIN = `A = Tensor(256, 256, dtype=fp16)
B = Tensor(256, 256, dtype=fp16)
C = matmul(A, B)
Y = relu(C)
`;

class FakeWorker {
  static instances: FakeWorker[] = [];

  readonly messages: AnalysisRequest[] = [];
  readonly name: string;
  terminated = false;
  onmessage: ((event: MessageEvent<AnalysisResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  constructor(_url: URL, options?: WorkerOptions) {
    this.name = options?.name ?? "";
    FakeWorker.instances.push(this);
  }

  postMessage(message: AnalysisRequest): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  respond(response: AnalysisResponse): void {
    this.onmessage?.({ data: response } as MessageEvent<AnalysisResponse>);
  }
}

beforeEach(() => {
  vi.resetModules();
  FakeWorker.instances = [];
  vi.stubGlobal("Worker", FakeWorker);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("analysis worker client", () => {
  it("isolates layout from queries and resends graph data when a layout is superseded", async () => {
    const client = await import("../../src/state/analysis-worker-client");
    const compiled = compileArtifact(CHAIN);
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const graph = compiled.resolved;
    const tiles = { Y: [64, 64] };
    const task = { tensorId: "Y", coord: [0, 0] };
    const query = client.replayInWorker({ graphId: 42, graph, tiles, task, depth: 1 });
    const queryWorker = FakeWorker.instances[0];
    const first = client.layoutInWorker({ graphId: 42, graph, graphPx: 1, axes: {} }).catch((error: unknown) => error);
    const oldLayout = FakeWorker.instances[1];
    const second = client.layoutInWorker({ graphId: 42, graph, graphPx: 2, axes: {} });
    const layoutWorker = FakeWorker.instances[2];
    expect(oldLayout.terminated).toBe(true);
    expect(client.isAnalysisCancelled(await first)).toBe(true);
    expect(queryWorker.terminated).toBe(false);
    expect(layoutWorker.name).toBe("tilecone-layout");
    expect(layoutWorker.messages[0]).toMatchObject({ kind: "layout", graphId: 42, graph: compiled.artifact.resolved, graphPx: 2 });

    const layout = layoutArtifact(graph, 2);
    layoutWorker.respond({ id: layoutWorker.messages[0].id, kind: "layout", result: layout });
    await expect(second).resolves.toEqual(layout);
    const cached = client.layoutInWorker({ graphId: 42, graph, graphPx: 3, axes: {} }).catch((error: unknown) => error);
    expect(layoutWorker.messages[1]).not.toHaveProperty("graph");
    client.cancelLayout();
    expect(layoutWorker.terminated).toBe(true);
    expect(client.isAnalysisCancelled(await cached)).toBe(true);
    expect(queryWorker.terminated).toBe(false);
    queryWorker.respond({ id: queryWorker.messages[0].id, kind: "replay", result: replayArtifact(graph, tiles, task) });
    await expect(query).resolves.toHaveProperty("frames");
  });

  it("retires a stale layout when compilation replaces the graph", async () => {
    const client = await import("../../src/state/analysis-worker-client");
    const compiled = compileArtifact(CHAIN);
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const layout = client.layoutInWorker({ graphId: 41, graph: compiled.resolved, graphPx: 1, axes: {} }).catch((error: unknown) => error);
    const layoutWorker = FakeWorker.instances[0];
    const compilation = client.compileInWorker(CHAIN);
    const compileWorker = FakeWorker.instances[1];
    compileWorker.respond({ id: compileWorker.messages[0].id, kind: "compile", result: { ok: true, graphId: 42, artifact: compiled.artifact } });
    await compilation;
    expect(layoutWorker.terminated).toBe(true);
    expect(client.isAnalysisCancelled(await layout)).toBe(true);
  });

  it("terminates an obsolete compile when a newer build starts", async () => {
    const client = await import("../../src/state/analysis-worker-client");
    const obsolete = client.compileInWorker("Y = relu(A)\n").catch((error: unknown) => error);
    const first = FakeWorker.instances[0];
    const current = client.compileInWorker(CHAIN);
    const second = FakeWorker.instances[1];

    expect(first.terminated).toBe(true);
    expect(client.isAnalysisCancelled(await obsolete)).toBe(true);
    const request = second.messages[0];
    second.respond({ id: request.id, kind: "compile", result: { ok: false, diagnostics: [] } });
    await expect(current).resolves.toEqual({ ok: false, diagnostics: [] });
  });

  it("keeps compilation off the query queue and replaces stale analysis after compile", async () => {
    const client = await import("../../src/state/analysis-worker-client");
    const compiled = compileArtifact(CHAIN);
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const graph = hydrateResolvedGraph(structuredClone(compiled.artifact.resolved));

    const family = client.planInWorker({
      graphId: null,
      graph,
      tiles: { Y: [64, 64] },
    }).catch((error: unknown) => error);
    const oldQuery = FakeWorker.instances[0];

    const compile = client.compileInWorker(CHAIN);
    const compileWorker = FakeWorker.instances[1];
    expect(oldQuery.name).toBe("tilecone-query");
    expect(compileWorker.name).toBe("tilecone-compile");
    expect(compileWorker.messages[0].kind).toBe("compile");
    expect(oldQuery.terminated).toBe(false);

    const request = compileWorker.messages[0];
    compileWorker.respond({
      id: request.id,
      kind: "compile",
      result: { ok: true, graphId: 42, artifact: compiled.artifact },
    });

    await expect(compile).resolves.toMatchObject({ ok: true, graphId: 42 });
    expect(client.isAnalysisCancelled(await family)).toBe(true);
    expect(oldQuery.terminated).toBe(true);
    expect(FakeWorker.instances[2].messages[0]).toMatchObject({
      kind: "register",
      graphId: 42,
    });
  });

  it("terminates a superseded query instead of waiting for it", async () => {
    const client = await import("../../src/state/analysis-worker-client");
    const compiled = compileArtifact(CHAIN);
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const graph = hydrateResolvedGraph(structuredClone(compiled.artifact.resolved));

    const family = client.planInWorker({
      graphId: null,
      graph,
      tiles: { Y: [64, 64] },
    }).catch((error: unknown) => error);
    const first = FakeWorker.instances[0];
    const tiles = { Y: [64, 64] };
    const task = { tensorId: "Y", coord: [0, 0] };
    const replay = client.replayInWorker({ graphId: null, graph, tiles, task, depth: 1 });
    const second = FakeWorker.instances[1];

    expect(first.terminated).toBe(true);
    expect(client.isAnalysisCancelled(await family)).toBe(true);
    expect(second.messages[0].kind).toBe("replay");

    const request = second.messages[0];
    second.respond({
      id: request.id,
      kind: "replay",
      result: replayArtifact(graph, tiles, task),
    });
    // Sixteen tasks of Y, all under the frame cap, so every one is a frame.
    await expect(replay).resolves.toMatchObject({ tasks: 16 });
    expect((await replay).frames).toHaveLength(16);
  });
});


it("resends graph data when returning to a graph evicted by a completed query", async () => {
  const client = await import("../../src/state/analysis-worker-client");
  const compiled = compileArtifact(CHAIN);
  if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
  for (const graphId of [41, 42, 41]) {
    const query = client.planInWorker({ graphId, graph: compiled.resolved, tiles: {} });
    const worker = FakeWorker.instances[0];
    const request = worker.messages[worker.messages.length - 1];
    expect(request).toMatchObject({ graphId, graph: compiled.artifact.resolved });
    worker.respond({ id: request.id, kind: "plan", result: planArtifact(compiled.resolved, {}) });
    await query;
  }
});
