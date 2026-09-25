import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { hydrateResolvedGraph } from "../core/graph";
import { box } from "../core/region";
import { compileArtifact, reuseArtifact } from "../ui/analysis-jobs";
import type { AnalysisRequest, AnalysisResponse } from "../ui/analysis-protocol";

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
  it("terminates an obsolete compile when a newer build starts", async () => {
    const client = await import("../ui/analysis-worker-client");
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
    const client = await import("../ui/analysis-worker-client");
    const compiled = compileArtifact(CHAIN);
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const graph = hydrateResolvedGraph(structuredClone(compiled.artifact.resolved));

    const family = client.familyInWorker({
      graphId: null,
      graph,
      tiles: { Y: [64, 64] },
      tensorId: "Y",
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
    const client = await import("../ui/analysis-worker-client");
    const compiled = compileArtifact(CHAIN);
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const graph = hydrateResolvedGraph(structuredClone(compiled.artifact.resolved));

    const family = client.familyInWorker({
      graphId: null,
      graph,
      tiles: { Y: [64, 64] },
      tensorId: "Y",
    }).catch((error: unknown) => error);
    const first = FakeWorker.instances[0];
    const region = box([0, 64], [0, 64]);
    const reuse = client.reuseInWorker({ graphId: null, graph, tensorId: "Y", box: region });
    const second = FakeWorker.instances[1];

    expect(first.terminated).toBe(true);
    expect(client.isAnalysisCancelled(await family)).toBe(true);
    expect(second.messages[0].kind).toBe("reuse");

    const request = second.messages[0];
    second.respond({
      id: request.id,
      kind: "reuse",
      result: reuseArtifact(graph, "Y", region),
    });
    await expect(reuse).resolves.toMatchObject({
      estimates: expect.arrayContaining([expect.objectContaining({ tensorId: "A" })]),
    });
  });
});
