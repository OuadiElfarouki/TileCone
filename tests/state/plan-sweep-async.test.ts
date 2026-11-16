import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { compileArtifact, sweepArtifact } from "../../src/worker/analysis-jobs";
import type { AnalysisRequest, AnalysisResponse } from "../../src/worker/analysis-protocol";

const DSL = "A = Tensor(4, 4)\nB = Tensor(4, 4)\nY = matmul(A, B)\n";
const REPLACEMENT = "X = Tensor(4, 4)\nY = relu(X)\n";

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
  postMessage(message: AnalysisRequest): void { this.messages.push(message); }
  terminate(): void { this.terminated = true; }
  respond(response: AnalysisResponse): void {
    this.onmessage?.({ data: response } as MessageEvent<AnalysisResponse>);
  }
}

beforeEach(() => {
  vi.resetModules();
  FakeWorker.instances = [];
  vi.stubGlobal("Worker", FakeWorker);
});
afterEach(() => { vi.unstubAllGlobals(); });

describe("asynchronous sweep cancellation", () => {
  it("returns to idle when an abandoned successful build cancels the current sweep", async () => {
    const { useStore } = await import("../../src/state/store");
    const state = () => useStore.getState();
    state().applyDSL(DSL);
    expect(state().dslText).toBe(DSL);
    const graph = state().resolved;
    const sweeping = state().runSweep("all");
    const sweepWorker = FakeWorker.instances.find((worker) => worker.name === "tilecone-sweep")!;
    expect(state().sweep.status).toBe("running");
    const building = state().applyDSLAsync(REPLACEMENT);
    state().setDraftText(`${REPLACEMENT}# changed\n`);
    const compileWorker = FakeWorker.instances.find((worker) => worker.name === "tilecone-compile")!;
    const request = compileWorker.messages.find((message) => message.kind === "compile")!;
    const compiled = compileArtifact(REPLACEMENT);
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    compileWorker.respond({ id: request.id, kind: "compile", result: {
      ok: true, graphId: 7, artifact: compiled.artifact,
    } });
    await expect(building).resolves.toBe(false);
    await sweeping;
    expect(sweepWorker.terminated).toBe(true);
    expect(state().resolved).toBe(graph);
    expect(state().sweep).toEqual({ status: "idle" });
  });

  it("does not let a cancelled sweep reset a newer sweep", async () => {
    const { useStore } = await import("../../src/state/store");
    const state = () => useStore.getState();
    state().applyDSL(DSL);
    expect(state().dslText).toBe(DSL);
    const first = state().runSweep("all");
    const second = state().runSweep("current");
    await first;
    expect(state().sweep).toEqual({ status: "running", scope: "current" });
    const workers = FakeWorker.instances.filter((entry) => entry.name === "tilecone-sweep");
    const worker = workers[workers.length - 1];
    const request = worker.messages.find((message) => message.kind === "sweep")!;
    const result = sweepArtifact(state().resolved!, { budget: state().planBudget });
    worker.respond({ id: request.id, kind: "sweep", result });
    await second;
    expect(state().sweep).toMatchObject({ status: "done", scope: "current", result });
  });
});
