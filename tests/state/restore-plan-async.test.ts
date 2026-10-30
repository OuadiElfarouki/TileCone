/** Restore is a single compile-lane transaction, isolated from inspector queries. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { compileArtifact, planArtifact, planRestoreArtifact } from "../../src/worker/analysis-jobs";
import type { AnalysisRequest, AnalysisResponse } from "../../src/worker/analysis-protocol";

const CHAIN = "A = Tensor(16, 16, dtype=float16)\nB = Tensor(16, 16, dtype=float16)\n" +
  "W = Tensor(16, 8, dtype=float16)\nC = matmul(A, B)\nY = matmul(C, W)\n";

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

/** The first unanswered request of `kind` on a live worker, once one is posted. */
async function pending(kind: AnalysisRequest["kind"], answered: Set<number>) {
  let found: { worker: FakeWorker; request: AnalysisRequest } | undefined;
  await vi.waitFor(() => {
    for (const worker of FakeWorker.instances) {
      if (worker.terminated) continue;
      const request = worker.messages.find((m) => m.kind === kind && !answered.has(m.id));
      if (request) found = { worker, request };
    }
    expect(found).toBeDefined();
  });
  answered.add(found!.request.id);
  return found!;
}

const compiled = compileArtifact(CHAIN);
if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);

const WORKSPACE = {
  dsl: CHAIN,
  direction: "backward" as const,
  tileScale: 0,
  snapToGrid: true,
  axisMode: "symbolic" as const,
  parts: null,
  planTiles: { C: [4, 4], Y: [4, 4] },
  planTask: { tensorId: "Y", coord: [1, 1] },
  keptTilings: [{ Y: [8, 8] }, { C: [16, 16], Y: [4, 8] }] as Record<string, number[]>[],
  planView: true,
};

beforeEach(() => {
  vi.resetModules();
  FakeWorker.instances = [];
  vi.stubGlobal("Worker", FakeWorker);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("restoring a shared plan through the Worker", () => {
  it("evaluates each kept tiling before installing the workspace", async () => {
    const { useStore } = await import("../../src/state/store");
    const answered = new Set<number>();
    const restoring = useStore.getState().restoreWorkspaceAsync(WORKSPACE);

    const restore = await pending("restore", answered);
    expect(restore.request).toMatchObject({ plan: {
      tiles: WORKSPACE.planTiles, task: WORKSPACE.planTask, kept: WORKSPACE.keptTilings,
    } });
    expect(useStore.getState().dslText).not.toBe(CHAIN);
    restore.worker.respond({ id: restore.request.id, kind: "restore", result: {
      ok: true, graphId: 7, artifact: compiled.artifact,
      restoredPlan: planRestoreArtifact(compiled.resolved, {
        tiles: WORKSPACE.planTiles, task: WORKSPACE.planTask, kept: WORKSPACE.keptTilings,
      }),
    } });

    await expect(restoring).resolves.toBe(true);
    const state = useStore.getState();
    expect(state.dslText).toBe(CHAIN);
    expect({ ...state.planTiles }).toEqual(WORKSPACE.planTiles);
    expect(state.planTask).toEqual(WORKSPACE.planTask);
    expect(state.inspectorTab).toBe("plan");
    expect(state.keptPlans.map((k) => ({ ...k.tiles }))).toEqual(WORKSPACE.keptTilings);
    expect(state.keptPlans.map((k) => k.total)).toEqual(
      WORKSPACE.keptTilings.map((tiles) => planArtifact(compiled.resolved, tiles).total)
    );
  });

  it("gives way to a build started while a kept plan is being measured", async () => {
    const { useStore } = await import("../../src/state/store");
    const answered = new Set<number>();
    const restoring = useStore.getState().restoreWorkspaceAsync(WORKSPACE);

    const restore = await pending("restore", answered);

    const other = "X = Tensor(4, 4, dtype=float32)\nY = relu(X)\n";
    const building = useStore.getState().applyDSLAsync(other);
    expect(restore.worker.terminated).toBe(true);
    await expect(restoring).resolves.toBe(false);

    const next = compileArtifact(other);
    if (!next.ok) throw new Error(next.diagnostics[0].message);
    const rebuild = await pending("compile", answered);
    rebuild.worker.respond({ id: rebuild.request.id, kind: "compile", result: { ok: true, graphId: 8, artifact: next.artifact } });
    await expect(building).resolves.toBe(true);

    const state = useStore.getState();
    expect(state.dslText).toBe(other);
    expect(state.plan).toBeNull();
    expect(state.keptPlans).toEqual([]);
  });

  it("isolates restore from a cancelled inspector query that retries the old graph", async () => {
    const { useStore } = await import("../../src/state/store");
    const client = await import("../../src/state/analysis-worker-client");
    const oldArgs = { graphId: 3, graph: compiled.resolved, tiles: { Y: [8, 8] } };
    const oldReport = client.planInWorker(oldArgs).catch((error: unknown) => {
      if (!client.isAnalysisCancelled(error)) throw error;
      return client.planInWorker(oldArgs).catch(() => null);
    });
    const restoring = useStore.getState().restoreWorkspaceAsync(WORKSPACE);
    const answered = new Set<number>();
    const restore = await pending("restore", answered);
    const competing = client.planInWorker(oldArgs).catch(() => null);
    await Promise.resolve(); // the cancelled old inspector retries
    expect(restore.worker.terminated).toBe(false);
    restore.worker.respond({ id: restore.request.id, kind: "restore", result: {
      ok: true, graphId: 7, artifact: compiled.artifact,
      restoredPlan: planRestoreArtifact(compiled.resolved, {
        tiles: WORKSPACE.planTiles, task: WORKSPACE.planTask, kept: WORKSPACE.keptTilings,
      }),
    } });
    await expect(restoring).resolves.toBe(true);
    await Promise.all([oldReport, competing]);
    expect(useStore.getState().keptPlans).toHaveLength(2);
  });

  it("installs a large task's worker summary without computing it again on the UI thread", async () => {
    const { useStore } = await import("../../src/state/store");
    const jobs = await import("../../src/worker/analysis-jobs");
    const dsl = "X = Tensor(100000000)\nP = relu(X)\nY = sum(P, axis=0)\n";
    const large = compileArtifact(dsl);
    if (!large.ok) throw new Error(large.diagnostics[0].message);
    const workspace = { ...WORKSPACE, dsl, planTiles: { P: [1], Y: [] },
      planTask: { tensorId: "Y", coord: [] }, keptTilings: [] };
    const restoredPlan = planRestoreArtifact(large.resolved, {
      tiles: workspace.planTiles, task: workspace.planTask, kept: [],
    });
    const fallback = vi.spyOn(jobs, "planRestoreArtifact").mockImplementation(() => {
      throw new Error("restore measurements must come from the worker");
    });
    try {
      const restoring = useStore.getState().restoreWorkspaceAsync(workspace);
      const restore = await pending("restore", new Set());
      restore.worker.respond({ id: restore.request.id, kind: "restore", result: {
        ok: true, graphId: 7, artifact: large.artifact, restoredPlan: structuredClone(restoredPlan),
      } });
      await expect(restoring).resolves.toBe(true);
      expect(fallback).not.toHaveBeenCalled();
      expect(useStore.getState().planSupply?.producerGroups[0].count).toBe(100000000);
    } finally { fallback.mockRestore(); }
  });
});
