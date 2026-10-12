import { describe, expect, it } from "vitest";
import { hydrateResolvedGraph } from "../../src/core/graph";
import { compileArtifact, layoutArtifact, planArtifact, planRestoreArtifact, replayArtifact } from "../../src/worker/analysis-jobs";
import { box } from "../../src/core/region";

const CHAIN = `A = Tensor(256, 256, dtype=fp16)
B = Tensor(256, 256, dtype=fp16)
W = Tensor(256, 128, dtype=fp16)
C = matmul(A, B)
Y = matmul(C, W)
`;

describe("analysis worker jobs", () => {
  it("lays out remapped planes identically from live and serialized graph data", () => {
    const compiled = compileArtifact("X = Tensor(4, 32, 256)\nY = relu(X)\n");
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const tensorId = Object.values(compiled.resolved.tensors).find((tensor) => tensor.name === "X")!.id;
    const axes: Record<string, [number, number]> = { [tensorId]: [0, 1] };
    const layout = layoutArtifact(compiled.resolved, compiled.artifact.graphPx, axes);
    expect(layoutArtifact(structuredClone(compiled.artifact.resolved), compiled.artifact.graphPx, axes)).toEqual(layout);
    expect(structuredClone(layout)).toEqual(layout);
    const remapped = layout.nodes.find((node) => node.kind === "tensor" && node.id === tensorId)!;
    const original = compiled.artifact.layout.nodes.find((node) => node.kind === "tensor" && node.id === tensorId)!;
    expect([remapped.w, remapped.h]).not.toEqual([original.w, original.h]);
    expect(layoutArtifact(compiled.resolved, compiled.artifact.graphPx)).toEqual(compiled.artifact.layout);
  });

  it("returns a structured-cloneable resolved graph and structural layout", () => {
    const result = compileArtifact(CHAIN);
    if (!result.ok) throw new Error(result.diagnostics[0].message);

    expect("shapesOf" in result.artifact.resolved).toBe(false);
    expect(() => structuredClone(result.artifact)).not.toThrow();
    expect(result.artifact.layout.nodes).toHaveLength(
      Object.keys(result.artifact.resolved.tensors).length + result.artifact.resolved.nodes.length
    );
    expect(result.artifact.layout.nodes.every((node) =>
      [node.x, node.y, node.w, node.h].every(Number.isFinite)
    )).toBe(true);

    const hydrated = hydrateResolvedGraph(structuredClone(result.artifact.resolved));
    expect(hydrated.shapesOf(["C", "Y"])).toEqual([[256, 256], [256, 128]]);
  });

  it("preserves compiler diagnostics across the worker boundary", () => {
    const result = compileArtifact("Y = relu(Missing)\nZ = relu(AlsoMissing)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics.map((diagnostic) => diagnostic.span.start.line)).toEqual([1, 2]);
  });

  it("evaluates a plan from serialized graph data, into a report that crosses back", () => {
    const result = compileArtifact(CHAIN);
    if (!result.ok) throw new Error(result.diagnostics[0].message);

    const report = planArtifact(structuredClone(result.artifact.resolved), {
      C: [64, 64],
      Y: [64, 64],
    });
    expect(() => structuredClone(report)).not.toThrow();
    const y = report.families.find((family) => family.tensorId === "Y")!;
    expect(y.status).toBe("evaluated");
    if (y.status !== "evaluated") return;
    expect(y.tasks).toBe(8);
    expect(y.boundary.map((row) => row.tensorId)).toEqual(["C", "W"]);
    expect(report.total?.tasks).toBe(report.families.reduce((n, family) => n + family.tasks, 0));
  });

  it("returns a structured-cloneable family replay for canvas playback", () => {
    const result = compileArtifact(CHAIN);
    if (!result.ok) throw new Error(result.diagnostics[0].message);

    const replay = replayArtifact(
      structuredClone(result.artifact.resolved),
      { C: [64, 64], Y: [64, 64] },
      { tensorId: "Y", coord: [0, 0] }
    );
    const cloned = structuredClone(replay);
    expect(cloned.frames[0].reach).toBeInstanceOf(Map);
    expect(cloned.frames.length).toBe(replay.tasks);
    expect(cloned.frames[0].region.boxes).toEqual([box([0, 64], [0, 64])]);
  });
});


it("validates every restored plan and task before returning cloneable measurements", () => {
  const compiled = compileArtifact(CHAIN);
  if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
  const request = { tiles: { C: [64, 64], Y: [64, 64] },
    task: { tensorId: "Y", coord: [0, 0] }, kept: [{ Y: [128, 128] }] };
  const restored = planRestoreArtifact(compiled.resolved, request);
  expect(structuredClone(restored)).toEqual(restored);
  expect(restored.reports[0]).toEqual(planArtifact(compiled.resolved, request.kept[0]));
  expect(() => planRestoreArtifact(compiled.resolved, { ...request, kept: [{ Missing: [1] }] })).toThrow();
  expect(() => planRestoreArtifact(compiled.resolved, { ...request, task: { tensorId: "Y", coord: [99, 0] } })).toThrow();
  expect(() => planRestoreArtifact(compiled.resolved, { ...request, tiles: { Y: [0, 64] } })).toThrow();
});
