import { describe, expect, it } from "vitest";
import { hydrateResolvedGraph } from "../core/graph";
import { compileArtifact, familyArtifact } from "../ui/analysis-jobs";

const CHAIN = `A = Tensor(256, 256, dtype=fp16)
B = Tensor(256, 256, dtype=fp16)
W = Tensor(256, 128, dtype=fp16)
C = matmul(A, B)
Y = matmul(C, W)
`;

describe("analysis worker jobs", () => {
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

  it("evaluates a tiled family from serialized graph data", () => {
    const result = compileArtifact(CHAIN);
    if (!result.ok) throw new Error(result.diagnostics[0].message);

    const report = familyArtifact(
      structuredClone(result.artifact.resolved),
      { C: [64, 64], Y: [64, 64] },
      "Y"
    );
    expect(report.status).toBe("evaluated");
    if (report.status !== "evaluated") return;
    expect(report.tasks).toBe(8);
    expect(report.boundary.map((row) => row.tensorId)).toEqual(["C", "W"]);
  });
});
