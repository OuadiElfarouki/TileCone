import { describe, expect, it } from "vitest";
import { hydrateResolvedGraph } from "../../src/core/graph";
import { compileArtifact, planArtifact, reuseArtifact } from "../../src/worker/analysis-jobs";
import { box } from "../../src/core/region";

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

  it("returns a structured-cloneable reuse trace for canvas playback", () => {
    const result = compileArtifact(CHAIN);
    if (!result.ok) throw new Error(result.diagnostics[0].message);

    const sweep = reuseArtifact(
      structuredClone(result.artifact.resolved),
      "Y",
      box([0, 64], [0, 64])
    );
    expect(() => structuredClone(sweep)).not.toThrow();
    expect(sweep.frames).toHaveLength(8);
    expect(sweep.frames[0].box).toEqual(box([0, 64], [0, 64]));
  });
});
