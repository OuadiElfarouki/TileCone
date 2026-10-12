import { describe, expect, it } from "vitest";
import { hydrateResolvedGraph, resolvedGraphData } from "../../src/core/graph";
import { compileDSL } from "../../src/parse/compiler";

const GEMM = `M = 256
N = 128
K = 512
L = 32
A = Tensor(M, K, dtype=fp16)
B = Tensor(K, N, dtype=fp16)
C = matmul(A, B)
`;

describe("parameters used by shape resolution", () => {
  it("records distinct bindings in declaration order and retains unused declarations", () => {
    const { graph, resolved } = compileDSL(GEMM);
    expect(resolved.shapeParameters).toEqual(["M", "N", "K"]);
    expect(resolved.params.L).toBe(32);
    expect(graph.tensors.A.resolved).toBeUndefined();
  });

  it("reads exact symbols inside dimension expressions, not prefixes or matching values", () => {
    const source = "K = 4\nKK = 4\nN = 2\nUnused = 8\nX = Tensor((KK + N) * 2)\n";
    const { resolved } = compileDSL(source);
    expect(resolved.shapeParameters).toEqual(["KK", "N"]);
    expect(resolved.tensors.X.resolved).toEqual([12]);
  });

  it.each([
    ["reshape", "P = 4\nQ = 2\nUnused = 8\nX = Tensor(8)\nY = reshape(X, shape=[P, Q])\n", ["P", "Q"]],
    ["split", "P = 2\nQ = 6\nUnused = 8\nX = Tensor(8)\nY, Z = split(X, axis=0, sizes=[P, Q])\n", ["P", "Q"]],
    ["expand", "N = 8\nUnused = 8\nX = Tensor(1)\nY = expand(X, shape=[N])\n", ["N"]],
    ["opaque", 'N = 8\nUnused = 8\nY = opaque(op="Foreign", shapes=[[N]], dtypes=[fp32])\n', ["N"]],
  ])("includes bindings used only by %s output-shape attributes", (_op, source, expected) => {
    expect(compileDSL(source as string).resolved.shapeParameters).toEqual(expected);
  });

  it("ignores comments, axis labels, operator labels and numeric coincidences", () => {
    const source = 'L = 4\n# L is mentioned here\nX = Tensor(L=4)\nY = opaque(X, op="L", shapes=[[4]])\n';
    expect(compileDSL(source).resolved.shapeParameters).toEqual([]);
  });

  it("carries parameter usage across the worker serialization boundary", () => {
    const { resolved } = compileDSL(GEMM);
    const data = structuredClone(resolvedGraphData(resolved));
    expect(data.shapeParameters).toEqual(["M", "N", "K"]);
    expect(hydrateResolvedGraph(data).shapeParameters).toEqual(["M", "N", "K"]);
  });
});
