import { describe, expect, it } from "vitest";
import {
  CompilationError,
  compileDSL,
  tryCompileDSL,
} from "../parse/compiler";
import { parseDSL } from "../parse/dsl";
import { resolveGraph } from "../core/graph";

describe("DSL compiler facade", () => {
  it("produces source, unresolved IR, resolved IR, source map, and executor", () => {
    const source = `params M=2 K=3 N=4
input A [M, K] f16
weight B [K, N] f16
C = matmul(A, B)
`;
    const program = compileDSL(source);

    expect(program.source).toBe(source);
    expect(program.graph.tensors.C.resolved).toBeUndefined();
    expect(program.graph.tensors.C.producer).toBeUndefined();
    expect(program.resolved.tensors.C.resolved).toEqual([2, 4]);
    expect(program.resolved.tensors.C.producer).toEqual({ nodeId: "matmul_C", slot: 0 });
    expect(program.sourceMap.nodes.matmul_C.start.line).toBe(4);
    expect(program.sourceMap.tensors.B.start.line).toBe(3);
    expect(program.executor.graph).toBe(program.resolved);
  });

  it("keeps resolveGraph referentially transparent", () => {
    const graph = parseDSL(`input X [2, 3] f32
Y = softmax(X, axis=-1)
`);
    const before = JSON.stringify(graph);
    const first = resolveGraph(graph);
    const second = resolveGraph(graph);

    expect(JSON.stringify(graph)).toBe(before);
    expect(first).not.toBe(second);
    expect(first.tensors).not.toBe(graph.tensors);
    expect(first.tensors.Y.resolved).toEqual(second.tensors.Y.resolved);
  });

  it("normalizes syntax errors with precise source coordinates", () => {
    const result = tryCompileDSL("  input X [4] f32\n  Y = softmax(X, axis=)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;

    expect(result.diagnostics).toHaveLength(1);
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_SYNTAX",
      severity: "error",
      span: { start: { line: 2 } },
    });
    expect(result.diagnostics[0].span.start.column).toBeGreaterThan(2);
  });

  it("maps an unknown op diagnostic back to its DSL statement", () => {
    const result = tryCompileDSL(`input X [4] f32

Y = mystery(X)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_UNKNOWN_OP",
      span: { start: { line: 3, column: 1 } },
    });
    expect(result.diagnostics[0].message).toMatch(/node "mystery_Y"/);
  });

  it("maps invalid attrs back to the operation statement", () => {
    const result = tryCompileDSL(`input X [4] f32
Y = softmax(X, axiz=-1)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_INVALID_ATTRIBUTES",
      span: { start: { line: 2 } },
    });
  });

  it("maps shape-inference failures back to the operation statement", () => {
    const result = tryCompileDSL(`input A [2, 3] f32
input B [4, 5] f32
C = matmul(A, B)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_SHAPE",
      span: { start: { line: 3 } },
    });
  });

  it("maps unresolved input dimensions back to the declaration", () => {
    const result = tryCompileDSL(`input X [Missing, 4] f32
Y = softmax(X, axis=-1)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_SHAPE",
      span: { start: { line: 1 } },
    });
    expect(result.diagnostics[0].message).toMatch(/tensor "X"/);
  });

  it("throws a structured CompilationError from the convenience API", () => {
    let caught: unknown;
    try {
      compileDSL("input X [4] f32\nY = nope(X)\n");
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(CompilationError);
    expect((caught as CompilationError).diagnostics[0].code).toBe("SEM_UNKNOWN_OP");
    expect((caught as Error).message).toMatch(/^line 2:/);
  });

  it("does not mistake identifiers beginning with params for a params statement", () => {
    const program = compileDSL(`input X [4] f32
paramsResult = identity(X)
`);
    expect(program.resolved.tensors.paramsResult.resolved).toEqual([4]);
  });

  it("rejects duplicate parameter definitions instead of silently overwriting", () => {
    const result = tryCompileDSL("params M=4 M=8\ninput X [M] f32\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_DUPLICATE_PARAM",
      span: { start: { line: 1 } },
    });
  });

  it("validates parameter bindings even when they are not referenced", () => {
    const result = tryCompileDSL("\nparams M=0\ninput X [4] f32\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_SHAPE",
      span: { start: { line: 2 } },
    });
  });

  it("rejects duplicate named attributes", () => {
    const result = tryCompileDSL(`input X [4] f32
Y = softmax(X, axis=0, axis=-1)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_DUPLICATE_ATTRIBUTE",
      span: { start: { line: 2 } },
    });
  });

  it("preserves hashes and escaped quotes inside string attributes", () => {
    const program = compileDSL(`input X [4] f32
Y = identity(X, note="a#b \\"quoted\\"") # an actual comment
`);
    expect(program.graph.nodes[0].attrs.note).toBe('a#b "quoted"');
  });
});
