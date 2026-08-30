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

  it.each([
    ["elementwise", "Y = elementwise(fn=relu, nary=1)", "at least 1 input"],
    ["concat", "Y = concat(axis=0)", "at least 1 input"],
    ["einsum", 'Y = einsum("->")', "at least 1 input"],
    [
      "normalize",
      "Y = normalize(kind=layernorm, axes=[0], hasWeight=false, hasBias=false)",
      "1 to 3 inputs",
    ],
  ])("requires at least one input for variadic %s", (_op, statement, expected) => {
    const result = tryCompileDSL(`${statement}\n`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_ARITY",
      span: { start: { line: 1 } },
    });
    expect(result.diagnostics[0].message).toContain(`expected ${expected}, got 0`);
  });

  it("enforces the maximum normalize input count", () => {
    const result = tryCompileDSL(`input X [4] f32
input W [4] f32
input B [4] f32
input Extra [4] f32
Y = normalize(X, W, B, Extra, kind=layernorm, axes=[0], hasWeight=true, hasBias=true)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_ARITY",
      span: { start: { line: 5 } },
    });
    expect(result.diagnostics[0].message).toMatch(/expected 1 to 3 inputs, got 4/);
  });

  it("checks elementwise nary against its actual input count", () => {
    const result = tryCompileDSL(`input X [4] f32
Y = elementwise(X, fn=relu, nary=2)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_ARITY",
      span: { start: { line: 2 } },
    });
    expect(result.diagnostics[0].message).toMatch(/nary=2 does not match 1 input/);
  });

  it("checks einsum equation operands against its actual input count", () => {
    const result = tryCompileDSL(`input A [2, 3] f32
Y = einsum("ij,jk->ik", A)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_ARITY",
      span: { start: { line: 2 } },
    });
    expect(result.diagnostics[0].message).toMatch(
      /equation declares 2 operands, got 1 input/
    );
  });

  it("checks normalize flags against its actual input count", () => {
    const result = tryCompileDSL(`input X [4] f32
input W [4] f32
Y = normalize(X, W, kind=layernorm, axes=[0], hasWeight=false, hasBias=false)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_ARITY",
      span: { start: { line: 3 } },
    });
    expect(result.diagnostics[0].message).toMatch(/flags require 1 input, got 2/);
  });

  it("checks split output count against sizes", () => {
    const result = tryCompileDSL(`input X [4] f32
A, B = split(X, axis=0, sizes=[4])
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_ARITY",
      span: { start: { line: 2 } },
    });
    expect(result.diagnostics[0].message).toMatch(/sizes declares 1 outputs, got 2/);
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

  it.each([
    [
      "conv",
      `input X [2, 3] f32
input W [4, 3] f32
Y = conv(X, W, stride=[], pads=[], dilation=[], groups=1)
`,
      2,
      3,
    ],
    [
      "pool",
      `input X [1, 2, 3, 4, 5, 6] f32
Y = pool(X, kind=max, kernelShape=[1,1,1,1], stride=[1,1,1,1], pads=[[0,0],[0,0],[0,0],[0,0]])
`,
      6,
      2,
    ],
  ])("rejects unsupported %s activation ranks", (op, source, rank, line) => {
    const result = tryCompileDSL(source);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_SHAPE",
      span: { start: { line } },
    });
    expect(result.diagnostics[0].message).toContain(
      `${op}: activation rank ${rank} is unsupported`
    );
  });

  it("requires convolution weight rank to match activation rank", () => {
    const result = tryCompileDSL(`input X [1, 3, 8, 8] f32
input W [4, 3, 3] f32
Y = conv(X, W, stride=[1,1], pads=[[1,1],[1,1]], dilation=[1,1], groups=1)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_SHAPE",
      span: { start: { line: 3 } },
    });
    expect(result.diagnostics[0].message).toMatch(
      /weight rank 3 must match activation rank 4/
    );
  });

  it("infers cast dtypes through subsequent preserving operations", () => {
    const program = compileDSL(`input X [4] f32
Y = cast(X, dtype=f16)
Z = contiguous(Y)
`);

    expect(program.resolved.tensors.X.dtype).toBe("f32");
    expect(program.resolved.tensors.Y.dtype).toBe("f16");
    expect(program.resolved.tensors.Z.dtype).toBe("f16");
  });

  it("infers gather output dtype from data and requires i32 indices", () => {
    const valid = compileDSL(`input D [8, 4] f16
input I [3] i32
Y = gather(D, I, axis=0, indexValues=[1, 5, 2])
`);
    expect(valid.resolved.tensors.Y.dtype).toBe("f16");

    const invalid = tryCompileDSL(`input D [8, 4] f16
input I [3] f32
Y = gather(D, I, axis=0)
`);
    expect(invalid.ok).toBe(false);
    if (invalid.ok) return;
    expect(invalid.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_DTYPE",
      span: { start: { line: 3 } },
    });
    expect(invalid.diagnostics[0].message).toMatch(/indices must be i32/);
  });

  it("rejects ambiguous mixed-dtype compute inputs", () => {
    const result = tryCompileDSL(`input A [4] f16
input B [4] f32
Y = add(A, B)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_DTYPE",
      span: { start: { line: 3 } },
    });
    expect(result.diagnostics[0].message).toMatch(/input dtypes must match.*f16, f32/);
  });

  it("rejects an unknown cast target as an invalid attribute", () => {
    const result = tryCompileDSL("input X [4] f32\nY = cast(X, dtype=f64)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_INVALID_ATTRIBUTES",
      span: { start: { line: 2 } },
    });
  });

  it.each([
    ["softmax", "Y = softmax(X, axis=2)"],
    ["cumsum", "Y = cumsum(X, axis=-3)"],
    ["reduce", "Y = sum(X, axes=[9])"],
    ["concat", "Y = concat(X, X, axis=9)"],
  ])("rejects an out-of-range %s axis during compilation", (_name, statement) => {
    const result = tryCompileDSL(`input X [2, 8] f32\n${statement}\n`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_SHAPE",
      span: { start: { line: 2 } },
    });
    expect(result.diagnostics[0].message).toMatch(/axis .* out of range for rank 2/);
  });

  it("rejects mismatched slice attribute ranks", () => {
    const result = tryCompileDSL(`input X [2, 8] f32
Y = slice(X, starts=[0, 0], stops=[2], steps=[1, 1])
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({ phase: "semantic", code: "SEM_SHAPE" });
    expect(result.diagnostics[0].message).toMatch(/stops has length 1, expected rank 2/);
  });

  it("rejects non-representable inferred extents at the graph boundary", () => {
    const extent = Number.MAX_SAFE_INTEGER;
    const result = tryCompileDSL(`input X [${extent}] f32
Y = concat(X, X, axis=0)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({ phase: "semantic", code: "SEM_SHAPE" });
    expect(result.diagnostics[0].message).toMatch(/inferred output 0, axis 0 has invalid extent/);
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
