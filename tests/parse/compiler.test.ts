import { describe, expect, it } from "vitest";
import { full } from "../../src/core/region";
import {
  CompilationError,
  compileDSL,
  tryCompileDSL,
} from "../../src/parse/compiler";
import { parseDSL, toDSL } from "../../src/parse/dsl";
import { resolveGraph, resolveGraphCollecting } from "../../src/core/graph";
import { G } from "../corpus/harness";

describe("DSL compiler facade", () => {
  it("rejects concat operands with different ranks", () => {
    const result = tryCompileDSL(`A = Tensor(2, 3, 4)
B = Tensor(2, 3)
C = concat([A, B], axis=1)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({ phase: "semantic", code: "SEM_SHAPE" });
    expect(result.diagnostics[0].message).toBe("concat: arrays[1] (B) has rank 2, expected rank 3");
  });

  it("rejects extra convolution and pooling spatial attributes", () => {
    for (const source of [
      `X = Tensor(1, 2, 8)\nW = Tensor(3, 2, 3)\nY = conv2d(X, W, stride=[1, 9], padding=[0, 9], dilation=[1, 9])\n`,
      `X = Tensor(1, 2, 8)\nY = max_pool2d(X, [2, 9], padding=[0, 9])\n`,
    ]) {
      const result = tryCompileDSL(source);
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.diagnostics[0].message).toMatch(/has 2 entries for 1 spatial axis/);
    }
  });

  it("rejects normalization over no axes", () => {
    const result = tryCompileDSL(
      "X = Tensor(2, 3)\nY = layer_norm(X, axis=[])\n"
    );
    expect(result.ok).toBe(false);
    if (!result.ok)
      expect(result.diagnostics[0]).toMatchObject({
        code: "DSL_INVALID_ARGUMENT",
        message: "axis: expected an axis or a non-empty list of axes",
      });
  });

  it("keeps special identifier names and rejects __proto__ as an unknown parameter", () => {
    const named = compileDSL("__proto__ = 4\ntoString = Tensor(__proto__)\n");
    expect(named.resolved.tensors["toString"].resolved).toEqual([4]);

    const inherited = tryCompileDSL("X = Tensor(constructor)\n");
    expect(inherited.ok).toBe(false);
    if (!inherited.ok) expect(inherited.diagnostics[0].message).toContain(
      'unbound symbolic dim "constructor"'
    );

    const result = tryCompileDSL("X = Tensor(4)\nY = relu(X, __proto__=123)\n");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.diagnostics[0].message).toBe('relu() has no parameter "__proto__"; relu(x)');
    const call = tryCompileDSL("X = Tensor(4)\nY = __proto__(X)\n");
    expect(call.ok).toBe(false);
    if (!call.ok) expect(call.diagnostics[0].message).toBe('unknown function "__proto__"');
  });

  it("round-trips string attributes that look like boolean literals", () => {
    for (const value of ["true", "false"]) {
      const first = compileDSL(`X = Tensor(4)\nY = opaque(X, op="${value}", shapes=[[4]])\n`);
      const second = compileDSL(toDSL(first.graph));
      expect(second.resolved.topo[0].attrs.op).toBe(value);
    }
  });

  it("reports excessive expression nesting instead of overflowing the stack", () => {
    const depth = 2_000;
    expect(() => tryCompileDSL(`X = Tensor(${"(".repeat(depth)}1${")".repeat(depth)})\n`)).not.toThrow();
    const result = tryCompileDSL(`X = Tensor(${"(".repeat(depth)}1${")".repeat(depth)})\n`);
    expect(result.ok).toBe(false);
  });

  it("reports excessive expression length instead of overflowing during evaluation", () => {
    const expression = new Array(2_000).fill("1").join("+");
    expect(() => tryCompileDSL(`X = Tensor(${expression})\n`)).not.toThrow();
    expect(tryCompileDSL(`X = Tensor(${expression})\n`).ok).toBe(false);
  });

  it("rejects tensors whose exact storage count is not safely representable", () => {
    const result = tryCompileDSL(`X = Tensor(${Number.MAX_SAFE_INTEGER}, 3)\n`);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.diagnostics[0].message).toMatch(/safe integer range/);
  });

  it("applies operation-specific dtype rules", () => {
    for (const source of [
      "X = Tensor(4, dtype=int32)\nY = softmax(X, axis=0)\n",
      "X = Tensor(4, dtype=int32)\nY = mean(X, axis=0)\n",
      "X = Tensor(1, 1, 4, dtype=int32)\nY = avg_pool1d(X, 2, stride=1)\n",
    ]) {
      const result = tryCompileDSL(source);
      expect(result.ok, source).toBe(false);
      if (!result.ok) expect(result.diagnostics[0].code).toBe("SEM_DTYPE");
    }
  });

  it("produces source, unresolved IR, resolved IR, source map, and executor", () => {
    const source = `M = 2
K = 3
N = 4
A = Tensor(M, K, dtype=float16)
B = Parameter(K, N, dtype=float16)
C = matmul(A, B)
`;
    const program = compileDSL(source);

    expect(program.source).toBe(source);
    expect(program.graph.tensors.C.resolved).toBeUndefined();
    expect(program.graph.tensors.C.producer).toBeUndefined();
    expect(program.resolved.tensors.C.resolved).toEqual([2, 4]);
    expect(program.resolved.tensors.C.producer).toEqual({ nodeId: "matmul_C", slot: 0 });
    expect(program.sourceMap.nodes.matmul_C.start.line).toBe(6);
    expect(program.sourceMap.tensors.B.start.line).toBe(5);
    expect(program.executor.graph).toBe(program.resolved);
  });

  it("keeps resolveGraph referentially transparent", () => {
    const graph = parseDSL(`X = Tensor(2, 3, dtype=float32)
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
    const result = tryCompileDSL("  X = Tensor(4, dtype=float32)\n  Y = softmax(X, axis=)\n");
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

  it("underlines the call name for an unknown function", () => {
    const result = tryCompileDSL(`X = Tensor(4, dtype=float32)

Y = mystery(X)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_UNKNOWN_FUNCTION",
      message: 'unknown function "mystery"',
      span: { start: { line: 3 } },
    });
    // The op name itself, not the statement: `mystery` begins at column 5.
    const { start, end } = result.diagnostics[0].span;
    expect(start.column).toBe(5);
    expect(end.column - start.column).toBe("mystery".length);
  });

  it("underlines a keyword the function does not take", () => {
    const result = tryCompileDSL(`X = Tensor(4, dtype=float32)
Y = softmax(X, axiz=-1)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_UNKNOWN_KEYWORD",
      message: 'softmax() has no parameter "axiz" (did you mean axis?); softmax(x, *, axis)',
      span: { start: { line: 2, column: "Y = softmax(X, ".length + 1 } },
    });
  });

  it.each([
    ["relu", "Y = relu()", "DSL_MISSING_ARGUMENT", "relu() is missing x: relu(x)"],
    ["layer_norm", "Y = layer_norm(axis=0)", "DSL_MISSING_ARGUMENT", "layer_norm() is missing x: layer_norm(x, weight=None, bias=None, *, axis=…)"],
    ["concat", "Y = concat([], axis=0)", "DSL_EXPECTED_TENSOR", "arrays: expected a list of tensors, as in [A, B]"],
  ])("requires the tensors %s takes", (_name, statement, code, message) => {
    const result = tryCompileDSL(`${statement}\n`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({ phase: "parse", code, message, span: { start: { line: 1 } } });
  });

  it("leaves a variadic operation's minimum input count to the operation", () => {
    const result = tryCompileDSL('Y = einsum("->")\n');
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({ phase: "semantic", code: "SEM_ARITY" });
    expect(result.diagnostics[0].message).toContain("expected at least 1 input, got 0");
  });

  it("refuses more tensors than a function has places for", () => {
    const result = tryCompileDSL(`X = Tensor(4, dtype=float32)
W = Tensor(4, dtype=float32)
B = Tensor(4, dtype=float32)
Extra = Tensor(4, dtype=float32)
Y = layer_norm(X, W, B, Extra, axis=0)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_TOO_MANY_POSITIONAL",
      message: "layer_norm() takes 3 positional arguments (x, weight, bias), got 4; axis is written by name",
      span: { start: { line: 5, column: "Y = layer_norm(X, W, B, ".length + 1 } },
    });
  });

  it("checks einsum equation operands against its actual input count", () => {
    const result = tryCompileDSL(`A = Tensor(2, 3, dtype=float32)
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

  it("checks split output count against sizes", () => {
    const result = tryCompileDSL(`X = Tensor(4, dtype=float32)
A, B = split(X, [4], axis=0)
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
    const result = tryCompileDSL(`A = Tensor(2, 3, dtype=float32)
B = Tensor(4, 5, dtype=float32)
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
      "conv1d",
      `X = Tensor(2, 3, dtype=float32)
W = Tensor(4, 3, dtype=float32)
Y = conv1d(X, W)
`,
      "expects 1 spatial axis (a rank-3 input), got a rank-2 input",
      3,
    ],
    [
      "max_pool3d",
      `X = Tensor(1, 2, 3, 4, 5, 6, dtype=float32)
Y = max_pool3d(X, 1)
`,
      "expects 3 spatial axes (a rank-5 input), got a rank-6 input",
      2,
    ],
  ])("rejects an operand of another rank than %s is written for", (_name, source, expected, line) => {
    const result = tryCompileDSL(source);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_INVALID_ATTRIBUTES",
      span: { start: { line } },
    });
    expect(result.diagnostics[0].message).toContain(expected);
  });

  it("requires convolution weight rank to match activation rank", () => {
    const result = tryCompileDSL(`X = Tensor(1, 3, 8, 8, dtype=float32)
W = Tensor(4, 3, 3, dtype=float32)
Y = conv2d(X, W, padding=1)
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
    const program = compileDSL(`X = Tensor(4, dtype=float32)
Y = astype(X, float16)
Z = contiguous(Y)
`);

    expect(program.resolved.tensors.X.dtype).toBe("f32");
    expect(program.resolved.tensors.Y.dtype).toBe("f16");
    expect(program.resolved.tensors.Z.dtype).toBe("f16");
  });

  it("infers gather output dtype from data and accepts standard integer indices", () => {
    for (const dtype of ["int32", "int64"]) {
      const valid = compileDSL(`D = Tensor(8, 4, dtype=float16)
I = Tensor(3, dtype=${dtype})
Y = take(D, I, axis=0, index_values=[1, 5, 2])
`);
      expect(valid.resolved.tensors.Y.dtype).toBe("f16");
    }

    const invalid = tryCompileDSL(`D = Tensor(8, 4, dtype=float16)
I = Tensor(3, dtype=float32)
Y = take(D, I, axis=0)
`);
    expect(invalid.ok).toBe(false);
    if (invalid.ok) return;
    expect(invalid.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_DTYPE",
      span: { start: { line: 3 } },
    });
    expect(invalid.diagnostics[0].message).toBe("take: indices must be int32 or int64, got float32");
  });

  /* Compute ops promote; data-movement ops do not. The split is deliberate:
     an fp16 activation against an fp32 residual is ordinary inference practice
     and has one obvious result, while concatenating them has none - the output
     is a single buffer and the author has to say which type it holds. */
  it.each([
    ["float16 + float32", "float16", "float32", "f32"],
    ["float16 + bfloat16 widens to float32", "float16", "bfloat16", "f32"],
    ["float8 + float16", "float8", "float16", "f16"],
    ["int8 + float16", "int8", "float16", "f16"],
    ["int8 + int32", "int8", "int32", "i32"],
  ])("promotes mixed compute inputs: %s", (_label, a, b, expected) => {
    const result = tryCompileDSL(`A = Tensor(4, dtype=${a})
B = Tensor(4, dtype=${b})
Y = add(A, B)
`);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.program.resolved.tensors.Y.dtype).toBe(expected);
  });

  it("promotes through a matmul and keeps each operand's own bytes", () => {
    const result = tryCompileDSL(`A = Tensor(4, 8, dtype=float16)
W = Parameter(8, 4, dtype=float32)
Y = matmul(A, W)
`);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.program.resolved.tensors.Y.dtype).toBe("f32");
    expect(result.program.resolved.tensors.A.dtype).toBe("f16");
    expect(result.program.resolved.tensors.W.dtype).toBe("f32");
  });

  it("still rejects mixed dtypes where there is no result type to pick", () => {
    const result = tryCompileDSL(`A = Tensor(2, 3, dtype=float16)
B = Tensor(4, 3, dtype=float32)
Y = concat([A, B], axis=0)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_DTYPE",
      span: { start: { line: 3 } },
    });
    expect(result.diagnostics[0].message).toMatch(/input dtypes must match.*float16, float32/);
    expect(result.diagnostics[0].message).toMatch(/insert an astype/);
  });

  /* A dtype is spelled by the binder, because the DSL spells dtypes `float32`
     and the IR spells them `f32`. Owning the conversion means owning the
     rejection: an unknown spelling cannot reach the op's attribute kind, so it
     is a parse error with a span on the value rather than a semantic one on the
     statement. `f64` is doubly wrong here - an internal spelling, and not a
     dtype this system has. */
  it("rejects an unknown cast target when it reads the dtype", () => {
    const result = tryCompileDSL("X = Tensor(4, dtype=float32)\nY = astype(X, f64)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_INVALID_ARGUMENT",
      span: { start: { line: 2, column: "Y = astype(X, ".length + 1 } },
    });
    expect(result.diagnostics[0].message).toMatch(/^dtype: expected one of float32, /);
  });

  it("accepts a cast written in the DSL's own dtype spelling", () => {
    // float16, not bool: the two spellings differ between the DSL and the IR,
    // so this shows the conversion happening rather than a name that survives.
    const result = tryCompileDSL("X = Tensor(4, dtype=float32)\nY = astype(X, float16)\n");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.program.graph.nodes[0].attrs).toMatchObject({ dtype: "f16" });
    expect(result.program.resolved.tensors.Y.dtype).toBe("f16");
  });

  it.each([
    ["softmax", "Y = softmax(X, axis=2)"],
    ["cumsum", "Y = cumulative_sum(X, axis=-3)"],
    ["reduce", "Y = sum(X, axis=9)"],
    ["concat", "Y = concat([X, X], axis=9)"],
  ])("rejects an out-of-range %s axis during compilation, at the attribute", (_name, statement) => {
    const result = tryCompileDSL(`X = Tensor(2, 8, dtype=float32)\n${statement}\n`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    const axisAt = statement.search(/ax(is|es)=/) + 1;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_INVALID_ATTRIBUTES",
      span: { start: { line: 2, column: axisAt } },
    });
    expect(result.diagnostics[0].message).toMatch(/axis .* out of range for rank 2/);
  });

  it("slices the axes a subscript leaves out whole, as NumPy does", () => {
    const program = compileDSL(`X = Tensor(2, 8, dtype=float32)
Y = X[0:1]
`);
    expect(program.resolved.tensors.Y.resolved).toEqual([1, 8]);
  });

  it("rejects non-representable inferred extents at the graph boundary", () => {
    const extent = Math.floor(Number.MAX_SAFE_INTEGER / 2) + 1;
    const result = tryCompileDSL(`X = Tensor(${extent}, dtype=float8)
Y = concat([X, X], axis=0)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({ phase: "semantic", code: "SEM_SHAPE" });
    expect(result.diagnostics[0].message).toMatch(/inferred output 0, axis 0 has invalid extent/);
  });

  it("maps unresolved input dimensions back to the declaration", () => {
    const result = tryCompileDSL(`X = Tensor(Missing, 4, dtype=float32)
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
      compileDSL("X = Tensor(4, dtype=float32)\nY = nope(X)\n");
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(CompilationError);
    expect((caught as CompilationError).diagnostics[0].code).toBe("DSL_UNKNOWN_FUNCTION");
    expect((caught as Error).message).toMatch(/^line 2:/);
  });

  it("accepts identifiers beginning with params", () => {
    const program = compileDSL(`X = Tensor(4, dtype=float32)
paramsResult = identity(X)
`);
    expect(program.resolved.tensors.paramsResult.resolved).toEqual([4]);
  });

  it("rejects duplicate parameter definitions instead of silently overwriting", () => {
    const result = tryCompileDSL("M = 4\nM = 8\nX = Tensor(M, dtype=float32)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_DUPLICATE_PARAM",
      span: { start: { line: 2 } },
    });
  });

  /* Dimensions, tensors, and operation outputs are all assignments now, so they
     share one namespace and can collide in ways the keyword-separated grammar
     could not express. Silently letting the later binding win is the dangerous
     outcome: a shape would then be resolved against a name whose value the
     reader cannot see on the line that uses it. */
  it.each([
    ["a tensor over a dimension", "M = 4\nM = Tensor(4, dtype=float32)\n", "DSL_DUPLICATE_TENSOR"],
    ["a dimension over a tensor", "M = Tensor(4, dtype=float32)\nM = 8\n", "DSL_DUPLICATE_PARAM"],
    ["an operation output over a dimension", "M = 4\nX = Tensor(M, dtype=float32)\nM = relu(X)\n", "DSL_DUPLICATE_TENSOR"],
  ])("rejects %s claiming a name already bound", (_case, source, code) => {
    const result = tryCompileDSL(source);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code,
      // The second binding is the error, not the first: the name was available
      // when it was first used.
      span: { start: { line: source.trimEnd().split("\n").length } },
    });
    expect(result.diagnostics[0].message).toMatch(/symbol "M" redefined/);
  });

  it("validates parameter bindings even when they are not referenced", () => {
    const result = tryCompileDSL("\nM = 0\nX = Tensor(4, dtype=float32)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_SHAPE",
      span: { start: { line: 2 } },
    });
  });

  it("rejects a reduction over no axes", () => {
    // Accepted, it returns its input unchanged under a name that says a
    // reduction happened, and the dependency note it would carry is silent.
    const result = tryCompileDSL("X = Tensor(4, 6, dtype=float32)\nY = sum(X, axis=[])\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "semantic",
      code: "SEM_INVALID_ATTRIBUTES",
      span: { start: { line: 2, column: "Y = sum(X, ".length + 1 } },
    });
  });

  it("rejects a parameter given twice, at the second one", () => {
    const result = tryCompileDSL(`X = Tensor(4, dtype=float32)
Y = softmax(X, axis=0, axis=-1)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      phase: "parse",
      code: "DSL_DUPLICATE_ARGUMENT",
      span: { start: { line: 2, column: "Y = softmax(X, axis=0, ".length + 1 } },
    });
  });

  it("preserves hashes and escaped quotes inside string attributes", () => {
    // A lexer test: `#` inside a string is not a comment, `\"` survives, and the
    // trailing comment is still stripped.
    const graph = parseDSL(`X = Tensor(4, dtype=float32)
Y = opaque(X, op="a#b \\"quoted\\"", shapes=[[4]]) # an actual comment
`);
    expect(graph.nodes[0].attrs.op).toBe('a#b "quoted"');
  });

  it("rejects a keyword the function does not take", () => {
    const result = tryCompileDSL(`X = Tensor(4, dtype=float32)
Y = identity(X, note="whatever")
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      code: "DSL_UNKNOWN_KEYWORD",
      message: 'identity() has no parameter "note"; identity(x)',
      span: { start: { line: 2 } },
    });
  });

  it("names the near miss when a keyword is misspelled, or is another library's", () => {
    // `keepdim` is torch's spelling of `keepdims`. Accepted, it would have to
    // mean the same thing, and the language would have two names for it.
    const torch = tryCompileDSL(`X = Tensor(4, 8, dtype=float32)
Y = sum(X, axis=-1, keepdim=true)
`);
    expect(torch.ok).toBe(false);
    if (!torch.ok) expect(torch.diagnostics[0].message).toMatch(/^sum\(\) has no parameter "keepdim" \(did you mean keepdims\?\)/);
    const typo = tryCompileDSL(`X = Tensor(4, 8, dtype=float32)
Y = sum(X, axsi=-1)
`);
    expect(typo.ok).toBe(false);
    if (!typo.ok) expect(typo.diagnostics[0].message).toMatch(/did you mean axis\?/);
  });

  it("gives the signature when nothing is close", () => {
    const result = tryCompileDSL(`X = Tensor(4, 8, dtype=float32)
Y = sum(X, axis=-1, banana=true)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0].message).toBe('sum() has no parameter "banana"; sum(x, *, axis=…, keepdims=…)');
  });

  it("rejects a keyword on a function that takes none", () => {
    const result = tryCompileDSL(`A = Tensor(4, 8, dtype=float32)
B = Parameter(8, 4, dtype=float32)
C = matmul(A, B, transposeB=true)
`);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0].message).toBe('matmul() has no parameter "transposeB"; matmul(x1, x2)');
  });
});

/* What the operations themselves enforce, whatever front end built the graph.
   The DSL cannot write most of these - a signature binds inputs and implied
   attributes consistently - so they are checked on graphs directly. */
describe("operation invariants below the DSL", () => {
  const firstError = (graph: ReturnType<typeof G>) => resolveGraphCollecting(graph).errors[0];

  it("checks elementwise nary against its actual input count", () => {
    const error = firstError(G({ X: [4] }, [["n", "elementwise", ["X"], ["Y"], { fn: "relu", nary: 2 }]]));
    expect(error.code).toBe("GRAPH_ARITY");
    expect(error.message).toMatch(/nary=2 does not match 1 input/);
  });

  it("checks a function's own arity", () => {
    const error = firstError(G({ A: [4], B: [4] }, [["n", "elementwise", ["A", "B"], ["Y"], { fn: "relu", nary: 2 }]]));
    expect(error.message).toMatch(/relu takes exactly 1 input/);
  });

  it("checks normalize flags against its actual input count", () => {
    const error = firstError(
      G({ X: [4], W: [4] }, [["n", "normalize", ["X", "W"], ["Y"], { kind: "layer_norm", axes: [0], hasWeight: false, hasBias: false }]])
    );
    expect(error.code).toBe("GRAPH_ARITY");
    expect(error.message).toMatch(/flags require 1 input, got 2/);
  });

  it("enforces the maximum normalize input count", () => {
    const error = firstError(
      G({ X: [4], W: [4], B: [4], E: [4] }, [["n", "normalize", ["X", "W", "B", "E"], ["Y"], { kind: "layer_norm", axes: [0] }]])
    );
    expect(error.code).toBe("GRAPH_ARITY");
    expect(error.message).toMatch(/expected 1 to 3 inputs, got 4/);
  });

  it.each([
    ["conv", G({ X: [2, 3], W: [4, 3] }, [["n", "conv", ["X", "W"], ["Y"], {}]]), 2],
    ["pool", G({ X: [1, 2, 3, 4, 5, 6] }, [["n", "pool", ["X"], ["Y"], { kernelShape: 1 }]]), 6],
  ])("rejects unsupported %s activation ranks", (op, graph, rank) => {
    const error = firstError(graph);
    expect(error.code).toBe("GRAPH_SHAPE");
    expect(error.message).toContain(`${op}: activation rank ${rank} is unsupported`);
  });
});

/* The elementwise function table is the definition of what `fn` may be, so
   these check the two holes it closes: an unrecognized name used to resolve
   and quietly cost one FLOP, and a function's own arity was never checked
   against the operands it was actually given. */
describe("elementwise function table", () => {
  it("rejects an unknown function, naming the near one", () => {
    const result = tryCompileDSL("X = Tensor(4)\nY = guelu(X)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      code: "DSL_UNKNOWN_FUNCTION",
      message: 'unknown function "guelu" (did you mean gelu?)',
    });
  });

  it("rejects a unary function given two operands", () => {
    const result = tryCompileDSL("A = Tensor(4)\nB = Tensor(4)\nY = relu(A, B)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0]).toMatchObject({
      code: "DSL_TOO_MANY_POSITIONAL",
      message: "relu() takes 1 positional argument (x), got 2",
    });
  });

  it("rejects a strictly binary function given three operands", () => {
    const result = tryCompileDSL("A = Tensor(4)\nB = Tensor(4)\nC = Tensor(4)\nY = divide(A, B, C)\n");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.diagnostics[0].message).toBe("divide() takes 2 positional arguments (x1, x2), got 3");
  });

  it("accepts an n-ary chain of an associative function", () => {
    const result = tryCompileDSL("A = Tensor(4)\nB = Tensor(4)\nC = Tensor(4)\nY = add(A, B, C)\n");
    expect(result.ok).toBe(true);
  });

  it("charges a transcendental more than an add", () => {
    const cheap = compileDSL("X = Tensor(64)\nY = relu(X)\n");
    const dear = compileDSL("X = Tensor(64)\nY = gelu(X)\n");
    const flopsOf = (p: ReturnType<typeof compileDSL>) =>
      p.executor.metrics("Y", full(p.resolved.tensors.Y.resolved!)).flops.value!;
    expect(flopsOf(dear)).toBeGreaterThan(flopsOf(cheap));
  });
});
