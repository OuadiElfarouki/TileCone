/**
 * Multi-error recovery and argument-level spans.
 *
 * The old front end reported exactly one diagnostic per compile and underlined
 * whole statements, so a document with four independent mistakes took four
 * edit-compile cycles and never said which part of a line was wrong. These
 * pin both behaviours.
 */

import { describe, expect, it } from "vitest";
import { tryCompileDSL } from "../../src/parse/compiler";
import { parseProgram } from "../../src/parse/parser";
import { lowerProgram } from "../../src/parse/lower";
import { resolveGraph } from "../../src/core/graph";
import { G } from "../corpus/harness";

const diagnosticsOf = (source: string) => {
  const result = tryCompileDSL(source);
  return result.ok ? [] : result.diagnostics;
};

describe("error recovery: every independent failure is reported", () => {
  it("reports four unrelated errors in one pass, in source order", () => {
    const diagnostics = diagnosticsOf(`A = Tensor(4, 5)
B = Tensor(9, 7)
C = matmul(A, B)
D = relu(Q)
E = softmax(A)
F = softmax(B, axis=5)
`);
    expect(diagnostics.map((d) => d.span.start.line)).toEqual([3, 4, 5, 6]);
    expect(diagnostics.map((d) => d.code)).toEqual([
      "SEM_SHAPE",
      "DSL_UNKNOWN_TENSOR",
      "DSL_MISSING_ARGUMENT",
      "SEM_INVALID_ATTRIBUTES",
    ]);
  });

  it("keeps going after a line that does not parse at all", () => {
    const diagnostics = diagnosticsOf(`A = Tensor(4, 4)
this is not a statement
B = Tensor(2, 2)
C = matmul(A, B)
`);
    // The bad line, then the genuinely wrong matmul below it.
    expect(diagnostics.map((d) => d.span.start.line)).toEqual([2, 4]);
    expect(diagnostics[0].phase).toBe("parse");
    expect(diagnostics[1].phase).toBe("semantic");
  });

  it("does not cascade: a statement reading a failed one is silent", () => {
    const diagnostics = diagnosticsOf(`A = Tensor(4, 5)
B = Tensor(9, 7)
C = matmul(A, B)
D = relu(C)
E = relu(D)
F = relu(E)
`);
    // Only the matmul is wrong; C, D, E, F are all consequences of it.
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0].span.start.line).toBe(3);
  });

  it("does not cascade from an unparseable declaration either", () => {
    const diagnostics = diagnosticsOf(`X = Tensor(4, 4
Y = relu(X)
Z = relu(Y)
`);
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0].span.start.line).toBe(1);
  });

  it("reports independent errors on both sides of a good statement", () => {
    const diagnostics = diagnosticsOf(`A = Tensor(4, 4, dtype=quux)
B = Tensor(4, 4)
C = relu(B)
D = Tensor(4, 4, dtype=alsobad)
`);
    expect(diagnostics.map((d) => d.span.start.line)).toEqual([1, 4]);
  });

  it("reports every unknown tensor in one statement separately", () => {
    const diagnostics = diagnosticsOf("Y = matmul(Missing1, Missing2)\n");
    expect(diagnostics).toHaveLength(2);
    expect(diagnostics.map((d) => d.message)).toEqual([
      'unknown tensor "Missing1"',
      'unknown tensor "Missing2"',
    ]);
  });

  it("still compiles a clean program with no diagnostics", () => {
    const result = tryCompileDSL("A = Tensor(4, 8)\nB = Tensor(8, 2)\nC = matmul(A, B)\n");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.diagnostics).toEqual([]);
    expect(result.program.resolved.tensors.C.resolved).toEqual([4, 2]);
  });
});

describe("argument-level spans", () => {
  /** The exact text a diagnostic underlines. */
  const underlined = (source: string, index = 0) => {
    const d = diagnosticsOf(source)[index];
    return source.slice(d.span.start.offset, d.span.end.offset);
  };

  it("underlines the offending tensor reference, not the line", () => {
    expect(underlined("A = Tensor(4)\nY = matmul(A, Nope)\n")).toBe("Nope");
  });

  it("underlines the misspelled keyword", () => {
    expect(underlined("X = Tensor(4, 5)\nY = sum(X, axis=1, keepdim=true)\n")).toBe("keepdim");
  });

  it("underlines the value a kind refuses", () => {
    expect(underlined("X = Tensor(4, 5)\nY = sum(X, axis=1.5)\n")).toBe("1.5");
  });

  it("underlines the bad dtype value in a declaration", () => {
    expect(underlined("X = Tensor(4, dtype=float64)\n")).toBe("float64");
  });

  it("underlines the call name for an unknown function", () => {
    expect(underlined("X = Tensor(4)\nY = nosuchop(X)\n")).toBe("nosuchop");
  });

  it("falls back to the statement when nothing narrower is known", () => {
    // A shape mismatch is about the pairing of two operands, not either one.
    expect(underlined("A = Tensor(4, 5)\nB = Tensor(9, 7)\nC = matmul(A, B)\n")).toBe(
      "C = matmul(A, B)"
    );
  });

  it("records a span for every input and named attribute", () => {
    const { program } = parseProgram("X = Tensor(4, 5)\nY = sum(X, axis=1, keepdims=true)\n");
    const { sourceMap } = lowerProgram(program);
    const args = sourceMap.nodeArgs.reduce_Y;
    expect(args.inputs).toHaveLength(1);
    // By the operation's attribute names; `fn` is implied by the name `sum`,
    // which has no span of its own beyond the callee.
    expect(Object.keys(args.attrs).sort()).toEqual(["axes", "keepdim"]);
    expect(args.callee.start.column).toBe(5);
  });
});

describe("the AST is produced before anything interprets it", () => {
  it("parses sugar as a plain call, leaving desugaring to lowering", () => {
    const { program, errors } = parseProgram("X = Tensor(4)\nY = relu(X)\n");
    expect(errors).toEqual([]);
    const call = program.stmts[1];
    expect(call.kind).toBe("call");
    if (call.kind !== "call") return;
    // The parser knows nothing about the op registry: `relu` is still `relu`.
    expect(call.callee.value).toBe("relu");
    expect(call.args).toHaveLength(1);
    // Lowering is what turns it into an elementwise node.
    expect(lowerProgram(program).graph.nodes[0]).toMatchObject({
      op: "elementwise",
      attrs: { fn: "relu", nary: 1 },
    });
  });

  it("keeps an error statement in place of a line it could not read", () => {
    const { program, errors } = parseProgram("A = Tensor(4)\n???\nB = relu(A)\n");
    expect(errors).toHaveLength(1);
    expect(program.stmts.map((s) => s.kind)).toEqual(["declare", "error", "call"]);
  });
});

/* The resolver speaks in graph terms - node ids, registry attribute names,
   "input 1" - and the compiler puts the source's terms back, by one rule read
   off the signature, so no message says something the author did not write. */
describe("messages in the source's terms", () => {
  const message = (source: string) => {
    const result = tryCompileDSL(source);
    return result.ok ? result.diagnostics[0]?.message : result.diagnostics[0].message;
  };

  it("names the function as called and each operand by its parameter", () => {
    expect(message("A = Tensor(4, 8)\nB = Tensor(4, 5)\nC = matmul(A, B)\n")).toBe(
      "matmul: the contracted axis differs: axis 1 of x1 (A) is 8, axis 0 of x2 (B) is 4"
    );
    expect(message("A = Tensor(2, 3, 4)\nB = Tensor(2, 3)\nC = concat([A, B], axis=1)\n")).toBe(
      "concat: arrays[1] (B) has rank 2, expected rank 3"
    );
  });

  it("names an attribute by its keyword, and one the name implies by the name", () => {
    expect(message("X = Tensor(1, 3, 8, 8)\nW = Tensor(4, 3, 3, 3)\nY = conv2d(X, W, padding=[1, 1, 1])\n")).toBe(
      "conv2d: padding: has 3 entries for 2 axes"
    );
    expect(message("X = Tensor(2, 8)\nW = Tensor(4, 3)\nY = conv1d(X, W)\n")).toBe(
      "conv1d: expects 1 spatial axis (a rank-3 input), got a rank-2 input"
    );
  });

  it("names a subscript by the tensor it slices", () => {
    expect(message("X = Tensor(4, 6)\nZ = X[:, :, :]\n")).toBe("X[…]: has 3 entries for 2 axes");
  });

  it("spells dtypes the DSL's way", () => {
    expect(message("A = Tensor(4, dtype=fp16)\nB = Tensor(4, dtype=fp32)\nC = concat([A, B])\n")).toBe(
      "concat: input dtypes must match, got [fp16, fp32] - insert an astype(...) to choose the one you want"
    );
  });

  it("puts warnings in the same terms", () => {
    expect(message("M = 4\nN = 4\nA = Tensor(3, M)\nB = Tensor(N, 5)\nC = matmul(A, B)\n")).toMatch(
      /^matmul: the contracted axes \(x1 \(A\) axis 1, x2 \(B\) axis 0\) must be equal; M and N agree for M=4, N=4/
    );
  });

  it("leaves a graph built by tooling in the graph's terms", () => {
    const graph = G({ A: [4, 8], B: [4, 5] }, [["n", "matmul", ["A", "B"], ["C"]]]);
    expect(() => resolveGraph(graph)).toThrow(/^node "n" \(matmul\): shape inference failed: the contracted axis differs: axis 1 of input 0 is 8/);
  });
});
