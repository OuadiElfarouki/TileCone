/**
 * Multi-error recovery and argument-level spans.
 *
 * The old front end reported exactly one diagnostic per compile and underlined
 * whole statements, so a document with four independent mistakes took four
 * edit-compile cycles and never said which part of a line was wrong. These
 * pin both behaviours.
 */

import { describe, expect, it } from "vitest";
import { tryCompileDSL } from "../parse/compiler";
import { parseProgram } from "../parse/parser";
import { lowerProgram } from "../parse/lower";

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
E = sum(C)
F = conv(A, B, stride=[1])
`);
    expect(diagnostics.map((d) => d.span.start.line)).toEqual([3, 4, 5, 6]);
    expect(diagnostics.map((d) => d.code)).toEqual([
      "SEM_SHAPE",
      "DSL_UNKNOWN_TENSOR",
      "DSL_MISSING_ATTRIBUTE",
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

  it("underlines the misspelled attribute", () => {
    expect(underlined("X = Tensor(4, 5)\nY = reduce(X, fn=sum, axes=[1], keepdims=true)\n")).toBe(
      "keepdims=true"
    );
  });

  it("underlines the bad dtype value in a declaration", () => {
    expect(underlined("X = Tensor(4, dtype=float64)\n")).toBe("float64");
  });

  it("underlines the call name for an unknown op", () => {
    expect(underlined("X = Tensor(4)\nY = nosuchop(X)\n")).toBe("nosuchop");
  });

  it("falls back to the statement when nothing narrower is known", () => {
    // A shape mismatch is about the pairing of two operands, not either one.
    expect(underlined("A = Tensor(4, 5)\nB = Tensor(9, 7)\nC = matmul(A, B)\n")).toBe(
      "C = matmul(A, B)"
    );
  });

  it("records a span for every input and named attribute", () => {
    const { program } = parseProgram("X = Tensor(4, 5)\nY = reduce(X, fn=sum, axes=[1])\n");
    const { sourceMap } = lowerProgram(program);
    const args = sourceMap.nodeArgs.reduce_Y;
    expect(args.inputs).toHaveLength(1);
    expect(Object.keys(args.attrs).sort()).toEqual(["axes", "fn"]);
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
