/**
 * The DSL's signatures, checked in both directions off the table itself.
 *
 * Every function has one program here, and the coverage test fails for a
 * function added without one. Each program must compile, print back as a call
 * to the same function, print as a fixpoint, and recompile to the same
 * canonical attributes: what the DSL accepts it can write, and the reverse.
 */

import { describe, expect, it } from "vitest";
import { parseDSL, toDSL } from "../../src/parse/dsl";
import { compileDSL, tryCompileDSL } from "../../src/parse/compiler";
import { SIGNATURES, signatureFor, signatureLine } from "../../src/parse/signatures";
import { ELEMENTWISE_FNS } from "../../src/core/ops/elementwise";
import { resolveGraph } from "../../src/core/graph";

const X = "X = Tensor(4, 6)\n";
const XY = "X = Tensor(4, 6)\nY = Tensor(4, 6)\n";
const spatialInputs = (n: number) => {
  const dims = Array.from({ length: n }, () => 8).join(", ");
  const kernel = Array.from({ length: n }, () => 3).join(", ");
  return `X = Tensor(1, 2, ${dims})\nW = Tensor(3, 2, ${kernel})\n`;
};

const elementwisePrograms = Object.fromEntries(
  Object.entries(ELEMENTWISE_FNS).map(([fn, spec]) => [
    fn,
    spec.maxInputs === 1 ? `${X}Z = ${fn}(X)\n` : spec.maxInputs === 2 ? `${XY}Z = ${fn}(X, Y)\n` : `${XY}Z = ${fn}(X, Y, Y)\n`,
  ])
);

const reductionPrograms = Object.fromEntries(
  ["sum", "prod", "mean", "max", "min", "logsumexp"].map((fn) => [fn, `${X}Z = ${fn}(X, axis=1, keepdims=true)\n`])
);

const spatialPrograms = Object.fromEntries(
  [1, 2, 3].flatMap((n) => [
    [`conv${n}d`, `${spatialInputs(n)}Z = conv${n}d(X, W, stride=2, padding=1)\n`],
    [`max_pool${n}d`, `${spatialInputs(n)}Z = max_pool${n}d(X, 2)\n`],
    [`avg_pool${n}d`, `${spatialInputs(n)}Z = avg_pool${n}d(X, 3, stride=1, padding=1)\n`],
  ])
);

/** One program per function; every statement after the declarations is the call. */
const PROGRAMS: Record<string, string> = {
  einsum: 'A = Tensor(4, 6)\nB = Tensor(6, 5)\nZ = einsum("ij,jk->ik", A, B)\n',
  matmul: "A = Tensor(4, 6)\nB = Tensor(6, 5)\nZ = matmul(A, B)\n",
  linear: `${X}W = Tensor(5, 6)\nZ = linear(X, W)\n`,
  ...elementwisePrograms,
  ...reductionPrograms,
  softmax: `${X}Z = softmax(X, axis=-1)\n`,
  layer_norm: `${X}W = Tensor(6)\nB = Tensor(6)\nZ = layer_norm(X, W, B)\n`,
  rms_norm: `${X}W = Tensor(6)\nZ = rms_norm(X, W)\n`,
  reshape: `${X}Z = reshape(X, [6, 4])\n`,
  permute_dims: `${X}Z = permute_dims(X, [1, 0])\n`,
  broadcast_to: "X = Tensor(1, 6)\nZ = broadcast_to(X, [4, 6])\n",
  concat: `${X}Z = concat([X, X], axis=1)\n`,
  split: `${X}Z0, Z1 = split(X, 3, axis=1)\n`,
  pad: `${X}Z = pad(X, 1, mode=edge)\n`,
  ...spatialPrograms,
  cumulative_sum: `${X}Z = cumulative_sum(X, axis=1, reverse=true)\n`,
  take: "E = Tensor(10, 6)\nI = Tensor(3, dtype=int64)\nZ = take(E, I, axis=0)\n",
  astype: `${X}Z = astype(X, float16)\n`,
  identity: `${X}Z = identity(X)\n`,
  contiguous: `${X}Z = contiguous(X)\n`,
  opaque: `${X}Z = opaque(X, op="Foo", shapes=[[4, 6]])\n`,
};

const resolvedAttrs = (source: string) => compileDSL(source).resolved.topo.map((n) => n.attrs);

describe("every function is written back as itself", () => {
  it("has a program for every function, and a function for every program", () => {
    expect(Object.keys(PROGRAMS).sort()).toEqual(SIGNATURES.map((s) => s.name).sort());
  });

  it.each(SIGNATURES.map((sig) => [sig.name] as const))("%s", (name) => {
    const source = PROGRAMS[name];
    const result = tryCompileDSL(source);
    expect(result.ok, result.ok ? "" : result.diagnostics.map((d) => d.message).join("; ")).toBe(true);
    const printed = toDSL(parseDSL(source));
    expect(printed).toContain(`${name}(`);
    expect(toDSL(parseDSL(printed))).toBe(printed);
    expect(resolvedAttrs(printed)).toEqual(resolvedAttrs(source));
  });

  it("has no two functions of one operation claiming the same node", () => {
    const seen = new Map<string, string>();
    for (const sig of SIGNATURES) {
      const key = `${sig.op}:${JSON.stringify(sig.implied)}`;
      expect(seen.get(key), `${sig.name} and ${seen.get(key)}`).toBeUndefined();
      seen.set(key, sig.name);
    }
  });
});

describe("printing", () => {
  it.each(["sum", "prod", "mean", "max", "min", "logsumexp"])("round-trips resolved scalar %s reductions", (fn) => {
    const program = compileDSL(`X = Tensor(4, 6)\nY = ${fn}(X)\nZ = ${fn}(Y)\n`);
    expect(resolveGraph(program.resolved).tensors.Z.resolved).toEqual([]);
    const printed = toDSL(program.resolved);
    const roundTrip = compileDSL(printed);
    expect(roundTrip.resolved.tensors.Z.resolved).toEqual([]);
    expect(roundTrip.resolved.topo.map((n) => n.attrs)).toEqual(program.resolved.topo.map((n) => n.attrs));
    expect(toDSL(roundTrip.resolved)).toBe(printed);
  });

  it("writes tensors and leading parameters positionally, the rest by name", () => {
    const printed = toDSL(parseDSL(`${spatialInputs(2)}Z = conv2d(X, W, stride=[2, 2], padding=[[1, 1], [1, 1]])\n`));
    expect(printed).toContain("Z = conv2d(X, W, stride=2, padding=1)");
  });

  it("keeps attributes the name does not carry, and leaves out the ones it does", () => {
    const printed = toDSL(parseDSL(`${X}Z = sum(X, axis=1, keepdims=true)\n`));
    expect(printed).toContain("Z = sum(X, axis=1, keepdims=true)");
    expect(printed).not.toContain("fn=");
  });

  it("names a later optional input when an earlier one is left out", () => {
    const printed = toDSL(parseDSL(`${X}B = Tensor(6)\nZ = layer_norm(X, bias=B)\n`));
    expect(printed).toContain("Z = layer_norm(X, bias=B)");
  });

  it("keeps axis names and dimension expressions as written", () => {
    const source = "S = 8\nX = Tensor(b=2, s=S)\nZ = softmax(X, axis=s)\nY = X[:, 0:S/2]\n";
    const printed = toDSL(parseDSL(source));
    expect(printed).toContain("Z = softmax(X, axis=s)");
    expect(printed).toContain("Y = X[:, 0:S/2]");
  });

  it("writes dtypes in the DSL's spelling", () => {
    expect(toDSL(parseDSL(`${X}Z = astype(X, bfloat16)\n`))).toContain("Z = astype(X, bfloat16)");
  });
});

const diagnostics = (source: string) => {
  const result = tryCompileDSL(source);
  return result.ok ? [] : result.diagnostics;
};
const first = (source: string) => diagnostics(source)[0];

describe("binding", () => {
  it("rejects a huge scalar split before allocating its pieces", () => {
    const d = first("X = Tensor(10000000000)\nA, B = split(X, 1)\n");
    expect(d.code).toBe("SEM_INVALID_ATTRIBUTES");
    expect(d.message).toBe("split: sizes: declares 10000000000 outputs, got 2");
  });

  it("counts the shorter final split piece when checking outputs", () => {
    const source = "D = 5\nX = Tensor(12)\nA, B, C = split(X, D)\n";
    const program = compileDSL(source);
    expect(["A", "B", "C"].map((id) => program.resolved.tensors[id].resolved)).toEqual([[5], [5], [2]]);
    expect(program.resolved.topo[0].attrs.sizes).toEqual(["D", "D", 2]);
    expect(first(source.replace("A, B, C", "A, B")).message).toContain("declares 3 outputs, got 2");
  });

  it("accepts a parameter by position or by name, with the same result", () => {
    expect(resolvedAttrs(`${X}Z = reshape(X, [6, 4])\n`)).toEqual(resolvedAttrs(`${X}Z = reshape(X, shape=[6, 4])\n`));
  });

  it("refuses a keyword-only parameter given by position, and says which are", () => {
    const d = first(`${spatialInputs(2)}Z = conv2d(X, W, 2)\n`);
    expect(d.code).toBe("DSL_TOO_MANY_POSITIONAL");
    expect(d.message).toBe(
      "conv2d() takes 2 positional arguments (input, weight), got 3; " +
        "stride, padding, dilation, groups are written by name"
    );
    expect(d.span.start.column).toBe("Z = conv2d(X, W, ".length + 1);
  });

  it("refuses a positional argument after a keyword", () => {
    expect(first(`${X}Z = reshape(X, shape=[6, 4], X)\n`).code).toBe("DSL_POSITIONAL_AFTER_KEYWORD");
  });

  it("refuses a parameter given twice", () => {
    const d = first(`${X}Z = reshape(X, [6, 4], shape=[24])\n`);
    expect(d).toMatchObject({ code: "DSL_DUPLICATE_ARGUMENT", message: "reshape() got two values for shape" });
  });

  it("names a missing parameter, with the signature", () => {
    expect(first(`${X}Z = softmax(X)\n`)).toMatchObject({
      code: "DSL_MISSING_ARGUMENT",
      message: "softmax() is missing axis: softmax(x, *, axis)",
    });
    expect(first("A = Tensor(4, 4)\nZ = matmul(A)\n").message).toBe("matmul() is missing x2: matmul(x1, x2)");
  });

  it("suggests the keyword another library uses for the same thing", () => {
    expect(first(`${X}Z = sum(X, dim=1)\n`).message).toMatch(/^sum\(\) has no parameter "dim" \(did you mean axis\?\)/);
    expect(first(`${spatialInputs(2)}Z = conv2d(X, W, pads=1)\n`).message).toMatch(/did you mean padding\?/);
    expect(first(`${X}Z = pad(X, pads=1)\n`).message).toMatch(/did you mean pad_width\?/);
    expect(first(`${X}Z = sum(X, keepdim=true)\n`).message).toMatch(/did you mean keepdims\?/);
  });

  it("underlines the keyword it does not know", () => {
    const d = first(`${X}Z = sum(X, dim=1)\n`);
    expect(d.code).toBe("DSL_UNKNOWN_KEYWORD");
    expect(d.span.start.column).toBe("Z = sum(X, ".length + 1);
    expect(d.span.end.column - d.span.start.column).toBe(3);
  });

  it("refuses a value spelled for another kind", () => {
    expect(first(`${X}Z = astype(X, "float16")\n`).message).toBe(
      "dtype: expected one of float32, float16, bfloat16, float8, int64, int32, uint8, int8, bool"
    );
    expect(first(`${X}Z = softmax(X, axis="s")\n`).message).toBe("axis: quotes are for free text; write s without them");
    expect(first(`${X}Z = pad(X, 1, mode="edge")\n`).message).toBe("mode: write edge without quotes");
    expect(first("A = Tensor(4, 6)\nZ = einsum(ij->i, A)\n").code).toBeDefined();
  });

  it("checks a value against its kind, at the value", () => {
    const d = first(`${spatialInputs(2)}Z = conv2d(X, W, stride=0)\n`);
    expect(d).toMatchObject({ code: "DSL_INVALID_ARGUMENT", message: "stride: expected an int ≥ 1, or one per spatial axis" });
    expect(d.span.start.column).toBe("Z = conv2d(X, W, stride=".length + 1);
  });

  it("suggests the function for another library's name, or a near spelling", () => {
    expect(first(`${X}Z = transpose(X, [1, 0])\n`)).toMatchObject({
      code: "DSL_UNKNOWN_FUNCTION",
      message: 'unknown function "transpose" (did you mean permute_dims?)',
    });
    expect(first(`${X}Z = sofmax(X, axis=1)\n`).message).toBe('unknown function "sofmax" (did you mean softmax?)');
    expect(first(`${X}Z = slice(X)\n`).message).toBe('unknown function "slice" (did you mean a subscript: X[a:b, :, ::2]?)');
  });

  it("reads an axis by the name its operand carries", () => {
    const program = compileDSL("X = Tensor(b=2, s=8, e=4)\nZ = sum(X, axis=[s, e])\n");
    expect(program.resolved.topo[0].attrs).toMatchObject({ axes: [1, 2] });
    expect(program.resolved.tensors.Z.resolved).toEqual([2]);
  });

  it("reduces over every axis when no axis is written", () => {
    expect(compileDSL(`${X}Z = sum(X)\n`).resolved.tensors.Z.resolved).toEqual([]);
  });
});

describe("subscripts", () => {
  const shapeOf = (source: string) => compileDSL(source).resolved.tensors.Z.resolved;

  it("slices with NumPy's rules: omitted parts, steps, negative bounds, fewer items than axes", () => {
    expect(shapeOf(`${X}Z = X[1:3, ::2]\n`)).toEqual([2, 3]);
    expect(shapeOf(`${X}Z = X[-2:]\n`)).toEqual([2, 6]);
    expect(shapeOf(`${X}Z = X[:, :-1]\n`)).toEqual([4, 5]);
    expect(shapeOf(`${X}Z = X[0:100]\n`)).toEqual([4, 6]);
  });

  it("takes bounds as dimension expressions", () => {
    expect(shapeOf("S = 8\nX = Tensor(2, S)\nZ = X[:, S/4:S/2]\n")).toEqual([2, 2]);
  });

  it.each([
    ["-S:", [2]],
    ["-(2):", [2]],
    ["S-S:S", [2]],
    [":S-S", [0]],
    [":-S", [6]],
    ["-S*10:S*10", [8]],
    ["::S", [4]],
  ])("evaluates signed bounds in %s", (subscript, expected) => {
    expect(shapeOf(`S = 2\nX = Tensor(8)\nZ = X[${subscript}]\n`)).toEqual(expected);
  });

  it.each(["S-S", "-S", "S/3", "S/(S-S)", "S*9007199254740991", "Missing"])("rejects invalid step expression %s", (step) => {
    expect(first(`S = 2\nX = Tensor(8)\nZ = X[::${step}]\n`).code).toBe("SEM_INVALID_ATTRIBUTES");
  });

  it.each(["S-S", "-S"])("still requires positive authored extents: %s", (extent) => {
    expect(first(`S = 2\nX = Tensor(${extent})\n`).code).toBe("SEM_SHAPE");
  });

  it("refuses an index, naming the range that keeps the axis", () => {
    expect(first(`${X}Z = X[1, :]\n`)).toMatchObject({
      code: "DSL_UNSUPPORTED_INDEX",
      message: "an index would drop the axis, which slicing here does not do; write 1:1+1 to keep it",
    });
  });

  it("refuses more items than the tensor has axes, at the brackets", () => {
    const d = first(`${X}Z = X[:, :, :]\n`);
    expect(d.code).toBe("SEM_INVALID_ATTRIBUTES");
    expect(d.message).toContain("has 3 entries for 2 axes");
    expect(d.span.start.column).toBe("Z = X".length + 1);
  });

  it("reads back what it wrote", () => {
    const printed = toDSL(parseDSL(`${X}Z = X[1:3, ::2]\n`));
    expect(printed).toContain("Z = X[1:3, ::2]");
  });
});

describe("signature lines", () => {
  it("reads like a Python signature", () => {
    expect(signatureLine(signatureFor("conv2d")!)).toBe(
      "conv2d(input, weight, *, stride=…, padding=…, dilation=…, groups=…)"
    );
    expect(signatureLine(signatureFor("layer_norm")!)).toBe("layer_norm(x, weight=None, bias=None, *, axis=…)");
    expect(signatureLine(signatureFor("einsum")!)).toBe("einsum(subscripts, *operands)");
    expect(signatureLine(signatureFor("concat")!)).toBe("concat(arrays, *, axis=…)");
    expect(signatureLine(signatureFor("add")!)).toBe("add(x1, x2, *more)");
  });
});
