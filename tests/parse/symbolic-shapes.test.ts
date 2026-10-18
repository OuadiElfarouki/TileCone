import { describe, expect, it } from "vitest";
import { compileDSL, tryCompileDSL } from "../../src/parse/compiler";
import { EXAMPLES } from "../../src/examples";

/** The first warning of a program that built. */
const warning = (src: string) => {
  const result = tryCompileDSL(src);
  if (!result.ok) throw new Error(`expected the source to build: ${result.diagnostics[0].message}`);
  const [first] = result.diagnostics;
  if (!first) throw new Error("expected a warning");
  expect(first.severity).toBe("warning");
  return first;
};

const underlined = (src: string) => {
  const { span } = warning(src);
  return src.slice(span.start.offset, span.end.offset);
};

/** Builds, with nothing to warn about. */
const accepts = (src: string) => {
  const result = tryCompileDSL(src);
  expect(result.ok).toBe(true);
  expect(result.diagnostics).toEqual([]);
};

describe("extents that agree only for the current bindings", () => {
  const flattened = `M = 4
N = 4
K = 38
X = Tensor(M, N)
F = reshape(X, shape=[M*N])
Y = reshape(F, shape=[N/2, M*2])
`;

  it("builds a contraction whose sides are equal only because M == N, and warns", () => {
    const src = flattened + "A = Tensor(N*2, K)\nZ = matmul(Y, A)\n";
    const result = tryCompileDSL(src);
    expect(result.ok && result.program.resolved.tensors.Z.resolved).toEqual([2, 38]);
    const diagnostic = warning(src);
    expect(diagnostic.code).toBe("SEM_SYMBOLIC_SHAPE");
    expect(diagnostic.message).toContain("M*2 and N*2 agree for M=4, N=4 but are different expressions");
    expect(diagnostic.message).toContain("one symbol, or a derived one");
    expect(underlined(src)).toBe("A");
  });

  it("reports an error, not a warning, once the bindings differ", () => {
    const result = tryCompileDSL(flattened.replace("N = 4", "N = 8") + "A = Tensor(N*2, K)\nZ = matmul(Y, A)\n");
    expect(result.ok).toBe(false);
    expect(result.diagnostics.every((d) => d.severity === "error")).toBe(true);
  });

  it("accepts the same contraction once both sides are one expression", () => {
    accepts(flattened + "A = Tensor(M*2, K)\nZ = matmul(Y, A)\n");
  });

  it("checks a reshape's element count as an expression", () => {
    const src = "M = 4\nN = 4\nX = Tensor(M, N)\nY = reshape(X, shape=[N*N])\n";
    expect(warning(src).message).toContain("M*N and N*N agree");
    expect(underlined(src)).toBe("shape=[N*N]");
    accepts("H = 4\nD = 8\nX = Tensor(H*D, 3)\nY = reshape(X, shape=[(H*D)/H, H*3])\n");
  });

  it("treats a parameter matching a literal as a coincidence", () => {
    const src = "M = 8\nK = 8\nA = Tensor(M, K)\nB = Tensor(8, 3)\nC = matmul(A, B)\n";
    expect(warning(src).message).toContain("K and 8 agree for K=8");
  });

  it("names the label of a written einsum", () => {
    const src = 'M = 4\nN = 4\nA = Tensor(M, 3)\nB = Tensor(N, 5)\nC = einsum("ij,ik->jk", A, B)\n';
    expect(warning(src).message).toContain('the contracted axes "i"');
  });

  it("broadcasts against a written 1, not against a parameter bound to 1", () => {
    accepts("M = 4\nA = Tensor(M, 3)\nB = Tensor(1, 3)\nC = add(A, B)\n");
    const src = "M = 4\nN = 1\nA = Tensor(M)\nB = Tensor(N)\nC = add(A, B)\n";
    expect(warning(src).message).toContain("M and N agree for M=4, N=1");
    expect(underlined(src)).toBe("B");
  });

  it("checks split sizes, concat axes and normalization weights", () => {
    expect(
      warning("H = 4\nD = 2\nX = Tensor(H*D)\nA, B = split(X, axis=0, sizes=[D*2, H])\n").message
    ).toContain("D*2+H and H*D agree");
    expect(
      warning("R = 4\nC = 4\nA = Tensor(R, 2)\nB = Tensor(C, 3)\nY = concat(A, B, axis=1)\n").message
    ).toContain("R and C agree");
    expect(
      warning("S = 4\nE = 8\nF = 8\nX = Tensor(S, E)\nW = Tensor(F)\nY = layernorm(X, W)\n").message
    ).toContain("E and F agree");
  });

  it("takes a literal reshape target as written, not the input's expression", () => {
    const src = "M = 2\nX = Tensor(M, 8/M)\nY = reshape(X, shape=[2,4])\nA = Tensor(4,3)\nZ = matmul(Y,A)\n";
    accepts(src);
    expect(compileDSL(src).resolved.tensors.Y.symShape).toEqual([2, 4]);
  });

  it("compares every pair of operands, so an unknown first one hides nothing", () => {
    const unknown = "X0 = Tensor(8, 2)\nX = slice(X0, starts=[0, 0], stops=[4, 2], steps=[1, 1])\n";
    expect(warning(`M = 4\nN = 4\n${unknown}A = Tensor(M, 2)\nB = Tensor(N, 2)\nY = einsum("ij,ij,ij->ij", X, A, B)\n`).message)
      .toContain("M and N agree");
    expect(warning(`M = 4\nN = 4\n${unknown}A = Tensor(M, 3)\nB = Tensor(N, 5)\nY = concat(X, A, B, axis=1)\n`).message)
      .toContain("M and N agree");
  });

  it("compiles a long chain of derived symbols promptly", () => {
    const chain = Array.from({ length: 200 }, (_, i) => `D${i + 1} = D${i}+D${i}-D${i}`).join("\n");
    const started = performance.now();
    accepts(`D0 = 4\n${chain}\nX = Tensor(D200, 2)\nY = reshape(X, shape=[D0*2])\n`);
    expect(performance.now() - started).toBeLessThan(2000);
  });

  it("carries a symbolic split size downstream", () => {
    const { resolved } = compileDSL("H = 4\nD = 2\nX = Tensor(H*D+D)\nA, B = split(X, axis=0, sizes=[H*D, D])\n");
    expect(resolved.tensors.A.symShape).toEqual(["H*D"]);
    expect(resolved.tensors.B.symShape).toEqual(["D"]);
  });
});

describe("extents with no expression", () => {
  it("does not refute a reshape over a convolution's derived spatial axes", () => {
    // The 8 after the convolution is H/2, but no expression says so; treating
    // it as a literal would call `C*H/2*H/2` a mismatch.
    accepts(`N = 1
C = 4
H = 16
X = Tensor(N, C, H, H)
W = Tensor(C, C, 3, 3)
Y = conv(X, W, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1])
Z = reshape(Y, shape=[N, C*H/2*H/2])
`);
  });

  it("still checks the convolution's channels", () => {
    const src = `N = 1
C = 4
G = 4
X = Tensor(N, C, 8, 8)
W = Tensor(C, G, 3, 3)
Y = conv(X, W, stride=[1, 1], pads=[[1, 1], [1, 1]], dilation=[1, 1])
`;
    expect(warning(src).message).toContain("C and G agree");
  });

  it("does not refute what a barrier produced", () => {
    accepts('P = 2\nQ = 4\nX = opaque(op="Src", shapes=[[8]], dtypes=[fp32])\nY = reshape(X, shape=[P, Q])\n');
  });

  it("keeps the canonical display of a lost axis", () => {
    const { resolved } = compileDSL(`N = 1
C = 4
H = 16
X = Tensor(N, C, H, H)
W = Tensor(C, C, 3, 3)
Y = conv(X, W, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1])
`);
    expect(resolved.tensors.Y.symShape).toEqual(["N", "C", 8, 8]);
  });
});

describe("built-in examples", () => {
  it.each(EXAMPLES.map((example) => [example.name, example.dsl]))(
    "%s holds for every binding",
    (_name, dsl) => accepts(dsl)
  );
});
