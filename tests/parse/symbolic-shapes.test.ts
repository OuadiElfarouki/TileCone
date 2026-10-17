import { describe, expect, it } from "vitest";
import { compileDSL, tryCompileDSL } from "../../src/parse/compiler";
import { EXAMPLES } from "../../src/examples";

const rejection = (src: string) => {
  const result = tryCompileDSL(src);
  if (result.ok) throw new Error("expected the source to be rejected");
  return result.diagnostics[0];
};

const underlined = (src: string) => {
  const { span } = rejection(src);
  return src.slice(span.start.offset, span.end.offset);
};

const accepts = (src: string) => expect(tryCompileDSL(src).ok).toBe(true);

describe("extents that agree only for the current bindings", () => {
  const flattened = `M = 4
N = 4
K = 38
X = Tensor(M, N)
F = reshape(X, shape=[M*N])
Y = reshape(F, shape=[N/2, M*2])
`;

  it("rejects a contraction whose sides are equal only because M == N", () => {
    const src = flattened + "A = Tensor(N*2, K)\nZ = matmul(Y, A)\n";
    const diagnostic = rejection(src);
    expect(diagnostic.code).toBe("SEM_SYMBOLIC_SHAPE");
    expect(diagnostic.message).toContain("M*2 and N*2 are different expressions");
    expect(diagnostic.message).toContain("holds only for M=4, N=4");
    expect(underlined(src)).toBe("A");
  });

  it("accepts the same contraction once both sides are one expression", () => {
    accepts(flattened + "A = Tensor(M*2, K)\nZ = matmul(Y, A)\n");
  });

  it("checks a reshape's element count as an expression", () => {
    const src = "M = 4\nN = 4\nX = Tensor(M, N)\nY = reshape(X, shape=[N*N])\n";
    expect(rejection(src).message).toContain("M*N and N*N");
    expect(underlined(src)).toBe("shape=[N*N]");
    accepts("H = 4\nD = 8\nX = Tensor(H*D, 3)\nY = reshape(X, shape=[(H*D)/H, H*3])\n");
  });

  it("treats a parameter matching a literal as a coincidence", () => {
    const src = "M = 8\nK = 8\nA = Tensor(M, K)\nB = Tensor(8, 3)\nC = matmul(A, B)\n";
    expect(rejection(src).message).toContain("K and 8 are different expressions");
  });

  it("names the label of a written einsum", () => {
    const src = 'M = 4\nN = 4\nA = Tensor(M, 3)\nB = Tensor(N, 5)\nC = einsum("ij,ik->jk", A, B)\n';
    expect(rejection(src).message).toContain('the contracted axes "i"');
  });

  it("broadcasts against a written 1, not against a parameter bound to 1", () => {
    accepts("M = 4\nA = Tensor(M, 3)\nB = Tensor(1, 3)\nC = add(A, B)\n");
    const src = "M = 4\nN = 1\nA = Tensor(M)\nB = Tensor(N)\nC = add(A, B)\n";
    expect(rejection(src).message).toContain("M and N are different expressions");
    expect(underlined(src)).toBe("B");
  });

  it("checks split sizes, concat axes and normalization weights", () => {
    expect(
      rejection("H = 4\nD = 2\nX = Tensor(H*D)\nA, B = split(X, axis=0, sizes=[D*2, H])\n").message
    ).toContain("D*2+H and H*D");
    expect(
      rejection("R = 4\nC = 4\nA = Tensor(R, 2)\nB = Tensor(C, 3)\nY = concat(A, B, axis=1)\n").message
    ).toContain("R and C");
    expect(
      rejection("S = 4\nE = 8\nF = 8\nX = Tensor(S, E)\nW = Tensor(F)\nY = layernorm(X, W)\n").message
    ).toContain("E and F");
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
    expect(rejection(src).message).toContain("C and G");
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
