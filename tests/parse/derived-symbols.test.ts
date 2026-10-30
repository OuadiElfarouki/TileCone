import { describe, expect, it } from "vitest";
import { resolveGraph } from "../../src/core/graph";
import { compileDSL, tryCompileDSL } from "../../src/parse/compiler";
import { parseDSL, toDSL } from "../../src/parse/dsl";
import { graphToJSON, parseGraphJSON } from "../../src/parse/json";
import { sameParameterizedSource, withParamNamed } from "../../src/view/params";

const diagnostics = (src: string) => {
  const result = tryCompileDSL(src);
  if (result.ok) throw new Error("expected the source to be rejected");
  return result.diagnostics;
};

const heads = `B = 1
H = 4
S = 8
D = 2
E = H*D
X = Tensor(B, S, E, dtype=float16)
W = Parameter(E, E, dtype=float16)
Q = einsum("bse,ef->bsf", X, W)
Q4 = reshape(Q, [B, S, H, D])
Z = reshape(Q4, [B, S, E])
`;

describe("derived symbols", () => {
  it("sizes dimensions and keeps its own name in symbolic shapes", () => {
    const { resolved } = compileDSL(heads);
    expect(resolved.tensors.X.resolved).toEqual([1, 8, 8]);
    expect(resolved.tensors.X.symShape).toEqual(["B", "S", "E"]);
    expect(resolved.tensors.Z.symShape).toEqual(["B", "S", "E"]);
  });

  it("is not a parameter, but counts the parameters it is written over as used", () => {
    const { resolved } = compileDSL("H = 4\nD = 2\nUnused = 3\nE = H*D\nX = Tensor(E)\n");
    expect(Object.keys(resolved.params)).toEqual(["H", "D", "Unused"]);
    expect(resolved.shapeParameters).toEqual(["H", "D"]);
    expect(resolved.derived).toEqual({ E: "H*D" });
  });

  it("is the same expression as its definition", () => {
    // E against B*S*H*D is the reshape's element count check.
    expect(tryCompileDSL(heads).ok).toBe(true);
    const result = tryCompileDSL("M = 4\nN = 4\nE = M*N\nX = Tensor(E)\nY = reshape(X, [N*N])\n");
    expect(result.ok).toBe(true);
    expect(result.diagnostics[0].code).toBe("SEM_SYMBOLIC_SHAPE");
    expect(result.diagnostics[0].message).toContain("agree for M=4, N=4, where E = M*N");
  });

  it("chains over earlier derived symbols", () => {
    const { resolved } = compileDSL(
      "N = 8\nHALF = N/2\nQ = HALF/2\nX = Tensor(N)\nA, B = split(X, [HALF, Q*2], axis=0)\n"
    );
    expect(resolved.tensors.A.resolved).toEqual([4]);
    expect(resolved.shapeParameters).toEqual(["N"]);
  });

  it("is a broadcastable 1 when its definition is", () => {
    expect(tryCompileDSL("M = 4\nONE = M/M\nA = Tensor(M)\nB = Tensor(ONE)\nC = add(A, B)\n").ok).toBe(true);
  });

  it("reads only names defined above it, and reports that once", () => {
    const found = diagnostics("E = M*2\nM = 4\nX = Tensor(E)\nY = relu(X)\n");
    expect(found).toHaveLength(1);
    expect(found[0].message).toContain('unknown parameter "M"');
    expect(found[0].span.start.line).toBe(1);
  });

  it("is written over size parameters, not tensors", () => {
    expect(diagnostics("X = Tensor(4)\nE = X*2\n")[0].message).toContain('"X" is a tensor');
  });

  it("must come out a whole, positive extent", () => {
    const [diagnostic] = diagnostics("N = 5\nHALF = N/2\nX = Tensor(HALF)\n");
    expect(diagnostic.message).toContain("5/2 is not a whole number of elements");
    expect(diagnostic.span.start.line).toBe(2);
    expect(diagnostics("N = 4\nZERO = N-N\nX = Tensor(ZERO)\n")[0].message).toContain("resolves to 0");
  });

  it("cannot redefine a name", () => {
    expect(diagnostics("M = 4\nM = M*2\n")[0].message).toContain('symbol "M" redefined');
  });

  it("round-trips through toDSL and JSON", () => {
    const graph = parseDSL(heads);
    expect(toDSL(graph)).toBe(toDSL(parseDSL(toDSL(graph))));
    expect(toDSL(graph)).toContain("E = H*D\n");
    const fromJSON = parseGraphJSON(graphToJSON(graph));
    expect(fromJSON.derived).toEqual({ E: "H*D" });
    expect(resolveGraph(fromJSON).tensors.X.resolved).toEqual([1, 8, 8]);
  });

  it("rejects a cyclic or colliding definition in a graph built without the DSL", () => {
    const graph = parseDSL("H = 4\nX = Tensor(H)\n");
    expect(() => resolveGraph({ ...graph, derived: { A: "B", B: "A" } })).toThrow(/unbound symbolic dim "B"/);
    expect(() => resolveGraph({ ...graph, derived: { H: 2 } })).toThrow(/already defined/);
  });

  it("follows a parameter control's edit without changing the source's identity", () => {
    const edited = withParamNamed(heads, "H", 8)!;
    expect(edited).toContain("H = 8\n");
    expect(edited).toContain("E = H*D\n");
    expect(sameParameterizedSource(heads, edited)).toBe(true);
    expect(compileDSL(edited).resolved.tensors.X.resolved).toEqual([1, 8, 16]);
  });
});
