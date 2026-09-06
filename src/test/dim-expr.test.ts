import { describe, expect, it } from "vitest";
import { compileDSL, tryCompileDSL } from "../parse/compiler";
import { parseDSL, toDSL } from "../parse/dsl";

const shapeOf = (src: string, tensor: string) =>
  compileDSL(src).resolved.tensors[tensor].resolved;

const failure = (src: string) => {
  const result = tryCompileDSL(src);
  if (result.ok) throw new Error("expected the source to be rejected");
  return result.diagnostics[0];
};

describe("dimensions written as arithmetic", () => {
  const heads = "B = 1\nH = 4\nS = 128\nD = 32\n";

  it("resolves a product of parameters", () => {
    expect(shapeOf(heads + "X = Tensor(B, S, H*D, dtype=fp16)\n", "X")).toEqual([1, 128, 128]);
  });

  it("keeps the written form in the IR rather than a precomputed literal", () => {
    // The point of the feature: the relationship stays visible to `toDSL`, to
    // share links, and to the declared dimensions notes.ts reads.
    expect(parseDSL(heads + "X = Tensor(B, S, H*D, dtype=fp16)\n").tensors.X.shape).toEqual([
      "B",
      "S",
      "H*D",
    ]);
  });

  it("round-trips through toDSL alongside axis names", () => {
    const source = heads + "X = Tensor(batch=B, seq=S, emb=H*D, dtype=fp16)\n";
    expect(toDSL(parseDSL(source))).toBe(source);
  });

  it("honours precedence and parentheses", () => {
    expect(shapeOf(heads + "X = Tensor(H+D*2, (H+D)*2, dtype=fp16)\n", "X")).toEqual([68, 72]);
  });

  it("reads arithmetic in a shape-valued attribute", () => {
    const src =
      heads +
      "X = Tensor(B, S, H*D, dtype=fp16)\nY = reshape(X, shape=[B, S, H, (H*D)/H])\n";
    expect(shapeOf(src, "Y")).toEqual([1, 128, 4, 32]);
  });

  it("refuses a division that is not a whole number of elements", () => {
    const diagnostic = failure("H = 5\nE = 32\nX = Tensor(E/H, dtype=fp32)\nY = relu(X)\n");
    expect(diagnostic.message).toContain("32/5 is not a whole number of elements");
  });

  it("rejects unsafe intermediate arithmetic before rounding can hide it", () => {
    const diagnostic = failure(
      "N = 9007199254740991\nX = Tensor((N+2)-2, dtype=fp32)\nY = relu(X)\n"
    );
    expect(diagnostic.message).toContain("outside the safe integer range");
  });

  it("names the unbound symbol inside an expression", () => {
    const diagnostic = failure("H = 4\nX = Tensor(H*D, dtype=fp32)\nY = relu(X)\n");
    expect(diagnostic.message).toContain('unbound symbolic dim "D"');
  });

  it("still reads a lone literal as a number", () => {
    const graph = parseDSL("X = Tensor(4, 1e2, dtype=fp32)\n");
    expect(graph.tensors.X.shape).toEqual([4, 100]);
  });

  it("still rejects a negative literal dimension", () => {
    expect(failure("X = Tensor(-1, 8, dtype=fp32)\nY = relu(X)\n").message).toContain("bad dimension -1");
  });

  /* An empty axis was accepted where it was written as a literal or fell out of
     arithmetic, while `M = 0` was already refused as a parameter binding and a
     reshape target of 0 was already refused by its schema. One rule now: an
     authored dimension is at least 1, wherever it is written. */
  it("rejects a zero dimension on the same terms as a negative one", () => {
    expect(failure("X = Tensor(0, 8, dtype=fp32)\nY = relu(X)\n").message).toContain(
      "bad dimension 0"
    );
    expect(failure("E = 4\nX = Tensor(E-4, dtype=fp32)\n").message).toContain(
      'dimension "E-4" resolves to 0'
    );
    expect(failure("M = 0\nX = Tensor(M, dtype=fp32)\n").message).toContain("bad binding M=0");
    expect(shapeOf("X = Tensor(1, 8, dtype=fp32)\n", "X")).toEqual([1, 8]);
  });

  it("leaves scalar attributes alone", () => {
    const attrs = parseDSL("X = Tensor(2, 4, dtype=fp32)\nY = sum(X, axes=[-1], keepdim=true)\n").nodes[0]
      .attrs;
    expect(attrs).toMatchObject({ axes: [-1], keepdim: true, fn: "sum" });
  });
});
