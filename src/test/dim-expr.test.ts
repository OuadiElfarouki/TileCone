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
  const heads = "params B=1 H=4 S=128 D=32\n";

  it("resolves a product of parameters", () => {
    expect(shapeOf(heads + "input X [B, S, H*D] f16\n", "X")).toEqual([1, 128, 128]);
  });

  it("keeps the written form in the IR rather than a precomputed literal", () => {
    // The point of the feature: the relationship stays visible to `toDSL`, to
    // share links, and to the declared dimensions notes.ts reads.
    expect(parseDSL(heads + "input X [B, S, H*D] f16\n").tensors.X.shape).toEqual([
      "B",
      "S",
      "H*D",
    ]);
  });

  it("round-trips through toDSL alongside axis names", () => {
    const source = heads + "input X [batch: B, seq: S, emb: H*D] f16\n";
    expect(toDSL(parseDSL(source))).toBe(source);
  });

  it("honours precedence and parentheses", () => {
    expect(shapeOf(heads + "input X [H+D*2, (H+D)*2] f16\n", "X")).toEqual([68, 72]);
  });

  it("reads arithmetic in a shape-valued attribute", () => {
    const src =
      heads +
      "input X [B, S, H*D] f16\nY = reshape(X, shape=[B, S, H, (H*D)/H])\n";
    expect(shapeOf(src, "Y")).toEqual([1, 128, 4, 32]);
  });

  it("refuses a division that is not a whole number of elements", () => {
    const diagnostic = failure("params H=5 E=32\ninput X [E/H] f32\nY = relu(X)\n");
    expect(diagnostic.message).toContain("32/5 is not a whole number of elements");
  });

  it("rejects unsafe intermediate arithmetic before rounding can hide it", () => {
    const diagnostic = failure(
      "params N=9007199254740991\ninput X [(N+2)-2] f32\nY = relu(X)\n"
    );
    expect(diagnostic.message).toContain("outside the safe integer range");
  });

  it("names the unbound symbol inside an expression", () => {
    const diagnostic = failure("params H=4\ninput X [H*D] f32\nY = relu(X)\n");
    expect(diagnostic.message).toContain('unbound symbolic dim "D"');
  });

  it("still reads a lone literal as a number", () => {
    const graph = parseDSL("input X [4, 1e2] f32\n");
    expect(graph.tensors.X.shape).toEqual([4, 100]);
  });

  it("still rejects a negative literal dimension", () => {
    expect(failure("input X [-1, 8] f32\nY = relu(X)\n").message).toContain("bad dimension -1");
  });

  it("leaves scalar attributes alone", () => {
    const attrs = parseDSL("input X [2, 4] f32\nY = sum(X, axes=[-1], keepdim=true)\n").nodes[0]
      .attrs;
    expect(attrs).toMatchObject({ axes: [-1], keepdim: true, fn: "sum" });
  });
});
