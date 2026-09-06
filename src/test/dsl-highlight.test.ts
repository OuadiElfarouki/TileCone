import { describe, expect, it } from "vitest";
import { highlightDSL } from "../ui/dsl-highlight";

describe("DSL source highlighting", () => {
  it("preserves source text byte for byte", () => {
    const source = `M = 8\nX = Tensor(M, dtype=fp16)\nY = relu(X) # selected output\n`;
    expect(highlightDSL(source).map((token) => token.text).join("")).toBe(source);
  });

  it("highlights constructors, dtypes, literals, and operation calls", () => {
    const source = `M = 8\nW = Parameter(M, dtype=bf16)\nY = matmul(W, W, keepdim=true)\n`;
    const keywords = highlightDSL(source)
      .filter((token) => token.kind === "keyword")
      .map((token) => token.text);
    expect(keywords).toEqual(["Parameter", "bf16", "matmul", "true"]);
  });

  it("starts comments only at hashes outside strings", () => {
    const source = `Y = einsum("b#d,bde->b#e", X, W) # real comment\n# whole line\n`;
    const comments = highlightDSL(source)
      .filter((token) => token.kind === "comment")
      .map((token) => token.text);
    expect(comments).toEqual(["# real comment", "# whole line"]);
  });

  it("does not let an unfinished string swallow later lines", () => {
    const source = `Y = einsum("unfinished\n# still a comment\n`;
    expect(highlightDSL(source).filter((token) => token.kind === "comment")).toEqual([
      { kind: "comment", text: "# still a comment" },
    ]);
  });
});
