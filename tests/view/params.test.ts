import * as parser from "../../src/parse/parser";
import { describe, expect, it, vi } from "vitest";
import {
  parameterSourceKey,
  sameParameterizedSource,
  sourceParams,
  STRIDE,
  stepModeOf,
  stepValue,
  withParamNamed,
  withParamValue,
} from "../../src/view/params";
import { overlayTokens, withHints } from "../../src/view/dsl-highlight";

const source = `M = 256
N = 384  # not a power of two
K = 512

A = Tensor(M, K, dtype=float16)
`;

describe("size parameters in the draft", () => {
  it("reads each parameter with the span of its number and the end of its line", () => {
    const params = sourceParams(source);
    expect(params.map((p) => [p.name, p.value])).toEqual([["M", 256], ["N", 384], ["K", 512]]);
    const n = params[1];
    expect(source.slice(n.start, n.end)).toBe("384");
    expect(source.slice(n.start, n.lineEnd)).toBe("384  # not a power of two");
  });

  it("rewrites one number and leaves the rest of the text as written", () => {
    const n = sourceParams(source)[1];
    const next = withParamValue(source, n, 1024);
    expect(next).toBe(source.replace("N = 384", "N = 1024"));
    expect(sourceParams(next).map((p) => p.value)).toEqual([256, 1024, 512]);
  });

  it("offers no slider for a value that is not a positive integer", () => {
    expect(sourceParams("M = 0\nX = Tensor(4)\n")).toEqual([]);
  });

  it("rewrites a parameter found by name, and reports one the draft no longer has", () => {
    expect(withParamNamed(source, "K", 64)).toBe(source.replace("K = 512", "K = 64"));
    expect(withParamNamed(source, "Q", 64)).toBeNull();
  });
});

describe("stepping a parameter", () => {
  it("reads the mode from the value: powers of two, other evens, odds", () => {
    expect(stepModeOf(1)).toBe("double");
    expect(stepModeOf(256)).toBe("double");
    expect(stepModeOf(384)).toBe("stride");
    expect(stepModeOf(77)).toBe("unit");
  });

  it("doubles and halves a power of two", () => {
    expect(stepValue(256, "double", 1)).toBe(512);
    expect(stepValue(256, "double", -1)).toBe(128);
    expect(stepValue(1, "double", -1)).toBe(1);
  });

  it("distinguishes large integers from nearby powers of two", () => {
    expect(stepModeOf(2 ** 49 - 1)).toBe("unit");
    expect(stepModeOf(2 ** 50 + 2)).toBe("stride");
    expect(stepModeOf(2 ** 50)).toBe("double");
    expect(stepModeOf(Number.MAX_SAFE_INTEGER)).toBe("unit");
  });

  it("keeps every step in the safe integer range", () => {
    expect(stepValue(2 ** 52, "double", 1)).toBe(2 ** 52);
    expect(stepValue(Number.MAX_SAFE_INTEGER - 1, "stride", 1)).toBe(Number.MAX_SAFE_INTEGER - 1);
    expect(stepValue(Number.MAX_SAFE_INTEGER, "unit", 1)).toBe(Number.MAX_SAFE_INTEGER);
  });

  it("moves other even values by the stride and odd ones by one, never below 1", () => {
    expect(stepValue(384, "stride", 1)).toBe(384 + STRIDE);
    expect(stepValue(384, "stride", -1)).toBe(384 - STRIDE);
    expect(stepValue(20, "stride", -1)).toBe(20);
    expect(stepValue(77, "unit", 1)).toBe(78);
    expect(stepValue(1, "unit", -1)).toBe(1);
  });

  /* The mode is held for a run of steps: an odd start walks through 4 and 8
     one at a time, and only a fresh run from 8 doubles. */
  it("keeps the run's mode through values that would read differently", () => {
    const mode = stepModeOf(3);
    let value = 3;
    const seen = [value];
    while (value < 9) seen.push((value = stepValue(value, mode, 1)));
    expect(seen).toEqual([3, 4, 5, 6, 7, 8, 9]);
    expect(stepValue(8, stepModeOf(8), 1)).toBe(16);
  });
});

/* The hint is drawn from an empty marker, so the highlight layer keeps the
   textarea's characters exactly and the caret stays aligned. */
describe("parameter hints in the editor", () => {
  it("adds empty markers at each parameter's line end without changing the text", () => {
    const offsets = sourceParams(source).map((p) => p.lineEnd);
    const tokens = withHints(overlayTokens(source), offsets);
    expect(tokens.map((t) => t.text).join("")).toBe(overlayTokens(source).map((t) => t.text).join(""));
    const hints: number[] = [];
    let at = 0;
    for (const token of tokens) {
      if (token.kind === "hint") hints.push(at);
      at += token.text.length;
    }
    expect(hints).toEqual(offsets);
  });
});


describe("the source a build's parameter controls belong to", () => {
  const built = "M = 8\nX = Tensor(M * 2, dtype=float16)\nY = relu(X)\n";

  it("allows several parameter edits and ignores comments and formatting", () => {
    const draft = "# resized\n  M = 16 # next run\n\nX = Tensor(M*2, dtype=float16)\nY = relu(X)\n";
    expect(parameterSourceKey(draft)).toBe(parameterSourceKey(built));
  });

  it.each([
    ["empty source", ""],
    ["only parameters", "M = 8\n"],
    ["different operation", built.replace("relu", "exp")],
    ["different literal extent", built.replace("M * 2", "M * 3")],
    ["different dtype", built.replace("float16", "float32")],
    ["different parameter name", built.split("M").join("N")],
    ["invalid parameter", built.replace("M = 8", "M = 0")],
  ])("rejects %s", (_label, draft) => {
    expect(parameterSourceKey(draft)).not.toBe(parameterSourceKey(built));
  });

  it("rejects incomplete drafts rather than trusting recovered statements", () => {
    expect(parameterSourceKey(built + "Z = relu(\n")).toBeNull();
  });

  it("keeps string contents significant", () => {
    const a = 'N = 4\nX = Tensor(N, N)\nY = einsum("ij->ij", X)\n';
    expect(parameterSourceKey(a.replace("ij->ij", "ij->ji"))).not.toBe(parameterSourceKey(a));
  });
});


it("reuses source identities across store ticks and parses only a changed draft", () => {
  const built = "Q = 8\nInput = Tensor(Q)\nOutput = relu(Input)\n";
  const draft = built.replace("Q = 8", "Q = 16");
  const parse = vi.spyOn(parser, "parseProgram");
  try {
    expect(sameParameterizedSource(built, draft)).toBe(true);
    expect(parse).toHaveBeenCalledTimes(2);
    for (let tick = 0; tick < 20; tick++) expect(sameParameterizedSource(built, draft)).toBe(true);
    expect(parse).toHaveBeenCalledTimes(2);
    expect(sameParameterizedSource(built, draft.replace("Q = 16", "Q = 32"))).toBe(true);
    expect(parse).toHaveBeenCalledTimes(3);
    expect(sameParameterizedSource(built, draft.replace("relu", "exp"))).toBe(false);
    expect(parse).toHaveBeenCalledTimes(4);
  } finally { parse.mockRestore(); }
});
