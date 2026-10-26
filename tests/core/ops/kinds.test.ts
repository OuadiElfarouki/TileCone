/**
 * Attribute kinds: each written form, what it means, and how each refusal reads.
 *
 * The kinds are the single definition of what a value may be written as, so
 * these tests are the definition's examples. The last group checks the
 * property that makes short forms safe: a graph written with them resolves to
 * the same canonical attributes, and the same cones, as one written in full.
 */

import { describe, expect, it } from "vitest";
import { resolveGraph, resolveGraphCollecting } from "../../../src/core/graph";
import {
  axes,
  axis,
  canonicalAttrs,
  checkAttrs,
  KindCtx,
  KindIssue,
  padWidth,
  padding,
  permutation,
  spatial,
  spatialRankOf,
  splitSizes,
  type Kind,
} from "../../../src/core/ops/kinds";
import { box, fromBox } from "../../../src/core/region";
import { propagateBackward } from "../../../src/core/propagate";
import { G } from "../../corpus/harness";

const ctx = (shapes: number[][], names: (string | undefined)[][] = [], written = {}, params = {}, outputCount = 1): KindCtx => ({
  inShapes: shapes,
  inAxisNames: shapes.map((s, i) => names[i] ?? s.map(() => undefined)),
  written,
  params,
  outputCount,
});

/** The canonical value, after the context-free check passed. */
const canon = (kind: Kind, value: unknown, c: KindCtx) => {
  expect(kind.check(value)).toBeNull();
  return kind.canonical(value, c);
};

const issue = (kind: Kind, value: unknown, c: KindCtx): string => {
  expect(kind.check(value)).toBeNull();
  try {
    kind.canonical(value, c);
  } catch (e) {
    expect(e).toBeInstanceOf(KindIssue);
    return (e as Error).message;
  }
  throw new Error("expected the operands to rule the value out");
};

describe("axis", () => {
  const nchw = ctx([[2, 3, 4, 5]], [["n", "c", "h", "w"]]);
  it("reads an index, a negative index and a name", () => {
    expect(canon(axis(), 1, nchw)).toBe(1);
    expect(canon(axis(), -1, nchw)).toBe(3);
    expect(canon(axis(), "h", nchw)).toBe(2);
  });
  it("refuses an index out of range and a name the operand does not have", () => {
    expect(issue(axis(), 4, nchw)).toBe("axis 4 is out of range for rank 4");
    expect(issue(axis(), "seq", nchw)).toBe('no axis named "seq"; the axes are n, c, h, w');
    expect(issue(axis(), "seq", ctx([[2, 3]]))).toBe(`no axis named "seq"; this operand's axes have no names`);
  });
  it("refuses a name that more than one axis carries", () => {
    const scores = ctx([[1, 4, 8, 8]], [["batch", undefined, "seq", "seq"]]);
    expect(issue(axis(), "seq", scores)).toBe('"seq" names more than one axis (2 and 3); write the index');
  });
  it("refuses what is no axis at all, before the operands are known", () => {
    expect(axis().check(1.5)).toMatch(/^expected an axis/);
    expect(axis().check([0])).toMatch(/^expected an axis/);
  });
  it("reads the operand it is told to", () => {
    expect(canon(axis({ of: 1 }), -1, ctx([[2], [2, 3, 4]]))).toBe(2);
  });
});

describe("axes", () => {
  const c = ctx([[2, 3, 4]], [["b", "s", "e"]]);
  it("reads one axis or a list, distinct and ascending", () => {
    expect(canon(axes(), -1, c)).toEqual([2]);
    expect(canon(axes(), ["e", 0, -3], c)).toEqual([0, 2]);
  });
  it("means every axis when left out, where the operation says so", () => {
    expect(axes({ all: true }).default!(c)).toEqual([0, 1, 2]);
    expect(axes().default).toBeUndefined();
  });
  it("refuses an empty list", () => {
    expect(axes().check([])).toMatch(/^expected an axis or a non-empty list/);
  });
});

describe("permutation", () => {
  const c = ctx([[2, 3, 4]], [["b", "s", "e"]]);
  it("reads indices and names", () => {
    expect(canon(permutation(), [0, 2, 1], c)).toEqual([0, 2, 1]);
    expect(canon(permutation(), ["e", "b", -2], c)).toEqual([2, 0, 1]);
  });
  it("refuses a list that misses or repeats an axis", () => {
    expect(issue(permutation(), [0, 1], c)).toBe("[0, 1] does not name each of the 3 axes once");
    expect(issue(permutation(), [0, 0, 1], c)).toBe("[0, 0, 1] does not name each of the 3 axes once");
  });
});

describe("spatial", () => {
  const nchw = ctx([[1, 3, 8, 8]]);
  it("reads one int for every spatial axis, or one each", () => {
    expect(canon(spatial({ min: 1 }), 2, nchw)).toEqual([2, 2]);
    expect(canon(spatial({ min: 1 }), [2, 1], nchw)).toEqual([2, 1]);
  });
  it("refuses a list of the wrong length, and values below the minimum", () => {
    expect(issue(spatial({ min: 1 }), [2, 1, 1], nchw)).toBe("has 3 entries for 2 spatial axes");
    expect(spatial({ min: 1 }).check(0)).toBe("expected an int ≥ 1, or one per spatial axis");
  });
  it("leaves an operand without spatial axes to the operation's own error", () => {
    expect(canon(spatial({ min: 1 }), 2, ctx([[3, 8]]))).toEqual([]);
  });
});

describe("padding", () => {
  const ncl = ctx([[1, 3, 8]]);
  const nchw = ctx([[1, 3, 8, 8]]);
  it("reads an int, one int per axis, and one pair per axis", () => {
    expect(canon(padding(), 1, nchw)).toEqual([[1, 1], [1, 1]]);
    expect(canon(padding(), [1, 0], nchw)).toEqual([[1, 1], [0, 0]]);
    expect(canon(padding(), [[0, 1], [2, 3]], nchw)).toEqual([[0, 1], [2, 3]]);
    expect(canon(padding(), [[0, 1]], ncl)).toEqual([[0, 1]]);
  });
  it("refuses a negative amount and a list of the wrong length", () => {
    expect(padding().check(-1)).toMatch(/^expected an int ≥ 0/);
    expect(issue(padding(), [1, 1, 1], nchw)).toBe("has 3 entries for 2 axes");
  });
});

describe("padWidth", () => {
  const c = ctx([[4, 5]]);
  it("reads NumPy's forms: an int, one pair for every axis, one pair per axis", () => {
    expect(canon(padWidth(), 2, c)).toEqual([[2, 2], [2, 2]]);
    expect(canon(padWidth(), [1, 2], c)).toEqual([[1, 2], [1, 2]]);
    expect(canon(padWidth(), [[0, 1], [2, 3]], c)).toEqual([[0, 1], [2, 3]]);
  });
  it("refuses one pair per axis for the wrong number of axes", () => {
    expect(issue(padWidth(), [[0, 1], [2, 3], [4, 5]], c)).toBe("has 3 entries for 2 axes");
  });
});

describe("spatialRankOf", () => {
  it("is read off the operand when left out, and checked against it when written", () => {
    const nchw = ctx([[1, 3, 8, 8]]);
    expect(spatialRankOf().default!(nchw)).toBe(2);
    expect(canon(spatialRankOf(), 2, nchw)).toBe(2);
    expect(issue(spatialRankOf(), 1, nchw)).toBe(
      "expects 1 spatial axis (a rank-3 input), got a rank-4 input"
    );
  });
});

describe("splitSizes", () => {
  const kind = splitSizes({ axisAttr: "axis" });
  it("keeps a list of sizes as written", () => {
    expect(canon(kind, [2, "D"], ctx([[6]], [], { axis: 0 }))).toEqual([2, "D"]);
  });
  it("cuts equal pieces, keeping a symbol, and a shorter last piece", () => {
    expect(canon(kind, "D", ctx([[4, 12]], [], { axis: 1 }, { D: 4 }, 3))).toEqual(["D", "D", "D"]);
    expect(canon(kind, 5, ctx([[12]], [], { axis: 0 }, {}, 3))).toEqual([5, 5, 2]);
  });
  it("reads the axis by name", () => {
    expect(canon(kind, 2, ctx([[4, 6]], [["b", "e"]], { axis: "b" }, {}, 2))).toEqual([2, 2]);
  });
});

describe("whole attribute sets", () => {
  const kinds = { axis: axis(), stride: spatial({ min: 1, default: 1 }) };
  it("reports unknown attributes with a suggestion, and missing required ones", () => {
    expect(checkAttrs(kinds, { axis: 0, strid: 2 })).toEqual({
      attribute: "strid",
      message: 'unknown attribute "strid" (did you mean "stride"?)',
    });
    expect(checkAttrs(kinds, {})).toEqual({
      attribute: "axis",
      message: "axis: missing; expected an axis (an int, negative from the end, or an axis name)",
    });
  });
  it("fills defaults and names the attribute an operand rules out", () => {
    const c = ctx([[1, 3, 8, 8]], [], { axis: 1 });
    expect(canonicalAttrs(kinds, c)).toEqual({ attrs: { axis: 1, stride: [1, 1] } });
    expect(canonicalAttrs(kinds, { ...c, written: { axis: 7 } })).toEqual({
      problem: { attribute: "axis", message: "axis: axis 7 is out of range for rank 4" },
    });
  });
});

describe("short forms resolve like the full forms", () => {
  const convIn = { X: [1, 2, 9, 9], W: [4, 2, 3, 3] };
  const short = G(convIn, [["c", "conv", ["X", "W"], ["Y"], { stride: 2, pads: 1 }]]);
  const full = G(convIn, [
    ["c", "conv", ["X", "W"], ["Y"], { stride: [2, 2], pads: [[1, 1], [1, 1]], dilation: [1, 1], groups: 1 }],
  ]);

  it("gives a convolution the same canonical attributes and the same cone", () => {
    const a = resolveGraph(short);
    const b = resolveGraph(full);
    expect(a.topo[0].attrs).toEqual(b.topo[0].attrs);
    expect(a.topo[0].attrs).toMatchObject({ spatialRank: 2 });
    const tile = fromBox(box([0, 1], [0, 4], [1, 3], [2, 4]));
    const coneA = propagateBackward(a, { tensorId: "Y", region: tile });
    const coneB = propagateBackward(b, { tensorId: "Y", region: tile });
    expect(coneA.tensors.get("X")?.region).toEqual(coneB.tensors.get("X")?.region);
    expect(coneA.tensors.get("X")?.region.exact).toBe(true);
  });

  it("keeps the source graph as written", () => {
    resolveGraph(short);
    expect(short.nodes[0].attrs).toEqual({ stride: 2, pads: 1 });
  });

  it("lets a pool step by its own window when no stride is written", () => {
    const g = G({ X: [1, 2, 8, 8] }, [["p", "pool", ["X"], ["Y"], { kernelShape: 2 }]]);
    const r = resolveGraph(g);
    expect(r.tensors.Y.resolved).toEqual([1, 2, 4, 4]);
    expect(r.topo[0].attrs).toMatchObject({ stride: [2, 2], pads: [[0, 0], [0, 0]], dilation: [1, 1] });
  });

  it("reduces over every axis when none is written", () => {
    const r = resolveGraph(G({ X: [2, 3] }, [["r", "reduce", ["X"], ["Y"], { fn: "sum" }]]));
    expect(r.tensors.Y.resolved).toEqual([]);
  });

  it("reads an axis by the name the operand carries", () => {
    const g = G({ X: [2, 5] }, [["s", "softmax", ["X"], ["Y"], { axis: "seq" }]]);
    g.tensors.X.axisNames = ["batch", "seq"];
    expect(resolveGraph(g).topo[0].attrs).toEqual({ axis: 1 });
  });

  it("reports a spatial rank the operand does not have, at that attribute", () => {
    const g = G({ X: [1, 2, 9], W: [4, 2, 3] }, [["c", "conv", ["X", "W"], ["Y"], { spatialRank: 2 }]]);
    const { errors } = resolveGraphCollecting(g);
    expect(errors[0].code).toBe("GRAPH_INVALID_ATTRIBUTES");
    expect(errors[0].subject).toMatchObject({ attribute: "spatialRank" });
    expect(errors[0].message).toContain("expects 2 spatial axes (a rank-4 input), got a rank-3 input");
  });
});
