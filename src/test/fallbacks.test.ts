/**
 * The conservative branches, against the oracle.
 *
 * These are the branches where the engine is permitted to be imprecise, so they
 * are the ones where a mistake is silent: an exact result that is wrong fails
 * loudly against brute-force truth, while an inexact one that is too *small*
 * looks like a tighter answer and is the exact failure this project exists to
 * prevent. They were also the only branches the oracle never reached, because
 * every threshold needed a tensor far larger than one it can enumerate.
 *
 * Lowering the thresholds is what makes them testable. Everything here runs at
 * shapes small enough for the oracle while taking the same code path a 300-wide
 * diagonal or a 4000-element strided slice takes in production.
 */

import { describe, expect, it } from "vitest";
import { checkGraph, G } from "./harness";
import { resolveGraph } from "../core/graph";
import { propagateBackward, propagateForward } from "../core/propagate";
import { DEFAULT_LIMITS, type Limits } from "../core/ops/limits";
import { box, fromBox } from "../core/region";
import { computeOracle, regionToFlatSet, truthBackward, truthForward } from "./oracle";

/** Thresholds low enough that a handful of elements takes the fallback. */
const TIGHT: Partial<Limits> = {
  stridedEnum: 2,
  diagEnum: 2,
  reshapeRuns: 1,
  maxBoxes: 2,
};

describe("conservative branches stay supersets, checked against the oracle", () => {
  it("diagonal einsum past the enumeration cap", () =>
    checkGraph(G({ M: [6, 6] }, [["e", "einsum", ["M"], ["d"], { equation: "ii->i" }]]), {
      limits: TIGHT,
    }));

  it("diagonal einsum with a spectator axis", () =>
    checkGraph(G({ M: [4, 4, 3] }, [["e", "einsum", ["M"], ["d"], { equation: "iij->ij" }]]), {
      limits: TIGHT,
    }));

  it("diagonal contracted away entirely", () =>
    checkGraph(G({ M: [5, 5] }, [["e", "einsum", ["M"], ["t"], { equation: "ii->" }]]), {
      limits: TIGHT,
    }));

  it("strided slice past the enumeration cap", () =>
    checkGraph(
      G({ X: [12] }, [["s", "slice", ["X"], ["Y"], { starts: [1], stops: [12], steps: [3] }]]),
      { limits: TIGHT }
    ));

  it("strided slice on two axes at once", () =>
    checkGraph(
      G({ X: [7, 9] }, [
        ["s", "slice", ["X"], ["Y"], { starts: [1, 0], stops: [6, 9], steps: [2, 4] }],
      ]),
      { limits: TIGHT }
    ));

  it("dilated conv past the enumeration cap", () =>
    checkGraph(
      G({ X: [1, 1, 9], W: [2, 1, 3] }, [
        ["c", "conv", ["X", "W"], ["Y"], { stride: [1], pads: [[2, 2]], dilation: [2], groups: 1 }],
      ]),
      { limits: TIGHT, perTensorElementCap: 12 }
    ));

  it("dilated pool past the enumeration cap", () =>
    checkGraph(
      G({ X: [1, 1, 10] }, [
        [
          "p",
          "pool",
          ["X"],
          ["Y"],
          { kind: "max", kernelShape: [2], stride: [3], pads: [[0, 0]], dilation: [3] },
        ],
      ]),
      { limits: TIGHT, perTensorElementCap: 12 }
    ));

  it("reshape past the run cap", () =>
    checkGraph(G({ X: [4, 4] }, [["r", "reshape", ["X"], ["Y"], { shape: [2, 8] }]]), {
      limits: TIGHT,
    }));

  it("reshape past the run cap, three-way split", () =>
    checkGraph(G({ X: [2, 6] }, [["r", "reshape", ["X"], ["Y"], { shape: [3, 4] }]]), {
      limits: TIGHT,
    }));

  it("a fallback feeding another operation", () =>
    checkGraph(
      G({ M: [6, 6], v: [6] }, [
        ["e", "einsum", ["M"], ["d"], { equation: "ii->i" }],
        ["a", "elementwise", ["d", "v"], ["z"], { fn: "mul", nary: 2 }],
      ]),
      { limits: TIGHT }
    ));

  it("two fallbacks in series", () =>
    checkGraph(
      G({ X: [12] }, [
        ["s", "slice", ["X"], ["Y"], { starts: [0], stops: [12], steps: [3] }],
        ["r", "reshape", ["Y"], ["Z"], { shape: [2, 2] }],
      ]),
      { limits: TIGHT }
    ));
});

describe("lowered thresholds actually take the fallback", () => {
  /* A superset test passes trivially if the region is exact, so these assert
     that the tight limits above genuinely change which branch runs. Without
     this the suite above could be checking the ordinary path and nobody would
     know. */
  const tookFallback = (
    graph: ReturnType<typeof G>,
    tensorId: string,
    sel: Parameters<typeof fromBox>[0],
    limits: Partial<Limits> | undefined
  ) => {
    const g = resolveGraph(graph);
    const res = propagateBackward(
      g,
      { tensorId, region: fromBox(sel) },
      limits ? { ...DEFAULT_LIMITS, ...limits } : undefined
    );
    return [...res.tensors.values()].some((t) => !t.region.exact);
  };

  it.each([
    [
      "diagonal einsum",
      G({ M: [6, 6] }, [["e", "einsum", ["M"], ["d"], { equation: "ii->i" }]]),
      "d",
      box([0, 6]),
    ],
    [
      "strided slice",
      G({ X: [12] }, [["s", "slice", ["X"], ["Y"], { starts: [1], stops: [12], steps: [3] }]]),
      "Y",
      box([0, 3]),
    ],
    [
      "reshape runs",
      G({ X: [4, 4] }, [["r", "reshape", ["X"], ["Y"], { shape: [2, 8] }]]),
      "Y",
      box([0, 2], [1, 7]),
    ],
  ])("%s: inexact under tight limits, exact under the defaults", (_label, graph, tid, sel) => {
    expect(tookFallback(graph, tid, sel, TIGHT)).toBe(true);
    expect(tookFallback(graph, tid, sel, undefined)).toBe(false);
  });
});

describe("forward direction takes its fallbacks too", () => {
  it("strided slice forward", () =>
    checkGraph(
      G({ X: [12] }, [["s", "slice", ["X"], ["Y"], { starts: [1], stops: [12], steps: [3] }]]),
      { limits: TIGHT, backward: false }
    ));

  it("dilated conv forward", () =>
    checkGraph(
      G({ X: [1, 1, 9], W: [2, 1, 3] }, [
        ["c", "conv", ["X", "W"], ["Y"], { stride: [1], pads: [[2, 2]], dilation: [2], groups: 1 }],
      ]),
      { limits: TIGHT, backward: false, perTensorElementCap: 12 }
    ));

  it("a widened forward image still contains the truth", () => {
    const g = resolveGraph(
      G({ X: [12] }, [["s", "slice", ["X"], ["Y"], { starts: [0], stops: [12], steps: [3] }]])
    );
    const oracle = computeOracle(g);
    const sel = fromBox(box([0, 12]));
    const res = propagateForward(g, { tensorId: "X", region: sel }, {
      ...DEFAULT_LIMITS,
      ...TIGHT,
    });
    const truth = truthForward(oracle, "X", sel, "Y");
    const got = regionToFlatSet(res.tensors.get("Y")!.region, g.tensors.Y.resolved!);
    for (const f of truth) expect(got.has(f)).toBe(true);
  });

  it("a widened backward preimage still contains the truth", () => {
    const g = resolveGraph(G({ M: [6, 6] }, [["e", "einsum", ["M"], ["d"], { equation: "ii->i" }]]));
    const oracle = computeOracle(g);
    const sel = fromBox(box([0, 6]));
    const res = propagateBackward(g, { tensorId: "d", region: sel }, {
      ...DEFAULT_LIMITS,
      ...TIGHT,
    });
    const truth = truthBackward(g, oracle, "d", sel, "M");
    const got = regionToFlatSet(res.tensors.get("M")!.region, g.tensors.M.resolved!);
    for (const f of truth) expect(got.has(f)).toBe(true);
  });
});
