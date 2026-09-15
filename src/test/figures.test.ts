/**
 * What a figure claims, and what it refuses to claim.
 *
 * Every number the cost panel prints is wrong in one of three ways or none, and
 * before this the panel had one word for all of them. A barrier contributes
 * zero FLOPs - the only honest answer for an operation nobody described - while
 * the mapped operations around it contribute upper bounds over widened regions.
 * Their sum was printed with `≤`, which says "no more than this" about a number
 * that is not an upper bound on anything.
 */

import { describe, expect, it } from "vitest";
import { compileDSL } from "../parse/compiler";
import { computeMetrics, addFigures, figure, ratioFigure, sumFigures } from "../core/metrics";
import { propagateBackward } from "../core/propagate";
import { box, fromBox } from "../core/region";
import { resolveGraph } from "../core/graph";
import { parseGraphJSON } from "../parse/json";

const metricsOf = (source: string, tensorId: string, region: ReturnType<typeof fromBox>) => {
  const { resolved } = compileDSL(source);
  return computeMetrics(resolved, propagateBackward(resolved, { tensorId, region }));
};

/** An exact chain: modelled operations, no widening anywhere. */
const EXACT = `X = Tensor(8, dtype=fp32)
Y = relu(X)
Z = relu(Y)
`;

/** The same chain with one operation nobody described in the middle. */
const BARRIER = `X = Tensor(8, dtype=fp32)
H = opaque(X, op="Resize", shapes=[[8]])
Z = relu(H)
`;

describe("the figure algebra", () => {
  it("keeps the weaker claim when figures are summed", () => {
    const exact = figure(10, "exact");
    const upper = figure(5, "upper", ["widened"]);
    expect(addFigures(exact, upper)).toMatchObject({ value: 15, status: "upper" });
    expect(addFigures(exact, exact)).toMatchObject({ value: 20, status: "exact" });
  });

  /* Not a looser bound: an unknown quantity added to a known one gives a sum
     nobody can place, so the total loses its number rather than keeping the
     half that happened to be measurable. */
  it("loses the number entirely when an unknown is summed in", () => {
    const sum = addFigures(figure(10, "exact"), figure(0, "unknown", ["unknown work in Resize"]));
    expect(sum.value).toBeNull();
    expect(sum.status).toBe("unknown");
    expect(sum.reasons).toContain("unknown work in Resize");
  });

  it("sums an empty list to an exact zero", () => {
    expect(sumFigures([])).toMatchObject({ value: 0, status: "exact" });
  });

  /* A ratio has no direction even when both its parts do. Widening a region
     raises the numerator and the denominator at once, so the quotient can land
     either side of the truth - which is what `approximate` means, and what an
     inherited `upper` would have misstated. */
  it("makes a ratio of two bounds approximate rather than upper", () => {
    const ratio = ratioFigure(figure(10, "upper", ["a"]), figure(5, "upper", ["b"]));
    expect(ratio).toMatchObject({ value: 2, status: "approximate" });
    expect(ratio.reasons).toEqual(["a", "b"]);
  });

  it("has no ratio at all when its numerator is unknown", () => {
    const ratio = ratioFigure(figure(0, "unknown", ["unknown work in Resize"]), figure(5, "exact"));
    expect(ratio.value).toBeNull();
    expect(ratio.status).toBe("unknown");
  });

  it("stays exact when both sides are", () => {
    expect(ratioFigure(figure(10, "exact"), figure(4, "exact"))).toMatchObject({
      value: 2.5,
      status: "exact",
    });
  });
});

describe("cost figures over a real cone", () => {
  it("marks bytes held in a widened canonical dtype as an upper bound", () => {
    const resolved = resolveGraph(
      parseGraphJSON(
        JSON.stringify({
          nodes: [],
          tensors: {
            X: {
              id: "X",
              name: "X",
              shape: [8],
              dtype: "i32",
              dtypeWidening: {
                from: "uint16",
                note: "held as int32: two bytes per element becomes four",
              },
            },
          },
          params: {},
        })
      )
    );
    const m = computeMetrics(
      resolved,
      propagateBackward(resolved, { tensorId: "X", region: fromBox(box([0, 4])) })
    );

    expect(m.inputBytes).toMatchObject({ value: 16, status: "upper" });
    expect(m.inputBytes.reasons).toEqual([
      "held as int32: two bytes per element becomes four",
    ]);
    expect(m.tensors[0].exact).toBe(true);
    expect(m.tensors[0].byteFigure.status).toBe("upper");
  });

  it("calls every figure a count when nothing was widened", () => {
    const m = metricsOf(EXACT, "Z", fromBox(box([0, 4])));
    for (const f of [m.flops, m.inputBytes, m.intermediateBytes, m.outputBytes, m.unfusedBytes])
      expect(f.status).toBe("exact");
    expect(m.exact).toBe(true);
    expect(m.reasons).toEqual([]);
    expect(m.fusedIntensity.status).toBe("exact");
  });

  /* The failure this whole item exists to fix. Everything downstream of a
     barrier is still measurable in bytes - the shapes are known - so those
     figures stay bounds. Only the arithmetic is missing, and it is missing
     completely rather than approximately. */
  it("refuses a FLOP total across a barrier while keeping the byte bounds", () => {
    const m = metricsOf(BARRIER, "Z", fromBox(box([0, 4])));

    expect(m.flops.value).toBeNull();
    expect(m.flops.status).toBe("unknown");
    expect(m.flops.reasons).toEqual(["unknown work in Resize"]);

    // Bytes are still known: a barrier declares its shapes, which is what makes
    // it safe. They are bounds because the region through it was widened.
    expect(m.inputBytes.value).toBe(8 * 4);
    expect(m.inputBytes.status).toBe("upper");
    expect(m.outputBytes.value).toBe(4 * 4);
  });

  it("makes both intensities unavailable rather than dividing by an unknown", () => {
    const m = metricsOf(BARRIER, "Z", fromBox(box([0, 4])));
    expect(m.fusedIntensity.value).toBeNull();
    expect(m.unfusedIntensity.value).toBeNull();
    expect(m.unfusedIntensity.status).toBe("unknown");
  });

  /* A figure is qualified by the regions it actually summed, not by whether
     anything anywhere in the readout was approximate. The old single flag put
     `≤` on figures that were counts. */
  it("does not let one widened bucket qualify a figure it never touched", () => {
    // The seed is the whole of Z, so the output row is exact; the cone reaches
    // X only through the barrier, so the input row is widened.
    const m = metricsOf(BARRIER, "Z", fromBox(box([0, 8])));
    expect(m.outputBytes.status).toBe("exact");
    expect(m.inputBytes.status).toBe("upper");
  });

  it("names the operation whose arithmetic is missing", () => {
    const m = metricsOf(
      `X = Tensor(8, dtype=fp32)
H = opaque(X, op="Attention", domain="com.microsoft", shapes=[[8]])
Z = relu(H)
`,
      "Z",
      fromBox(box([0, 4]))
    );
    // The same words the card and the region's reasons use, so the figure can
    // be traced to the node responsible for it.
    expect(m.flops.reasons).toEqual(["unknown work in com.microsoft.Attention"]);
  });

  it("still counts FLOPs for an exact cone that never reaches the barrier", () => {
    // H is produced by the barrier, but a cone rooted at H itself crosses no
    // operation on the way: nothing unknown was summed.
    const m = metricsOf(BARRIER, "X", fromBox(box([0, 8])));
    expect(m.flops.value).toBe(0);
    expect(m.flops.status).toBe("exact");
  });
});
