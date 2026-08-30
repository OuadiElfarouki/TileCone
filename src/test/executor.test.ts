import { describe, expect, it } from "vitest";
import { ExecutionError, executeQuery } from "../core/executor";
import { box, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";

const gemm = () =>
  compileDSL(`input A [4, 6] f16
input B [6, 5] f16
C = matmul(A, B)
`);

describe("headless symbolic executor", () => {
  it("defaults to an upstream query", () => {
    const program = gemm();
    const result = program.executor.query({
      tensorId: "C",
      region: fromBox(box([1, 3], [2, 4])),
    });

    expect(result.direction).toBe("backward");
    expect(result.forward).toBeNull();
    expect(result.backward?.tensors.get("A")?.region.boxes).toEqual([
      box([1, 3], [0, 6]),
    ]);
    expect(result.backward?.tensors.get("B")?.region.boxes).toEqual([
      box([0, 6], [2, 4]),
    ]);
  });

  it("can execute both directions in one checked query", () => {
    const program = gemm();
    const result = executeQuery(program.resolved, {
      tensorId: "A",
      region: fromBox(box([1, 2], [0, 6])),
      direction: "both",
    });

    expect(result.backward?.tensors.has("A")).toBe(true);
    expect(result.forward?.tensors.get("C")?.region.boxes).toEqual([
      box([1, 2], [0, 5]),
    ]);
  });

  it("offers upstream, downstream, and metrics convenience methods", () => {
    const { executor } = gemm();
    const region = fromBox(box([0, 1], [0, 1]));

    expect(executor.upstream("C", region).direction).toBe("backward");
    expect(executor.downstream("C", region).direction).toBe("forward");
    expect(executor.metrics("C", region)).toMatchObject({
      flops: 12,
      inputBytes: 24,
      outputBytes: 2,
    });
  });

  it("counts the shared prefix work required by a partial scan selection", () => {
    const { executor } = compileDSL(`input X [128] f32
Y = cumsum(X, axis=0)
`);
    const region = {
      boxes: [box([3, 4]), box([127, 128])],
      exact: true,
      reasons: [],
    };

    expect(executor.metrics("Y", region).flops).toBe(128);
  });

  it("uses inferred cast dtypes for output and intermediate byte metrics", () => {
    const { executor } = compileDSL(`input X [4] f32
Y = cast(X, dtype=f8)
Z = cast(Y, dtype=f16)
`);
    const region = fromBox(box([0, 4]));

    expect(executor.metrics("Y", region)).toMatchObject({
      inputBytes: 16,
      intermediateBytes: 0,
      outputBytes: 4,
    });
    expect(executor.metrics("Z", region, true)).toMatchObject({
      inputBytes: 16,
      intermediateBytes: 4,
      outputBytes: 8,
    });
  });

  it("rejects unknown tensors with a stable error code", () => {
    const { executor } = gemm();
    expect(() => executor.upstream("Missing", fromBox(box([0, 1])))).toThrowError(
      expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_UNKNOWN_TENSOR" })
    );
  });

  it("validates the direction at the runtime boundary", () => {
    const program = gemm();
    expect(() =>
      executeQuery(program.resolved, {
        tensorId: "C",
        region: fromBox(box([0, 1], [0, 1])),
        direction: "sideways" as "backward",
      })
    ).toThrowError(expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_DIRECTION" }));
  });

  it("rejects region rank mismatches before propagation", () => {
    const { executor } = gemm();
    expect(() => executor.upstream("C", fromBox(box([0, 1])))).toThrowError(
      expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_REGION_RANK" })
    );
  });

  it("rejects non-integral, reversed, and out-of-bounds intervals", () => {
    const { executor } = gemm();
    for (const region of [
      { boxes: [box([0.5, 1], [0, 1])], exact: true, reasons: [] },
      { boxes: [box([2, 1], [0, 1])], exact: true, reasons: [] },
      { boxes: [box([0, 5], [0, 1])], exact: true, reasons: [] },
    ]) {
      expect(() => executor.upstream("C", region)).toThrowError(
        expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_REGION_BOUNDS" })
      );
    }
  });

  it("defensively copies and canonicalizes the selection", () => {
    const { executor } = gemm();
    const region = {
      boxes: [box([0, 2], [0, 1]), box([1, 3], [0, 1])],
      exact: true,
      reasons: [] as string[],
    };
    const result = executor.query({ tensorId: "C", region });

    region.boxes[0][0].lo = 99;
    expect(result.selection.region.boxes).toEqual([box([0, 3], [0, 1])]);
  });
});
