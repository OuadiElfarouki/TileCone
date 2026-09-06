import { describe, expect, it } from "vitest";
import { ExecutionError, executeQuery } from "../core/executor";
import { box, count, fromBox } from "../core/region";
import { compileDSL } from "../parse/compiler";

const gemm = () =>
  compileDSL(`A = Tensor(4, 6, dtype=fp16)
B = Tensor(6, 5, dtype=fp16)
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
    const { executor } = compileDSL(`X = Tensor(128, dtype=fp32)
Y = cumsum(X, axis=0)
`);
    const region = {
      boxes: [box([3, 4]), box([127, 128])],
      exact: true,
      reasons: [],
    };

    expect(executor.metrics("Y", region).flops).toBe(128);
  });

  it("charges overlapping output boxes once without constructing a partition", () => {
    const { executor } = compileDSL(`A = Tensor(512, 512, dtype=fp16)
B = Tensor(512, 512, dtype=fp16)
C = matmul(A, B)
`);
    const region = {
      boxes: Array.from({ length: 32 }, (_, i) => [
        box([i * 3, i * 3 + 2], [0, 512]),
        box([0, 512], [i * 3, i * 3 + 2]),
      ]).flat(),
      exact: true,
      reasons: [],
    };

    expect(executor.metrics("C", region).flops).toBe(count(region) * 2 * 512);
  });

  it("uses inferred cast dtypes for output and intermediate byte metrics", () => {
    const { executor } = compileDSL(`X = Tensor(4, dtype=fp32)
Y = cast(X, dtype=fp8)
Z = cast(Y, dtype=fp16)
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
