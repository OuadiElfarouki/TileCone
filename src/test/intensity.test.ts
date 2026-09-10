import { describe, expect, it } from "vitest";
import { compileDSL } from "../parse/compiler";
import { box, fromBox } from "../core/region";

/**
 * Arithmetic intensity, which had no coverage at all while it was a single
 * figure over a denominator that left the output out.
 */
describe("arithmetic intensity", () => {
  it("charges the tile's own write in both readings", () => {
    const { executor } = compileDSL(`A = Tensor(128, 128, dtype=fp32)
B = Tensor(128, 128, dtype=fp32)
C = matmul(A, B)
`);
    const m = executor.metrics("C", fromBox(box([0, 32], [0, 32])));

    // 32x32 tile of C reads a 32x128 band of A and a 128x32 band of B.
    expect(m.inputBytes).toBe(2 * 32 * 128 * 4);
    expect(m.outputBytes).toBe(32 * 32 * 4);
    expect(m.intermediateBytes).toBe(0);

    // The denominator is traffic, so it includes the write. Dropping the
    // output - what this figure used to do - reports 8.00 instead.
    expect(m.fusedIntensity).toBeCloseTo(m.flops / (m.inputBytes + m.outputBytes), 6);
    expect(m.fusedIntensity).toBeCloseTo(7.111, 3);
    expect(m.fusedIntensity).not.toBeCloseTo(m.flops / m.inputBytes, 3);
  });

  it("reports one figure twice when there is nothing to fuse", () => {
    const { executor } = compileDSL(`A = Tensor(128, 128, dtype=fp32)
B = Tensor(128, 128, dtype=fp32)
C = matmul(A, B)
`);
    const m = executor.metrics("C", fromBox(box([0, 32], [0, 32])));

    expect(m.intermediateBytes).toBe(0);
    expect(m.unfusedIntensity).toBeCloseTo(m.fusedIntensity, 9);
  });

  it("separates the two readings by exactly the intermediate traffic", () => {
    const { executor } = compileDSL(`X = Tensor(64, 64, dtype=fp32)
Y = relu(X)
Z = relu(Y)
`);
    const m = executor.metrics("Z", fromBox(box([0, 16], [0, 16])));

    // One 16x16 tile: read X, write Y, read Y, write Z. 512 FLOPs over two ops.
    expect(m.flops).toBe(512);
    expect(m.inputBytes).toBe(1024);
    expect(m.intermediateBytes).toBe(1024);
    expect(m.outputBytes).toBe(1024);

    expect(m.fusedIntensity).toBeCloseTo(512 / 2048, 6);
    expect(m.unfusedBytes).toBe(4096);
    expect(m.unfusedIntensity).toBeCloseTo(512 / 4096, 6);
    expect(m.fusedIntensity / m.unfusedIntensity).toBeCloseTo(2, 6);
  });

  it("charges shared inputs separately for each operation that reads them", () => {
    const { executor } = compileDSL(`X = Tensor(16, dtype=fp32)
Y = relu(X)
A = relu(Y)
B = relu(Y)
Z = add(A, B)
`);
    const m = executor.metrics("Z", fromBox(box([0, 16])));
    // Y: read X/write Y; A and B: each read Y/write output;
    // Z: read A and B/write Z. Nine tensor footprints, not six.
    expect(m.unfusedBytes).toBe(9 * 16 * 4);
  });

  it("deduplicates overlapping operand reads within one operation", () => {
    const { executor } = compileDSL(`X = Tensor(16, dtype=fp32)
Y = add(X, X)
`);
    expect(executor.metrics("Y", fromBox(box([0, 16]))).unfusedBytes).toBe(2 * 16 * 4);
  });

  it("never divides by zero on a cone with no traffic", () => {
    const { executor } = compileDSL(`X = Tensor(64, dtype=fp32)
Y = relu(X)
`);
    // An empty selection touches nothing, so every byte count is zero and the
    // ratio has no denominator to divide by.
    const m = executor.metrics("Y", { boxes: [], exact: true, reasons: [] });

    expect(m.inputBytes + m.intermediateBytes + m.outputBytes).toBe(0);
    expect(m.fusedIntensity).toBe(0);
    expect(m.unfusedIntensity).toBe(0);
  });
});
