import { describe, expect, it } from "vitest";
import { compileDSL } from "../parse/compiler";
import { coneReadout } from "../core/metrics";
import { box, fromBox } from "../core/region";

/**
 * One tensor in two operand slots is the case the overlapping-box representation
 * exists for. The bands must survive as bands, and every quantity must still be
 * measured on the union of them.
 */
describe("a tensor read through two operand slots", () => {
  const program = () => compileDSL(`N = 256\nA = Tensor(N, N, dtype=fp16)\nC = matmul(A, A)\n`);
  const tile = fromBox(box([64, 128], [32, 96]));

  it("reports the row band and the column band, not three fragments", () => {
    const p = program();
    const row = coneReadout(p.resolved, p.executor.upstream("C", tile)).find((r) => r.name === "A")!;
    expect(row.boxCount).toBe(2);
    expect(row.sliceExprs).toContain("A[64:128, 0:256]");
    expect(row.sliceExprs).toContain("A[0:256, 32:96]");
  });

  it("counts the shared square once and states it separately", () => {
    const p = program();
    const row = coneReadout(p.resolved, p.executor.upstream("C", tile)).find((r) => r.name === "A")!;
    expect(row.elements).toBe(28672); // not 32768: the bands share a 64x64 square
    expect(row.overlap).toBe(4096);
    expect(row.bytes).toBe(28672 * 2); // f16, union elements
  });

  it("does not pay for a shared element twice in the FLOP estimate", () => {
    // S is read through both slots of the matmul, so its backward region has
    // overlapping boxes. Summing per box would over-count where they meet.
    const src = `N = 64
A = Tensor(N, N, dtype=fp16)
T = transpose(A, perm=[1,0])
S = add(A, T)
U = matmul(S, S)
`;
    const p = compileDSL(src);
    const sel = fromBox(box([8, 16], [8, 16]));
    const region = p.executor.upstream("U", sel).tensors.get(
      Object.values(p.resolved.tensors).find((t) => t.name === "S")!.id
    )!.region;
    // the premise: the boxes really do overlap
    expect(region.boxes.length).toBeGreaterThan(1);
    // 9152 is the cost over the union. Summing the boxes instead pays 9216,
    // the extra 64 being the square where the two bands of S meet.
    expect(p.executor.metrics("U", sel).flops).toBe(9152);
  });
});
