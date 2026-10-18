import { describe, expect, it } from "vitest";
import { compareDims, symProduct, symSum } from "../../src/core/dim-algebra";

describe("comparing dimension expressions for every binding", () => {
  it.each([
    ["N/2*2", "N"],
    ["(H*D)/H", "D"],
    ["M*N", "N*M"],
    ["P+T", "T+P"],
    ["2*(A+B)", "A*2+B+B"],
    ["E/H*H", "E"],
    ["(A+B)/(C+D)", "(B+A)/(D+C)"],
    [8, "4*2"],
  ])("%s is the same expression as %s", (a, b) => {
    expect(compareDims(a, b)).toBe("same");
  });

  it.each([
    ["M*2", "N*2"],
    ["K", 8],
    ["M*N", "N*N"],
    ["D*2+H", "H*D"],
    ["N/2", "N"],
  ])("%s differs from %s for some binding", (a, b) => {
    expect(compareDims(a, b)).toBe("different");
  });

  it("does not claim a difference through a division by a sum", () => {
    expect(compareDims("X/(A+B)", "Y")).toBe("undecided");
  });

  it("treats an opaque symbol as unknown, but lets it cancel", () => {
    const opaque = new Set(["U"]);
    expect(compareDims("U", "M", { opaque })).toBe("undecided");
    expect(compareDims("U+M", "M+U", { opaque })).toBe("same");
    expect(compareDims("U+M", "U+N", { opaque })).toBe("different");
  });

  it("expands a derived symbol into its definition", () => {
    const definitions = { E: "H*D", HALF: "E/2" };
    expect(compareDims("E", "D*H", { definitions })).toBe("same");
    expect(compareDims("HALF*2", "H*D", { definitions })).toBe("same");
    expect(compareDims("E", "M*N", { definitions })).toBe("different");
  });

  it("expands each definition once, however often a chain reads it", () => {
    // Each link reads the one before twice; expanded naively this doubles per link.
    const definitions: Record<string, string> = {};
    for (let i = 1; i <= 64; i++) definitions[`D${i}`] = `D${i - 1}+D${i - 1}-D${i - 1}`;
    const started = performance.now();
    expect(compareDims("D64", "D0", { definitions })).toBe("same");
    expect(compareDims("D64", "D1*2", { definitions })).toBe("different");
    expect(performance.now() - started).toBeLessThan(500);
  });

  it("answers undecided past its work budget rather than stalling", () => {
    // Each definition is within the grammar's node cap; together they are not
    // within the comparison's budget, even expanded once each.
    const sum = Array.from({ length: 200 }, (_, i) => `P${i}`).join("+");
    const definitions: Record<string, string> = {};
    for (let i = 0; i < 250; i++) definitions[`S${i}`] = sum;
    const all = Object.keys(definitions).join("+");
    expect(compareDims(all, "Q", { definitions })).toBe("undecided");
    expect(compareDims("S0", "Q", { definitions })).toBe("different");
  });

  it("does not loop on a cyclic definition", () => {
    expect(compareDims("A", "B", { definitions: { A: "B+1", B: "A-1" } })).toBe("undecided");
  });

  it("gives up rather than expanding an unbounded product", () => {
    const factors = Array.from({ length: 16 }, (_, i) => `(A${i}+B${i})`);
    expect(compareDims(factors.join("*"), "C")).toBe("undecided");
  });

  it("builds products and sums the dimension grammar reads back", () => {
    expect(symProduct(["M", "N+1", 2])).toBe("M*(N+1)*2");
    expect(symProduct([2, 3])).toBe(6);
    expect(symSum(["P", 4])).toBe("P+4");
    expect(compareDims(symProduct(["N/2", "M*2"]), "M*N")).toBe("same");
  });
});
