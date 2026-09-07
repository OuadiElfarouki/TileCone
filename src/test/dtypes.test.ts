/**
 * The dtype promotion lattice.
 *
 * A join over a lattice owes three laws, and they are worth checking directly
 * rather than through a handful of examples: an asymmetric `promoteDType` would
 * make `add(a, b)` and `add(b, a)` different types, and a non-associative one
 * would make an n-ary add depend on the order its operands were written.
 */

import { describe, expect, it } from "vitest";
import { DTYPES, DType, DTYPE_BYTES, promoteDType, promoteDTypes } from "../core/dtypes";

const pairs: [DType, DType][] = DTYPES.flatMap((a) => DTYPES.map((b) => [a, b] as [DType, DType]));

describe("promoteDType is a join", () => {
  it("is idempotent", () => {
    for (const d of DTYPES) expect(promoteDType(d, d)).toBe(d);
  });

  it("is commutative", () => {
    for (const [a, b] of pairs)
      expect(promoteDType(a, b), `${a} vs ${b}`).toBe(promoteDType(b, a));
  });

  it("is associative", () => {
    for (const [a, b] of pairs)
      for (const c of DTYPES)
        expect(
          promoteDType(promoteDType(a, b), c),
          `(${a},${b}),${c}`
        ).toBe(promoteDType(a, promoteDType(b, c)));
  });

  /* Monotonic in width only *within* a family. Across families the category
     wins outright and may narrow in bytes: `f16` with `i32` is `f16`, matching
     PyTorch. NumPy's older value-based rule answered `f64`, which describes no
     kernel anyone runs. */
  it("never narrows within a family", () => {
    const family = (d: DType) =>
      d === "bool" ? "bool" : d === "i8" || d === "i32" ? "int" : "float";
    for (const [a, b] of pairs) {
      if (family(a) !== family(b)) continue;
      const r = promoteDType(a, b);
      expect(DTYPE_BYTES[r], `${a} + ${b} -> ${r}`).toBeGreaterThanOrEqual(
        Math.max(DTYPE_BYTES[a], DTYPE_BYTES[b])
      );
    }
  });

  it("keeps the float across families rather than widening for an int's range", () => {
    expect(promoteDType("f16", "i32")).toBe("f16");
    expect(promoteDType("f8", "i32")).toBe("f8");
  });

  it("always returns a real dtype", () => {
    for (const [a, b] of pairs) expect(DTYPES).toContain(promoteDType(a, b));
  });
});

describe("the lattice's specific commitments", () => {
  it("bool is the bottom of every chain", () => {
    for (const d of DTYPES) expect(promoteDType("bool", d)).toBe(d);
  });

  it("a float beats any integer", () => {
    for (const f of ["f8", "f16", "bf16", "f32"] as DType[])
      for (const i of ["i8", "i32"] as DType[]) expect(promoteDType(f, i)).toBe(f);
  });

  it("integers order by width", () => {
    expect(promoteDType("i8", "i32")).toBe("i32");
  });

  it("floats order f8 < f16 < f32", () => {
    expect(promoteDType("f8", "f16")).toBe("f16");
    expect(promoteDType("f16", "f32")).toBe("f32");
    expect(promoteDType("f8", "f32")).toBe("f32");
  });

  /* The one case where the result is wider than both operands. f16 and bf16 are
     the same width and neither contains the other - bf16 trades mantissa for
     f32's exponent range, f16 the reverse - so the only type holding both is
     f32. Picking either operand would silently discard range or precision. */
  it("f16 with bf16 widens to f32 rather than picking one", () => {
    expect(promoteDType("f16", "bf16")).toBe("f32");
    expect(promoteDType("bf16", "f16")).toBe("f32");
  });
});

describe("promoteDTypes folds the list", () => {
  it("agrees with a left fold of the pairwise join", () => {
    const list: DType[] = ["i8", "f16", "bool", "f32"];
    expect(promoteDTypes(list)).toBe(list.reduce(promoteDType));
  });

  it("is order-independent, which n-ary calls rely on", () => {
    const list: DType[] = ["bool", "i8", "bf16", "f8"];
    const reversed = [...list].reverse();
    expect(promoteDTypes(list)).toBe(promoteDTypes(reversed));
  });

  it("rejects an empty list rather than inventing a type", () => {
    expect(() => promoteDTypes([])).toThrow(/no dtypes/);
  });
});
