/** Canonical scalar types understood by the graph, DSL, and metric engine. */
export const DTYPES = ["f32", "f16", "bf16", "f8", "i32", "i8", "bool"] as const;
export type DType = (typeof DTYPES)[number];

export const DTYPE_BYTES: Record<DType, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  f8: 1,
  i32: 4,
  i8: 1,
  bool: 1,
};

/**
 * Promotion for operations that read more than one tensor.
 *
 * Real inference graphs mix precisions constantly - fp16 activations against an
 * fp32 residual or scale, an int8 weight against an fp16 input - and refusing
 * them made a large class of realistic programs unwritable. Byte accounting is
 * unaffected, because bytes are measured per tensor from that tensor's own
 * dtype; promotion only decides what the *result* is.
 *
 * The lattice is the conventional one, with a single deliberate strictness:
 *
 * - `bool` is the bottom of every chain and promotes to whatever it meets.
 * - Integers order by width; a float beats any integer, and does **not** widen
 *   to accommodate its range. `f16` with `i32` is `f16`. This follows PyTorch
 *   rather than NumPy's older value-based rule, which would answer `f64` here:
 *   the point of this lattice is to describe what an inference kernel actually
 *   does, and no kernel promotes fp16 activations to fp64 because an index
 *   tensor was int32. Category dominates width across families; width only
 *   orders types within one.
 * - Floats order `f8 < f16 ~ bf16 < f32`.
 * - **`f16` and `bf16` promote to `f32`, not to each other.** They have the same
 *   width and neither contains the other: bf16 has f32's exponent range with
 *   fewer mantissa bits, f16 the reverse trade. Picking either as the winner
 *   would silently discard range or precision, so the join is the type that
 *   holds both. This is what PyTorch does, and it is the one case where the
 *   result is wider than both operands.
 */
type Family = "bool" | "int" | "float";

const FAMILY: Record<DType, Family> = {
  bool: "bool",
  i8: "int",
  i32: "int",
  f8: "float",
  f16: "float",
  bf16: "float",
  f32: "float",
};

/** Rank within a family; only compared against members of the same family. */
const RANK: Record<DType, number> = {
  bool: 0,
  i8: 0,
  i32: 1,
  f8: 0,
  f16: 1,
  bf16: 1,
  f32: 2,
};

/** The narrowest type that holds both, per the lattice documented above. */
export function promoteDType(a: DType, b: DType): DType {
  if (a === b) return a;
  const fa = FAMILY[a];
  const fb = FAMILY[b];
  if (fa === "bool") return b;
  if (fb === "bool") return a;
  if (fa !== fb) return fa === "float" ? a : b;
  // Same family, different type: the equal-rank float pair is the only case
  // where neither operand can serve as the result.
  if (RANK[a] === RANK[b]) return "f32";
  return RANK[a] > RANK[b] ? a : b;
}

/** Fold `promoteDType` over a non-empty list. */
export function promoteDTypes(dtypes: DType[]): DType {
  if (!dtypes.length) throw new Error("promoteDTypes: no dtypes");
  return dtypes.reduce(promoteDType);
}
