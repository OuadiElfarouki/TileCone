/**
 * Canonical scalar types understood by the graph, DSL, and metric engine.
 *
 * `i64` and `u8` are here because imported models require them, not because the
 * DSL wanted more ways to write an integer. Every index, axis and shape tensor
 * an exporter writes is int64, and mapping those to `i32` would halve their
 * byte count everywhere a footprint is measured - a narrowing nobody asked for
 * and nothing would report. `u8` is the same argument one step ahead: a
 * quantised model's `zero_point` is uint8, and dropping it understates the
 * footprint of exactly the models people want to look at.
 */
export const DTYPES = ["f32", "f16", "bf16", "f8", "i64", "i32", "u8", "i8", "bool"] as const;
export type DType = (typeof DTYPES)[number];

/**
 * Provenance for a source dtype represented by a wider canonical dtype.
 *
 * The canonical dtype remains the type operation inference uses. This marker is
 * for byte accounting: its storage footprint is an upper bound on the source
 * tensor's footprint and must never be presented as an exact count.
 */
export type DTypeWidening = {
  from: string;
  note: string;
};

export const DTYPE_BYTES: Record<DType, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  f8: 1,
  i64: 8,
  i32: 4,
  u8: 1,
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
 * - Integers order `i8 ~ u8 < i32 < i64`.
 * - Floats order `f8 < f16 ~ bf16 < f32`.
 * - **Two distinct types of the same width promote to the next width up in
 *   their own family, not to each other.** `f16` and `bf16` are the same size
 *   and neither contains the other: bf16 has f32's exponent range with fewer
 *   mantissa bits, f16 the reverse trade. `i8` and `u8` are the same size and
 *   neither contains the other either: one reaches -128, the other 255. Picking
 *   a winner would silently discard range or precision in both cases, so the
 *   join is the narrowest type holding both - `f32` and `i32` respectively.
 *   This is what PyTorch does, and it is the only case where the result is
 *   wider than both operands.
 */
export type DTypeFamily = "bool" | "int" | "float";
type Family = DTypeFamily;

const FAMILY: Record<DType, Family> = {
  bool: "bool",
  i8: "int",
  u8: "int",
  i32: "int",
  i64: "int",
  f8: "float",
  f16: "float",
  bf16: "float",
  f32: "float",
};

/** Rank within a family; only compared against members of the same family. */
const RANK: Record<DType, number> = {
  bool: 0,
  i8: 0,
  u8: 0,
  i32: 1,
  i64: 2,
  f8: 0,
  f16: 1,
  bf16: 1,
  f32: 2,
};

/**
 * Each family's types by rank, used only for the equal-rank case.
 *
 * Where a rank holds two incomparable types the entry is one of them, which is
 * never consulted: the ladder is read at `rank + 1`, and a rank reached that
 * way is one the pair does not occupy. Written as a family's own ladder rather
 * than as a constant, because the answer used to be `f32` unconditionally -
 * correct while `f16`/`bf16` was the only such pair, and a silent change of
 * category the moment `i8`/`u8` joined them.
 */
const LADDER: Record<Family, DType[]> = {
  bool: ["bool"],
  int: ["i8", "i32", "i64"],
  float: ["f8", "f16", "f32"],
};

/**
 * Which family a type belongs to.
 *
 * Exported because the lattice's laws are only claimed within a family - width
 * never narrows inside one, while across families the category wins outright
 * and may narrow in bytes - so anything checking those laws has to ask the same
 * question the join asks. A second copy of this table in a test would be free
 * to drift, and a stale one would quietly stop testing the types it did not
 * know about.
 */
export const dtypeFamily = (d: DType): DTypeFamily => FAMILY[d];

/** The narrowest type that holds both, per the lattice documented above. */
export function promoteDType(a: DType, b: DType): DType {
  if (a === b) return a;
  const fa = FAMILY[a];
  const fb = FAMILY[b];
  if (fa === "bool") return b;
  if (fb === "bool") return a;
  if (fa !== fb) return fa === "float" ? a : b;
  // Same family, different type: at equal rank neither operand can serve as the
  // result, so the join is the next width up in that same family.
  if (RANK[a] === RANK[b]) return LADDER[fa][RANK[a] + 1];
  return RANK[a] > RANK[b] ? a : b;
}

/** Fold `promoteDType` over a non-empty list. */
export function promoteDTypes(dtypes: DType[]): DType {
  if (!dtypes.length) throw new Error("promoteDTypes: no dtypes");
  return dtypes.reduce(promoteDType);
}
