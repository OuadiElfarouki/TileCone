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
