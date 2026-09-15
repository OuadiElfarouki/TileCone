import { DType, type DTypeWidening } from "../core/dtypes";

/**
 * Which ONNX tensor element types this engine accepts, and what happens to the
 * rest.
 *
 * Stated once, here, rather than discovered per model. The alternative is an
 * importer that meets `UINT16` on its four-hundredth node and decides something
 * on the spot - and whatever it decides, a reader has no way to know it was
 * decided. Every type below is either mapped to something that means the same
 * thing, or named as unsupported with the reason.
 *
 * The rule is the engine's own: never narrow silently. `INT64` maps to `i64`
 * rather than to `i32` because an index tensor really is eight bytes an
 * element, and every footprint that counted it as four would be understating
 * the model by half. Where no exact equivalent exists, the entry says whether
 * widening is safe (a type our lattice contains) or whether the type is
 * refused, and refusal is a preflight error naming the tensor rather than a
 * guess.
 */

/** ONNX `TensorProto.DataType`, by its own numbering. */
export const ONNX_ELEM_TYPE = {
  UNDEFINED: 0,
  FLOAT: 1,
  UINT8: 2,
  INT8: 3,
  UINT16: 4,
  INT16: 5,
  INT32: 6,
  INT64: 7,
  STRING: 8,
  BOOL: 9,
  FLOAT16: 10,
  DOUBLE: 11,
  UINT32: 12,
  UINT64: 13,
  COMPLEX64: 14,
  COMPLEX128: 15,
  BFLOAT16: 16,
  FLOAT8E4M3FN: 17,
  FLOAT8E4M3FNUZ: 18,
  FLOAT8E5M2: 19,
  FLOAT8E5M2FNUZ: 20,
} as const;

export type OnnxElemType = (typeof ONNX_ELEM_TYPE)[keyof typeof ONNX_ELEM_TYPE];

/**
 * How an accepted type arrives.
 *
 * `exact` is a type we hold as it is. `widened` is one we hold in a larger type
 * because the exact one does not exist here - the values are all representable,
 * so no dependency or shape claim changes, but the byte count is our type's and
 * not the model's, which is a thing a footprint reader is owed.
 */
export type DTypeAcceptance =
  | { status: "exact"; dtype: DType }
  | { status: "widened"; dtype: DType; dtypeWidening: DTypeWidening }
  | { status: "unsupported"; from: string; reason: string };

const exact = (dtype: DType): DTypeAcceptance => ({ status: "exact", dtype });

const widened = (dtype: DType, from: string, note: string): DTypeAcceptance => ({
  status: "widened",
  dtype,
  dtypeWidening: { from, note },
});

const unsupported = (from: string, reason: string): DTypeAcceptance => ({
  status: "unsupported",
  from,
  reason,
});

const ACCEPT: Record<number, DTypeAcceptance> = {
  [ONNX_ELEM_TYPE.FLOAT]: exact("f32"),
  [ONNX_ELEM_TYPE.FLOAT16]: exact("f16"),
  [ONNX_ELEM_TYPE.BFLOAT16]: exact("bf16"),
  [ONNX_ELEM_TYPE.INT64]: exact("i64"),
  [ONNX_ELEM_TYPE.INT32]: exact("i32"),
  [ONNX_ELEM_TYPE.UINT8]: exact("u8"),
  [ONNX_ELEM_TYPE.INT8]: exact("i8"),
  [ONNX_ELEM_TYPE.BOOL]: exact("bool"),

  // The four float8 encodings differ in their exponent bias and in whether they
  // have infinities, none of which this engine can observe: it never reads a
  // value. One byte per element is the whole of what it needs, and `f8` is one
  // byte, so they are exact here in the only sense that applies.
  [ONNX_ELEM_TYPE.FLOAT8E4M3FN]: exact("f8"),
  [ONNX_ELEM_TYPE.FLOAT8E4M3FNUZ]: exact("f8"),
  [ONNX_ELEM_TYPE.FLOAT8E5M2]: exact("f8"),
  [ONNX_ELEM_TYPE.FLOAT8E5M2FNUZ]: exact("f8"),

  // Widened, and marked: the footprint these contribute is our type's, which is
  // larger than the model's. Overstating bytes keeps a total a bound in the
  // direction everything else here already bounds; understating would not.
  [ONNX_ELEM_TYPE.INT16]: widened(
    "i32",
    "int16",
    "held as int32: two bytes per element becomes four, so its footprint is an over-estimate"
  ),
  [ONNX_ELEM_TYPE.UINT16]: widened(
    "i32",
    "uint16",
    "held as int32: two bytes per element becomes four, so its footprint is an over-estimate"
  ),
  [ONNX_ELEM_TYPE.UINT32]: widened(
    "i64",
    "uint32",
    "held as int64, since uint32 exceeds int32's range: four bytes per element becomes eight"
  ),

  // Refused rather than approximated. A uint64 does not fit any integer we
  // have, and rounding its range down would make an index tensor claim a range
  // it cannot hold; a double is eight bytes we would have to call four.
  [ONNX_ELEM_TYPE.UINT64]: unsupported(
    "uint64",
    "no integer here holds its range, and narrowing an index type is not safe"
  ),
  [ONNX_ELEM_TYPE.DOUBLE]: unsupported(
    "double",
    "no 64-bit float here; holding it as f32 would halve its footprint"
  ),
  [ONNX_ELEM_TYPE.COMPLEX64]: unsupported("complex64", "complex values are not modelled"),
  [ONNX_ELEM_TYPE.COMPLEX128]: unsupported("complex128", "complex values are not modelled"),
  [ONNX_ELEM_TYPE.STRING]: unsupported(
    "string",
    "elements have no fixed width, so a region of them has no byte count"
  ),
  [ONNX_ELEM_TYPE.UNDEFINED]: unsupported(
    "undefined",
    "the model states no element type for this tensor"
  ),
};

/** What this engine does with one ONNX element type. */
export function acceptDType(elemType: number): DTypeAcceptance {
  return (
    ACCEPT[elemType] ??
    unsupported(`element type ${elemType}`, "not a type this ONNX version defines here")
  );
}

/** Every ONNX element type that reaches a canonical dtype, exactly or widened. */
export function acceptedElemTypes(): number[] {
  return Object.keys(ACCEPT)
    .map(Number)
    .filter((elemType) => ACCEPT[elemType].status !== "unsupported")
    .sort((a, b) => a - b);
}
