import type { ZodType } from "zod";
import type { Box, Region } from "../region";
import type { DType } from "../dtypes";

export type Attrs = Record<string, unknown>;
export type Cardinality = number | { min: number; max?: number };

export type OpCtx = {
  inShapes: number[][];
  outShapes: number[][];
  attrs: Attrs;
};

export interface OpSpec {
  name: string;
  attrSchema: ZodType<unknown>;
  arity: { inputs: Cardinality; outputs: Cardinality };

  /** Validate counts that depend on parsed attributes rather than static ranges. */
  validateArity?(inputCount: number, outputCount: number, attrs: Attrs): void;

  inferShapes(inShapes: number[][], attrs: Attrs, params?: Record<string, number>): number[][];

  /** Infer canonical output dtypes and validate input dtype compatibility. */
  inferDTypes(inDTypes: DType[], attrs: Attrs, outShapes: number[][]): DType[];

  /** Which parts of each input does this single output box depend on? One Region per input. */
  backward(outSlot: number, outBox: Box, ctx: OpCtx): Region[];

  /** Which parts of each output does this single input box influence? One Region per output. */
  forward(inSlot: number, inBox: Box, ctx: OpCtx): Region[];

  /**
   * Ground-truth pointwise semantics for the brute-force oracle: the exact input
   * elements (per input slot) that one output element reads. Must be derived from
   * the op's mathematical definition, not from `backward`.
   */
  oracleDeps(outSlot: number, outIndex: number[], ctx: OpCtx): number[][][];

  /** Approximate FLOPs to compute the given output box. Data movement ops return 0. */
  flopsFor(outSlot: number, outBox: Box, ctx: OpCtx): number;

  /**
   * Region-aware override for operations whose outputs share nonlocal work
   * (reductions behind softmax/normalization, scans, and similar kernels).
   * When absent, metrics sum `flopsFor` over the region's disjoint boxes.
   */
  flopsForRegion?(outSlot: number, outRegion: Region, ctx: OpCtx): number;
}

/** Thresholds for emitting enumerated boxes before falling back to inexact bounds. */
export const STRIDED_ENUM_CAP = 512;
export const DIAG_ENUM_CAP = 256;

/** Require all inputs to share one dtype and apply it to every inferred output. */
export function uniformDTypeOutputs(op: string) {
  return (inDTypes: DType[], _attrs: Attrs, outShapes: number[][]): DType[] => {
    const dtype = inDTypes[0];
    if (!dtype) throw new Error(`${op}: expected at least one input dtype`);
    const mismatch = inDTypes.find((candidate) => candidate !== dtype);
    if (mismatch)
      throw new Error(`${op}: input dtypes must match, got [${inDTypes.join(", ")}]`);
    return outShapes.map(() => dtype);
  };
}
