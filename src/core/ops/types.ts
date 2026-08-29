import type { ZodType } from "zod";
import type { Box, Region } from "../region";

export type Attrs = Record<string, unknown>;

export type OpCtx = {
  inShapes: number[][];
  outShapes: number[][];
  attrs: Attrs;
};

export interface OpSpec {
  name: string;
  attrSchema: ZodType<unknown>;
  arity: { inputs: number | "variadic"; outputs: number | "variadic" };

  inferShapes(inShapes: number[][], attrs: Attrs, params?: Record<string, number>): number[][];

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
}

/** Thresholds for emitting enumerated boxes before falling back to inexact bounds. */
export const STRIDED_ENUM_CAP = 512;
export const DIAG_ENUM_CAP = 256;
