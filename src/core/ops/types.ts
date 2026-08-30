import type { ZodType } from "zod";
import type { Box, Region } from "../region";
import type { Sym } from "../shapes";
import type { DType } from "../dtypes";

export type DependencyNoteDraft = {
  text: string;
  /** Identity of the constraint; equal keys are merged. */
  key: string;
  /** The tensor the note is about, used when merging. */
  subject: string;
};

export type Attrs = Record<string, unknown>;
export type Cardinality = number | { min: number; max?: number };

export type OpCtx = {
  inShapes: number[][];
  outShapes: number[][];
  attrs: Attrs;
};

/**
 * What an operation needs in order to describe, in words, the constraint it puts
 * on a cone that passes through it.
 *
 * Regions are the ones the current query actually produced, not hypothetical
 * ones: a note may only claim what the cone did. `undefined` means that tensor
 * is outside the cone entirely.
 */
export type NoteCtx = OpCtx & {
  inNames: string[];
  outNames: string[];
  /**
   * Declared (possibly symbolic) dimensions per input, parallel to `inShapes`.
   * A note should name an axis the way the source does — a reader who wrote
   * `input A [M, K]` is looking for `K`, not an internal einsum label.
   */
  inDims: Sym[][];
  inRegions: (Region | undefined)[];
  outRegions: (Region | undefined)[];
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
   * A note plus the identity of the constraint it describes. Notes sharing a
   * `key` are the same statement about different tensors — three QKV
   * projections all contracting the same axis — and are merged rather than
   * repeated, so a handful of look-alike notes cannot crowd out a different one.
   */

  /**
   * One sentence naming the dependency constraint this operation imposes on the
   * current cone — the thing a reader would need to know before trying to fuse
   * or tile across it. Return null when the operation constrains nothing worth
   * saying, which is the common case: elementwise work says nothing.
   *
   * This is presentation-adjacent but it belongs to the operation, because only
   * the operation knows *why* its cone has the shape it does. It must describe
   * what the given regions actually contain and never assert a general property
   * the current cone does not exhibit.
   */
  dependencyNote?(ctx: NoteCtx): DependencyNoteDraft | null;

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
