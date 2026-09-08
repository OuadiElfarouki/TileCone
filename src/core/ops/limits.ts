/**
 * Thresholds at which an operation stops naming a dependency set exactly and
 * falls back to a conservative bound.
 *
 * These are the branches where the engine is allowed to be imprecise, so they
 * are the branches most worth testing - and they were the only ones the oracle
 * never reached. Every cap is large enough that triggering it needs a tensor
 * far beyond what a brute-force oracle can enumerate, which left the safe path
 * covered exhaustively and the unsafe path covered by hand-written examples.
 *
 * Making them a parameter rather than module constants is what closes that: a
 * test can lower them until a four-element tensor takes the fallback, then
 * check the result against brute-force truth like any other case. Production
 * never passes anything and gets `DEFAULT_LIMITS`.
 */
export type Limits = {
  /** Strided slice/conv positions to enumerate before widening to a span. */
  stridedEnum: number;
  /** Diagonal positions to name individually before blocking the staircase. */
  diagEnum: number;
  /** Contiguous runs a reshape will decompose before taking a bounding box. */
  reshapeRuns: number;
  /** Boxes a region may hold before it is coarsened toward the cap. */
  maxBoxes: number;
};

export const DEFAULT_LIMITS: Limits = {
  stridedEnum: 512,
  diagEnum: 256,
  reshapeRuns: 4096,
  maxBoxes: 256,
};

/** The limits in force for a context, which is the defaults unless overridden. */
export function limitsOf(ctx: { limits?: Limits }): Limits {
  return ctx.limits ?? DEFAULT_LIMITS;
}
