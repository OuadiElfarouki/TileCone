/** User displacement from a tensor card's graph-layout position. */
export type TensorOffset = Readonly<{ dx: number; dy: number }>;

/** Sparse map: cards at their base position have no entry. */
export type TensorOffsets = Record<string, TensorOffset>;
