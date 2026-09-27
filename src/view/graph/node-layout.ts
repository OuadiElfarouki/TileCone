/** User displacement from a node's generated graph-layout position. */
export type NodeOffset = Readonly<{ dx: number; dy: number }>;

/**
 * Sparse map from a scene node key to its displacement; a node resting at its
 * generated position has no entry.
 *
 * The key is the scene's own node key - `t:<tensor>` for a card, `n:<node>` for
 * an operation - and not a bare id, because both kinds move and the two id
 * spaces are separate: a tensor named `mul` and a `mul` operation are different
 * nodes. DSL identifiers cannot contain `:`, so the prefix is unambiguous.
 */
export type NodeOffsets = Record<string, NodeOffset>;
