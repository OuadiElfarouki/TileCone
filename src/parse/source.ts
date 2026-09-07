/** Zero-based offsets plus one-based line/column coordinates for diagnostics. */
export type SourcePosition = {
  offset: number;
  line: number;
  column: number;
};

/** Half-open source span: start is inclusive, end is exclusive. */
export type SourceSpan = {
  start: SourcePosition;
  end: SourcePosition;
};

/**
 * Where a node's individual arguments were written.
 *
 * A statement span is enough to say "this line is wrong", which is all the
 * source map used to offer. Semantic errors usually know more than that: a bad
 * attribute knows which attribute, a dtype mismatch knows which operand. These
 * let a diagnostic underline the part it is actually about.
 */
export type NodeArgSpans = {
  /** One span per input, parallel to the node's `inputs`. */
  inputs: SourceSpan[];
  /** Span of each named attribute as written, keyed by attribute name. */
  attrs: Record<string, SourceSpan>;
  /** The call name itself. */
  callee: SourceSpan;
};

export type DSLSourceMap = {
  document: SourceSpan;
  params: Record<string, SourceSpan>;
  tensors: Record<string, SourceSpan>;
  nodes: Record<string, SourceSpan>;
  /** Per-argument spans, keyed by node id. Absent for a node built by tooling
   * rather than parsed from text. */
  nodeArgs: Record<string, NodeArgSpans>;
};

export function lineSpan(
  line: number,
  lineOffset: number,
  column: number,
  length: number
): SourceSpan {
  return {
    start: { offset: lineOffset + column - 1, line, column },
    end: { offset: lineOffset + column - 1 + length, line, column: column + length },
  };
}

export function documentSpan(source: string): SourceSpan {
  const lines = source.split("\n");
  const lastLine = lines.length;
  const lastColumn = lines[lines.length - 1].length + 1;
  return {
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: source.length, line: lastLine, column: lastColumn },
  };
}
