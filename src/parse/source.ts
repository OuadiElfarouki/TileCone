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

export type DSLSourceMap = {
  document: SourceSpan;
  params: Record<string, SourceSpan>;
  tensors: Record<string, SourceSpan>;
  nodes: Record<string, SourceSpan>;
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
