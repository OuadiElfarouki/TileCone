import { SymbolicExecutor } from "../core/executor";
import { Graph, GraphError, ResolvedGraph, resolveGraph } from "../core/graph";
import { DSLError, parseDSLWithSource } from "./dsl";
import { DSLSourceMap, SourceSpan } from "./source";

export type DiagnosticPhase = "parse" | "semantic";
export type DiagnosticSeverity = "error";

export type CompilerDiagnostic = {
  severity: DiagnosticSeverity;
  phase: DiagnosticPhase;
  code: string;
  message: string;
  span: SourceSpan;
};

export type CompiledDSL = {
  source: string;
  /** Parsed, unresolved graph; safe to rewrite or serialize. */
  graph: Graph;
  /** Validated graph with inferred shapes and topological metadata. */
  resolved: ResolvedGraph;
  sourceMap: DSLSourceMap;
  executor: SymbolicExecutor;
};

export type CompilationResult =
  | { ok: true; program: CompiledDSL; diagnostics: [] }
  | { ok: false; diagnostics: CompilerDiagnostic[] };

export class CompilationError extends Error {
  constructor(public diagnostics: CompilerDiagnostic[]) {
    super(diagnostics.length ? formatDiagnostic(diagnostics[0]) : "compilation failed");
    this.name = "CompilationError";
  }
}

export function formatDiagnostic(diagnostic: CompilerDiagnostic): string {
  return `line ${diagnostic.span.start.line}: ${diagnostic.message}`;
}

function semanticCode(error: GraphError): string {
  switch (error.code) {
    case "GRAPH_UNKNOWN_OP":
      return "SEM_UNKNOWN_OP";
    case "GRAPH_INVALID_ATTRIBUTES":
      return "SEM_INVALID_ATTRIBUTES";
    case "GRAPH_ARITY":
      return "SEM_ARITY";
    case "GRAPH_CYCLE":
      return "SEM_CYCLE";
    case "GRAPH_SHAPE":
    case "GRAPH_UNBOUND_SYMBOL":
      return "SEM_SHAPE";
    case "GRAPH_DTYPE":
      return "SEM_DTYPE";
    case "GRAPH_DEFINITION":
      return "SEM_DEFINITION";
    case "GRAPH_INVALID":
      return "SEM_INVALID_GRAPH";
  }
}

function semanticSpan(error: GraphError, sourceMap: DSLSourceMap): SourceSpan {
  const subject = error.subject;
  if (subject?.kind === "node" && sourceMap.nodes[subject.id]) return sourceMap.nodes[subject.id];
  if (subject?.kind === "tensor" && sourceMap.tensors[subject.id])
    return sourceMap.tensors[subject.id];
  if (subject?.kind === "parameter" && sourceMap.params[subject.id])
    return sourceMap.params[subject.id];
  return sourceMap.document;
}

/** Compile DSL text into one validated, executable symbolic program. */
export function tryCompileDSL(source: string): CompilationResult {
  let parsed: ReturnType<typeof parseDSLWithSource>;
  try {
    parsed = parseDSLWithSource(source);
  } catch (error) {
    if (error instanceof DSLError)
      return {
        ok: false,
        diagnostics: [
          {
            severity: "error",
            phase: "parse",
            code: error.code,
            message: error.detail,
            span: error.span,
          },
        ],
      };
    throw error;
  }

  let resolved: ResolvedGraph;
  try {
    resolved = resolveGraph(parsed.graph);
  } catch (error) {
    if (error instanceof GraphError) {
      const message = error.message;
      return {
        ok: false,
        diagnostics: [
          {
            severity: "error",
            phase: "semantic",
            code: semanticCode(error),
            message,
            span: semanticSpan(error, parsed.sourceMap),
          },
        ],
      };
    }
    throw error;
  }

  const program: CompiledDSL = {
    source,
    graph: parsed.graph,
    resolved,
    sourceMap: parsed.sourceMap,
    executor: new SymbolicExecutor(resolved),
  };
  return { ok: true, program, diagnostics: [] };
}

/** Throwing convenience for callers that require a valid program. */
export function compileDSL(source: string): CompiledDSL {
  const result = tryCompileDSL(source);
  if (!result.ok) throw new CompilationError(result.diagnostics);
  return result.program;
}
