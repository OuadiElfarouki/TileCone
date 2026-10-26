import { SymbolicExecutor } from "../core/executor";
import { Graph, ResolvedGraph, resolveGraphCollecting } from "../core/graph";
import { GraphError, GraphErrorSubject, GraphWarning } from "../core/shapes";
import { lowerProgram } from "./lower";
import { DSLError, parseProgram } from "./parser";
import { documentSpan, DSLSourceMap, SourceSpan } from "./source";
import { inSourceTerms } from "./wording";

export type DiagnosticPhase = "parse" | "semantic";
/** An error refuses the program; a warning comes with a program that built. */
export type DiagnosticSeverity = "error" | "warning";

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
  /** Warnings about the program as built, in source order. */
  warnings: CompilerDiagnostic[];
};

export type CompilationResult =
  /** `diagnostics` holds the program's warnings, and only warnings. */
  | { ok: true; program: CompiledDSL; diagnostics: CompilerDiagnostic[] }
  | { ok: false; diagnostics: CompilerDiagnostic[] };

export class CompilationError extends Error {
  constructor(public diagnostics: CompilerDiagnostic[]) {
    super(diagnostics.length ? formatDiagnostic(diagnostics[0]) : "compilation failed");
    this.name = "CompilationError";
  }
}

function formatDiagnostic(diagnostic: CompilerDiagnostic): string {
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

/**
 * The narrowest span that is still certainly about this error.
 *
 * A statement span is the fallback, not the goal: when the message names an
 * attribute the author wrote, underline that attribute instead of the line.
 * An unknown *op* underlines the call name for the same reason.
 */
function semanticSpan(
  error: { code: string; subject?: GraphErrorSubject },
  sourceMap: DSLSourceMap
): SourceSpan {
  const subject = error.subject;
  if (subject?.kind === "node" && sourceMap.nodes[subject.id]) {
    const args = sourceMap.nodeArgs[subject.id];
    if (args) {
      if (error.code === "GRAPH_UNKNOWN_OP") return args.callee;
      // The attribute the error names, when it names one and the author wrote
      // it: a defaulted attribute has no span because it appears in no text.
      const attr = subject.attribute;
      if (attr && args.attrs[attr]) return args.attrs[attr];
      if (subject.input !== undefined && args.inputs[subject.input]) return args.inputs[subject.input];
    }
    return sourceMap.nodes[subject.id];
  }
  if (subject?.kind === "tensor" && sourceMap.tensors[subject.id])
    return sourceMap.tensors[subject.id];
  if (subject?.kind === "parameter" && sourceMap.params[subject.id])
    return sourceMap.params[subject.id];
  return sourceMap.document;
}

/** A resolver message in the source's terms where it is about a node the
 * source wrote, and as the resolver put it otherwise. */
const messageOf = (
  problem: { message: string; detail?: string; subject?: GraphErrorSubject },
  sourceMap: DSLSourceMap,
  graph: Graph
): string =>
  (problem.detail !== undefined && inSourceTerms(problem.detail, problem.subject, sourceMap, graph)) ||
  problem.message;

const warningDiagnostic = (
  warning: GraphWarning,
  sourceMap: DSLSourceMap,
  graph: Graph
): CompilerDiagnostic => ({
  severity: "warning",
  phase: "semantic",
  code: "SEM_SYMBOLIC_SHAPE",
  message: messageOf(warning, sourceMap, graph),
  span: semanticSpan(warning, sourceMap),
});

const asDiagnostic = (error: DSLError, phase: DiagnosticPhase): CompilerDiagnostic => ({
  severity: "error",
  phase,
  code: error.code,
  message: error.detail,
  span: error.span,
});

/** Diagnostics in the order the author reads them, not the order phases ran. */
function inSourceOrder(diagnostics: CompilerDiagnostic[]): CompilerDiagnostic[] {
  return [...diagnostics].sort(
    (a, b) => a.span.start.line - b.span.start.line || a.span.start.column - b.span.start.column
  );
}

/**
 * Compile DSL text into one validated, executable symbolic program.
 *
 * Every phase reports everything it found. Parse errors are per line, lowering
 * errors are per statement, and resolution errors are per node or tensor, so a
 * document with four unrelated mistakes returns four diagnostics rather than
 * making the author fix and recompile four times.
 *
 * Later phases still run after an earlier one failed, on whatever the earlier
 * phase could make sense of. That is what turns "one error per compile" into
 * "one pass per compile": a bad line and a shape mismatch three lines down are
 * independent facts, and the author wants both.
 */
function tryCompileDSLUnchecked(source: string): CompilationResult {
  const { program: ast, errors: parseErrors } = parseProgram(source);
  const lowered = lowerProgram(ast);
  const diagnostics: CompilerDiagnostic[] = [
    ...parseErrors.map((e) => asDiagnostic(e, "parse")),
    ...lowered.errors.map((e) => asDiagnostic(e, "parse")),
  ];

  // Resolution runs on what lowering produced even when lowering failed: the
  // surviving statements are still worth checking, and their errors are real.
  const { resolved, errors: graphErrors } = resolveGraphCollecting(lowered.graph);
  for (const error of graphErrors)
    diagnostics.push({
      severity: "error",
      phase: "semantic",
      code: semanticCode(error),
      message: messageOf(error, lowered.sourceMap, lowered.graph),
      span: semanticSpan(error, lowered.sourceMap),
    });

  if (diagnostics.length || !resolved)
    return {
      ok: false,
      diagnostics: inSourceOrder(
        diagnostics.length
          ? diagnostics
          : [
              {
                severity: "error",
                phase: "semantic",
                code: "SEM_INVALID_GRAPH",
                message: "graph could not be resolved",
                span: lowered.sourceMap.document,
              },
            ]
      ),
    };

  const warnings = inSourceOrder(
    resolved.warnings.map((warning) => warningDiagnostic(warning, lowered.sourceMap, lowered.graph))
  );
  const program: CompiledDSL = {
    source,
    graph: lowered.graph,
    resolved,
    sourceMap: lowered.sourceMap,
    executor: new SymbolicExecutor(resolved),
    warnings,
  };
  return { ok: true, program, diagnostics: warnings };
}

/** The editor feeds this boundary untrusted text on every apply. Parser and
 * resolver defects must become a diagnostic, never tear down the workspace. */
export function tryCompileDSL(source: string): CompilationResult {
  try {
    return tryCompileDSLUnchecked(source);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      diagnostics: [{
        severity: "error",
        phase: "semantic",
        code: "SEM_INTERNAL",
        message: `compiler could not safely process this input: ${detail}`,
        span: documentSpan(source),
      }],
    };
  }
}

/** Throwing convenience for callers that require a valid program. */
export function compileDSL(source: string): CompiledDSL {
  const result = tryCompileDSL(source);
  if (!result.ok) throw new CompilationError(result.diagnostics);
  return result.program;
}
