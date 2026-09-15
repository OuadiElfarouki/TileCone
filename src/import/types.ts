import type { Graph } from "../core/graph";
import type { GraphErrorSubject } from "../core/shapes";

/**
 * What an import is, as a value the app holds.
 *
 * The workspace already holds `graph` and `resolved`; until now `applyDSL` was
 * the only way to put anything there, so "where a graph came from" was the DSL
 * text and nothing else. An imported model has no DSL text and must not be
 * given one: the printer is not a lossless serializer for graphs it did not
 * author, so a round trip through text renames the very tensors and nodes a
 * report addresses. Import therefore lands on the same `Graph` by its own path,
 * and carries its provenance in these types rather than in a string somebody
 * prints.
 *
 * The report is the other half of the contract the engine lives by. A region
 * may be a superset of the truth and never a subset, and conversion adds three
 * new ways to break that: an operation nobody modelled, a dependency that never
 * appears in a node's input list, and a conversion that quietly means something
 * else. The first is survivable precisely because it is *named* - a barrier
 * costs precision in a marked place. That marking is what this file types.
 */

/** Which vocabulary a model was converted from. */
export type ImportFormat = "onnx" | "torch" | "json";

export const IMPORT_FORMATS: ImportFormat[] = ["onnx", "torch", "json"];

/**
 * One thing the conversion did that the reader would otherwise have to take on
 * trust.
 *
 * Every entry addresses the *converted* graph by id, not the source file by
 * offset. That is the deliberate answer to the diagnostics gap: a DSL error
 * becomes a source span through the source map, and an import has no source
 * map, so a finding is presented by naming the node and highlighting it on the
 * canvas instead.
 *
 * `sourceOp` and `sourceName` keep the model's own words alongside ours. A
 * report that said only `opaque` would be useless for deciding what to
 * implement next.
 */
export type ImportEntry =
  /** A source node became a canonical operation. */
  | {
      kind: "mapped";
      /** Node id in the converted graph. */
      node: string;
      /** The operation in the source vocabulary, e.g. `Conv`. */
      sourceOp: string;
      /** The registered operation it became, e.g. `conv`. */
      op: string;
      /**
       * Whether the model's own inferred shapes were compared with ours and
       * agreed. False means the model carried no shape to compare, not that a
       * comparison failed: a mismatch degrades the node to a barrier.
       */
      shapeChecked: boolean;
    }
  /**
   * A static argument carried as an initializer became one of our attributes.
   *
   * This is metadata evaluation, not dataflow folding. Per the import contract
   * the dependency relation is over tensor elements *after static graph
   * configuration has been bound*, so a tensor that only says "reshape to this
   * shape" configures the executable graph rather than feeding it. Naming every
   * such evaluation here is what keeps that boundary visible instead of
   * silently erasing dataflow.
   */
  | {
      kind: "evaluated";
      node: string;
      /** Tensor in the source model whose bytes were read. */
      sourceName: string;
      /** The attribute it became, e.g. `shape`. */
      attribute: string;
    }
  /**
   * A source node became several of ours, or ours plus nodes the model does not
   * contain - `Conv` with a bias becomes `conv` + `add`.
   */
  | {
      kind: "rewritten";
      /** Every node id the rewrite produced, in graph order. */
      nodes: string[];
      sourceOp: string;
      /** The registered operations it became, parallel to `nodes`. */
      ops: string[];
    }
  /**
   * A dimension the file left free was given a value.
   *
   * `assumed` separates a binding the file stated from one we chose because a
   * shape has to be concrete before anything can be counted. An assumption the
   * reader cannot see is an assumption they cannot correct.
   */
  | { kind: "bound"; dimension: string; value: number; assumed: boolean }
  /**
   * A source node was carried as a barrier: shapes known, semantics not.
   *
   * Two of these in a report is a fact about precision you can act on. Two
   * silently dropped nodes is a wrong answer you cannot see, which is the
   * failure this whole path is arranged to avoid.
   */
  | { kind: "barrier"; node: string; sourceOp: string; reason: string }
  /**
   * A tensor could not keep the name the model gave it.
   *
   * The original is kept for display, so the reader can still find
   * `/model/layers.0/Add_output_0` on screen after we have given it an id we
   * can use.
   */
  | { kind: "renamed"; tensor: string; original: string };

export type ImportEntryKind = ImportEntry["kind"];

/** What the file said about itself. */
export type ImportOrigin = {
  fileName: string;
  format: ImportFormat;
  /** The effective opset of the default domain, where the format has one. */
  opset?: number;
  /** The tool that wrote the file, when it says. */
  producer?: string;
};

/**
 * Everything the conversion did, in one value.
 *
 * `sourceNodes` and `operations` are stated separately because they genuinely
 * differ: a rewrite inserts nodes the model does not contain, so "71 source
 * nodes -> 72 operations" is the honest header and a single count would hide
 * the insertion.
 */
export type ImportReport = {
  origin: ImportOrigin;
  sourceNodes: number;
  operations: number;
  entries: ImportEntry[];
};

/** A converted model, ready for the same install boundary the DSL uses. */
export type ImportResult = {
  graph: Graph;
  report: ImportReport;
};

/**
 * Why an import could not be installed.
 *
 * Addressed by subject rather than by span, for the reason above. `subject` is
 * absent when the failure is about the file as a whole - malformed JSON, an
 * envelope that is not one - and present when it is about a node or tensor the
 * canvas can highlight.
 */
export type ImportDiagnostic = {
  severity: "error";
  message: string;
  subject?: GraphErrorSubject;
};

export class ImportError extends Error {
  constructor(public diagnostics: ImportDiagnostic[]) {
    super(diagnostics[0]?.message ?? "import failed");
    this.name = "ImportError";
  }
}

/** An empty report for a source that makes no claims about its own conversion. */
export function emptyReport(origin: ImportOrigin, graph: Graph): ImportReport {
  return {
    origin,
    sourceNodes: graph.nodes.length,
    operations: graph.nodes.length,
    entries: [],
  };
}

/** How many entries of each kind, for the header lines of a rendered report. */
export function countByKind(report: ImportReport): Record<ImportEntryKind, number> {
  const counts: Record<ImportEntryKind, number> = {
    mapped: 0,
    evaluated: 0,
    rewritten: 0,
    bound: 0,
    barrier: 0,
    renamed: 0,
  };
  for (const entry of report.entries) counts[entry.kind]++;
  return counts;
}

/**
 * The nodes whose regions are bounds rather than counts.
 *
 * Derived from the entries rather than stored beside them: a barrier is the
 * only thing that produces unknown work, so a second field could disagree with
 * the first. Callers use this to say how many figures report a status instead
 * of a number.
 */
export function barrierNodes(report: ImportReport): string[] {
  return report.entries.flatMap((entry) => (entry.kind === "barrier" ? [entry.node] : []));
}

/** Every converted-graph node an entry addresses, for highlighting on canvas. */
export function entryNodes(entry: ImportEntry): string[] {
  switch (entry.kind) {
    case "mapped":
    case "evaluated":
    case "barrier":
      return [entry.node];
    case "rewritten":
      return entry.nodes;
    case "bound":
    case "renamed":
      return [];
  }
}
